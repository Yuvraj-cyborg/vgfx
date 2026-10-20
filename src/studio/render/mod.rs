//! Draws a frame on the CPU, no GPU, on one thread or several.
//!
//! The order of work, and where each part lives:
//!
//! 1. `light`   the key lamp's shadow map, from the faces turned away from it.
//! 2. `raster`  which triangle is in front at four samples in every pixel.
//! 3. here      each pixel's samples gathered into surfaces: one shading per
//!              surface, however many samples it covers.
//! 4. `floor`   the scene again upside down at half size, for the floor's
//!              mirror image; then the floor itself, per pixel.
//! 5. `post`    how open each point is to the room, at half size.
//! 6. `shade`   lamps, shadow, the room's reflection, for every surface.
//! 7. `post`    bloom, exposure, tone curve, vignette, dither, sRGB.
//!
//! On the way to a pixel only +, -, *, /, sqrt, floor, abs, comparisons and
//! integer work are used, so the bytes are the same on every machine.
//!
//! Each step after the setup is done a band of rows at a time (a row of
//! tiles), and a band writes only its own rows and works them out as the
//! whole picture would: the bands can go to any number of threads (`par`)
//! and the bytes stay the same.

use super::math::V3;
use super::scene::{Frame, Material, Mesh};

pub mod floor;
pub mod light;
mod par;
pub mod post;
pub mod raster;
pub mod shade;

pub use raster::NS;
use raster::{Frag, Pass, Setup, Tiles, Tri};
use shade::{Ctx, Inst, Lamp, Mat};

#[inline(always)]
pub(crate) fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub(crate) fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[inline(always)]
pub(crate) fn clamp01(x: f32) -> f32 {
    fmin(fmax(x, 0.0), 1.0)
}

/// A colour squeezed below 1 and weighted, so that a very bright surface
/// that covers a quarter of a pixel does not swamp the other three quarters.
/// The finish undoes it.
#[inline(always)]
pub(crate) fn compress(c: [f32; 3], wt: f32) -> [f32; 3] {
    let k = wt / (1.0 + fmax(fmax(c[0], c[1]), fmax(c[2], 0.0)));
    [
        fmax(c[0], 0.0) * k,
        fmax(c[1], 0.0) * k,
        fmax(c[2], 0.0) * k,
    ]
}

/// Undoes `compress` for a pixel whose every part is in, and applies the
/// exposure.
#[inline(always)]
pub(crate) fn expose(st: [f32; 3], exposure: f32) -> [f32; 3] {
    let m = fmax(st[0], fmax(st[1], st[2]));
    if m <= 0.0 {
        return st;
    }
    let k = exposure / (1.0 - fmin(m, 0.999));
    [st[0] * k, st[1] * k, st[2] * k]
}

/// The stages `Renderer::ms` reports, in order.
pub const STAGES: [&str; 7] = [
    "transform and setup",
    "shadow map",
    "visibility",
    "floor and mirror",
    "ambient occlusion",
    "shading",
    "bloom and finish",
];

pub struct Renderer {
    pub w: usize,
    pub h: usize,
    out: Vec<u8>,
    /// Each sample's depth and triangle: for each row of pixels, a row of
    /// each sample in turn (see `raster::KeyBand`).
    keys: Vec<u64>,
    tris: Vec<Tri>,
    hv: Vec<[f32; 4]>,
    vn: Vec<V3>,
    vbase: Vec<u32>,
    /// The picture's triangles, by band, and each band's surfaces.
    set: Setup,
    frags: Vec<Vec<Frag>>,
    open: Vec<u8>,
    /// Which 32-pixel tiles of the picture hold any mesh, and which may hold
    /// anything bright enough to bloom.
    tiles: Tiles,
    bright: Vec<u8>,
    hdr: Vec<[f32; 3]>,
    mats: Vec<Mat>,
    insts: Vec<Inst>,
    light: light::Light,
    mirror: floor::Mirror,
    ao: post::Ao,
    post: post::Post,
    e2: [f32; 65],
    cent: [(f32, f32); 16],
    /// Set to have `ms` filled in. Never changes a pixel.
    pub timing: bool,
    /// Milliseconds spent in each of `STAGES` by the last `render`.
    pub ms: [f64; 7],
    /// Triangles drawn, surfaces shaded, and triangles cut by the near
    /// plane, in the last `render`.
    pub counts: [usize; 3],
    /// Opt-in reuse of the key lamp's shadow map. `Some(n)` is the caller's
    /// promise that every `render` given the same `n` is given the very same
    /// meshes, unchanged. A frame whose number is the last frame's, and
    /// whose key lamp (place, aim, size) and instances (mesh and transform)
    /// are that frame's to the bit, keeps that frame's map instead of
    /// drawing it again: in a film where only the camera moves, the map is
    /// drawn once. The picture is the same either way. `None`, the default,
    /// draws the map every frame. Whether meshes changed is never guessed,
    /// so a caller that changes them must give a new number.
    pub mesh_generation: Option<u64>,
    /// Whether the last `render` kept the shadow map of the frame before.
    pub shadow_kept: bool,
    /// The number, and the lamp and instance bits, the map was drawn for.
    shadow_for: (Option<u64>, Vec<u32>),
    shadow_now: Vec<u32>,
    /// How many threads one `render` may use; 1, the default, draws on the
    /// calling thread alone. The work of a frame is cut into bands of rows
    /// that each write only their own rows, and each band is worked out as
    /// one thread would, so the picture is the same to the bit for any
    /// number. Threads are started for each frame and gone when `render`
    /// returns. In WebAssembly a frame is always drawn on one thread.
    pub threads: usize,
    /// Each thread's scratch rows: for the floor, and for the finish.
    rows_floor: Vec<Vec<[f32; floor::CH]>>,
    rows_tone: Vec<Vec<[f32; 3]>>,
    /// Runs of instances set up apart when several threads share the work.
    spans: Vec<core::ops::Range<usize>>,
    parts: Vec<raster::Part<Tri>>,
}

struct Clock(#[cfg(not(target_arch = "wasm32"))] Option<std::time::Instant>);

impl Clock {
    fn new(_on: bool) -> Clock {
        #[cfg(not(target_arch = "wasm32"))]
        return Clock(if _on {
            Some(std::time::Instant::now())
        } else {
            None
        });
        #[cfg(target_arch = "wasm32")]
        Clock()
    }
    fn lap(&mut self, _slot: &mut f64) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t) = self.0 {
            *_slot = t.elapsed().as_secs_f64() * 1000.0;
            self.0 = Some(std::time::Instant::now());
        }
    }
}

impl Renderer {
    pub fn new(w: usize, h: usize) -> Renderer {
        let (w, h) = (w.max(1), h.max(1));
        let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
        let cent = raster::cent_table();
        let tiles = Tiles::new(w, h, 5);
        Renderer {
            w,
            h,
            out: vec![0; w * h * 3],
            keys: vec![0; w * h * NS],
            tris: Vec::new(),
            hv: Vec::new(),
            vn: Vec::new(),
            vbase: Vec::new(),
            set: Setup::default(),
            frags: (0..tiles.th).map(|_| Vec::new()).collect(),
            open: vec![0; w * h],
            tiles,
            bright: vec![0; ((w + 31) >> 5) * ((h + 31) >> 5)],
            hdr: vec![[0.0; 3]; w * h],
            mats: Vec::new(),
            insts: Vec::new(),
            light: light::Light::new(),
            mirror: floor::Mirror::new(w2, h2),
            ao: post::Ao::new(w2, h2),
            post: post::Post::new(w, h),
            e2: shade::exp2_table(),
            cent,
            timing: false,
            ms: [0.0; 7],
            counts: [0; 3],
            mesh_generation: None,
            shadow_kept: false,
            shadow_for: (None, Vec::new()),
            shadow_now: Vec::new(),
            threads: 1,
            rows_floor: Vec::new(),
            rows_tone: Vec::new(),
            spans: Vec::new(),
            parts: Vec::new(),
        }
    }

    /// Bytes held between frames.
    pub fn bytes(&self) -> usize {
        self.out.len()
            + self.keys.len() * 8
            + (self.tris.capacity() + self.mirror.tris.capacity()) * core::mem::size_of::<Tri>()
            + self.hv.capacity() * 16
            + self.vn.capacity() * 12
            + self.set.bytes()
            + self.frags.iter().map(|f| f.capacity() * 12).sum::<usize>()
            + self.open.len()
            + self.hdr.len() * 12
            + self.light.bytes()
            + self.mirror.bytes()
            + self.ao.w * self.ao.h * 12
            + self.post.bytes()
            + (self.shadow_for.1.capacity() + self.shadow_now.capacity()) * 4
    }

    /// Draws `frame` and returns the picture: `w * h * 3` bytes, red green
    /// blue, rows from the top. A pure function of its arguments, on as
    /// many threads as `threads` says.
    pub fn render(&mut self, meshes: &[Mesh], materials: &[Material], frame: &Frame) -> &[u8] {
        let mut clock = Clock::new(self.timing);
        let (w, h) = (self.w, self.h);
        let threads = self.threads.clamp(1, par::MAX_THREADS);
        self.rows_floor.resize_with(threads, Vec::new);
        self.rows_tone.resize_with(threads, Vec::new);

        // ------------------------------------------------------------ setup
        raster::normals_with(
            meshes,
            frame,
            &mut self.vbase,
            &mut self.vn,
            threads,
            &mut self.spans,
        );
        self.mats.clear();
        self.mats.extend(materials.iter().map(Mat::of));
        // An instance that names a material that is not there is drawn grey.
        self.mats
            .push(Mat::of(&Material::paint([0.5, 0.5, 0.5], 0.6)));
        let last = self.mats.len() as u32 - 1;
        self.insts.clear();
        self.insts.extend(frame.instances.iter().map(|i| Inst {
            mat: i.material.min(last),
            glow: i.glow,
        }));

        let cam = &frame.camera;
        let fwd = (cam.target - cam.eye).norm();
        let mut right = fwd.cross(cam.up).norm();
        if right.dot(right) < 0.5 {
            right = fwd.cross(V3::Z).norm();
        }
        let up = right.cross(fwd);
        let tan = if cam.tan_half_fov_y > 1e-4 {
            cam.tan_half_fov_y
        } else {
            0.4
        };
        let fpx = h as f32 * 0.5 / tan;
        let near = fmax(cam.near, 1e-3);
        let eye = cam.eye;
        // World to pixels, for the picture and for the half-size picture of
        // the world turned upside down (y negated).
        let rows = |fp: f32, cx: f32, cy: f32, flip: f32| {
            let rw = [fwd.x, fwd.y * flip, fwd.z, -fwd.dot(eye)];
            let rx = [
                right.x * fp + rw[0] * cx,
                right.y * flip * fp + rw[1] * cx,
                right.z * fp + rw[2] * cx,
                -right.dot(eye) * fp + rw[3] * cx,
            ];
            let ry = [
                -up.x * fp + rw[0] * cy,
                -up.y * flip * fp + rw[1] * cy,
                -up.z * fp + rw[2] * cy,
                up.dot(eye) * fp + rw[3] * cy,
            ];
            [rx, ry, rw]
        };
        let lamp = |l: &super::scene::Light| Lamp {
            pos: l.pos,
            col: l.color,
            size: fmax(l.size, 1e-3),
        };
        let mut lamps = [lamp(&frame.key); 4];
        let nl = 1 + frame.fills.len().min(3);
        for (i, f) in frame.fills.iter().take(3).enumerate() {
            lamps[i + 1] = lamp(f);
        }
        clock.lap(&mut self.ms[0]);

        // ------------------------------------------------------ shadow map
        // Everything the map is made of besides the meshes, as bits.
        let now = &mut self.shadow_now;
        now.clear();
        let k = &frame.key;
        now.extend([k.pos.x, k.pos.y, k.pos.z, k.at.x, k.at.y, k.at.z, k.size].map(f32::to_bits));
        for i in &frame.instances {
            now.push(i.mesh);
            now.extend(i.xf.m.map(f32::to_bits));
        }
        self.shadow_kept = self.mesh_generation.is_some()
            && self.shadow_for.0 == self.mesh_generation
            && self.shadow_for.1 == self.shadow_now;
        if !self.shadow_kept {
            self.light.draw_with(meshes, frame, &self.vbase, threads);
            self.shadow_for.0 = self.mesh_generation;
            core::mem::swap(&mut self.shadow_for.1, &mut self.shadow_now);
        }
        clock.lap(&mut self.ms[1]);

        // ------------------------------------------------------ visibility
        let pass = Pass {
            m: rows(fpx, w as f32 * 0.5, h as f32 * 0.5, 1.0),
            near,
            w: w as i32,
            h: h as i32,
            front: -1.0,
        };
        let sh = self.tiles.sh;
        let cut = raster::setup_with(
            &pass,
            meshes,
            frame,
            &self.vbase,
            &self.vn,
            &mut self.hv,
            &mut self.tris,
            &mut self.set,
            sh,
            threads,
            &mut self.parts,
            &mut self.spans,
        );
        {
            // Each band of rows drawn, its tiles' marks cleared first, then
            // gathered into surfaces.
            let (tris, set) = (&self.tris, &self.set);
            let mut open = self.open.chunks_mut(w << sh);
            let mut frags = self.frags.iter_mut();
            let bands = raster::key_bands(&mut self.keys, w, h, &mut self.tiles)
                .map(move |(k, band)| (k, band, open.next().unwrap(), frags.next().unwrap()));
            // The bands with the most triangles first.
            let cost = |b: &(usize, _, _, _)| set.bins[b.0].len();
            par::each0_by_cost(threads, bands, cost, |(k, mut band, open, frags)| {
                band.marks.fill(0);
                raster::draw_band(&pass, sh, set, k, &mut band);
                frags.clear();
                raster::resolve_band(&band, w, sh, tris, frags, open);
            });
        }
        let surfaces = self.frags.iter().map(Vec::len).sum();
        self.counts = [self.tris.len(), surfaces, cut];
        clock.lap(&mut self.ms[2]);

        // ------------------------------------------------ floor and mirror
        let (mw, mh) = (self.mirror.w, self.mirror.h);
        let mfp = fpx * 0.5;
        let flip = |v: V3| V3 {
            x: v.x,
            y: -v.y,
            z: v.z,
        };
        let ray = |fp: f32, cx: f32, cy: f32| {
            (
                fwd - right * (cx / fp) + up * (cy / fp),
                right * (1.0 / fp),
                up * (-1.0 / fp),
            )
        };
        let (m0, mdx, mdy) = ray(mfp, w as f32 * 0.25, h as f32 * 0.25);
        let (r0, rdx, rdy) = ray(fpx, w as f32 * 0.5, h as f32 * 0.5);
        let mpass = Pass {
            m: rows(mfp, w as f32 * 0.25, h as f32 * 0.25, -1.0),
            near,
            w: mw as i32,
            h: mh as i32,
            front: 1.0,
        };
        if eye.y > 0.0 && frame.floor.mirror > 0.0 {
            raster::setup_with(
                &mpass,
                meshes,
                frame,
                &self.vbase,
                &self.vn,
                &mut self.hv,
                &mut self.mirror.tris,
                &mut self.mirror.set,
                self.mirror.tiles.sh,
                threads,
                &mut self.parts,
                &mut self.spans,
            );
        } else {
            // Nothing is mirrored: every band is left empty.
            self.mirror.set.reset(self.mirror.tiles.th);
        }
        let base = Ctx {
            eye,
            ray0: r0,
            rdx,
            rdy,
            lamps,
            nl,
            amb: frame.ambient,
            floor_base: frame.floor.base,
            floor_mirror: clamp01(frame.floor.mirror),
            floor_rough: fmin(fmax(frame.floor.rough, 0.045), 1.0),
            light: &self.light,
            vn: &self.vn,
            mats: &self.mats,
            insts: &self.insts,
            e2: &self.e2,
        };
        let under = Ctx {
            eye: flip(eye),
            ray0: flip(m0),
            rdx: flip(mdx),
            rdy: flip(mdy),
            ..base
        };
        self.mirror
            .draw_and_shade(Some(&mpass), &under, &self.cent, threads);
        self.bright.copy_from_slice(&self.tiles.f);
        self.mirror.floor_with(
            &base,
            &frame.floor,
            w,
            h,
            NS,
            &self.open,
            &self.tiles,
            mfp,
            frame.exposure,
            &mut self.hdr,
            &mut self.bright,
            threads,
            &mut self.rows_floor,
        );
        clock.lap(&mut self.ms[3]);

        // ----------------------------------------------- ambient occlusion
        // The floor's 1/depth as a plane over the screen, so that feet are
        // closed in by the floor they stand on.
        let fq = if eye.y > 0.0 {
            [-r0.y / eye.y, -rdx.y / eye.y, -rdy.y / eye.y]
        } else {
            [0.0; 3]
        };
        // Each row of the picture's keys holds a row of each sample in turn.
        self.ao.compute_with(
            &self.keys,
            NS * w,
            &self.tris,
            &self.tiles,
            w,
            h,
            right,
            up,
            fwd,
            fpx,
            fq,
            threads,
        );
        clock.lap(&mut self.ms[4]);

        // --------------------------------------------------------- shading
        // A band's surfaces are all in its own rows.
        let inv_ns = 1.0 / NS as f32;
        {
            let (tris, cent, ao) = (&self.tris, &self.cent, &self.ao);
            let bands = self.hdr.chunks_mut(w << sh).zip(&self.frags).enumerate();
            let cost = |b: &(usize, (_, &Vec<Frag>))| b.1.1.len();
            par::each0_by_cost(threads, bands, cost, |(k, (hdr, frags))| {
                let at = (k << sh) * w;
                for f in frags {
                    let t = &tris[f.tri as usize];
                    let (x, y) = (f.px as usize % w, f.px as usize / w);
                    let (ox, oy) = cent[(f.bits >> 4) as usize & 15];
                    let sf = shade::surf(&base, t, x as f32 + ox, y as f32 + oy);
                    let ao = ao.at(x, y, sf.w);
                    let c = compress(
                        shade::shade(&base, &sf, t, ao),
                        (f.bits & 15) as f32 * inv_ns,
                    );
                    let o = &mut hdr[f.px as usize - at];
                    o[0] += c[0];
                    o[1] += c[1];
                    o[2] += c[2];
                }
            });
        }
        clock.lap(&mut self.ms[5]);

        // ---------------------------------------------------------- finish
        let has_bloom = self.post.gather_all(
            &mut self.hdr,
            w,
            frame.exposure,
            frame.bloom,
            &self.open,
            &self.tiles,
            &self.bright,
            NS,
            threads,
        );
        if has_bloom {
            self.post.spread(frame.bloom, threads);
        }
        self.post.columns(w, h);
        self.post.tone_all(
            &self.hdr,
            w,
            h,
            sh,
            has_bloom,
            &mut self.out,
            threads,
            &mut self.rows_tone,
        );
        clock.lap(&mut self.ms[6]);
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::studio::math::{M4, cos, sin, v3};
    use crate::studio::scene::Instance;
    use crate::studio::shapes::{ball, bbox};

    fn meshes() -> Vec<Mesh> {
        vec![
            bbox(v3(0.0, 0.5, 0.0), v3(0.5, 0.5, 0.5), 0.06),
            ball(v3(0.0, 0.4, 0.0), 0.4, 1.0, 24),
        ]
    }

    fn materials() -> Vec<Material> {
        vec![
            Material::metal([0.9, 0.9, 0.92], 0.2),
            Material::paint([0.7, 0.2, 0.1], 0.5),
            Material::lamp([6.0, 5.0, 3.0]),
        ]
    }

    /// A small scene at moment `k`: the camera goes round; from moment 4
    /// the lamp has moved, and from moment 6 a part as well.
    fn frame(k: usize) -> Frame {
        let a = k as f32 * 0.4;
        let mut f = Frame::default();
        f.camera.eye = v3(6.0 * sin(a), 2.2, 6.0 * cos(a));
        f.camera.target = v3(0.0, 0.6, 0.0);
        f.fills.push(f.key);
        f.fills[0].pos = v3(-5.0, 3.0, 2.0);
        let lift = if k >= 6 { 0.3 } else { 0.0 };
        let at = |mesh, material, p| Instance {
            mesh,
            material,
            xf: M4::translate(p),
            glow: 1.0,
        };
        f.instances = vec![
            at(0, 0, v3(-0.8, lift, 0.0)),
            at(1, 1, v3(0.9, 0.0, 0.3)),
            at(1, 2, v3(0.0, 1.2, -0.9)),
        ];
        if k >= 4 {
            f.key.pos.x += 0.5;
        }
        f
    }

    #[test]
    fn a_kept_shadow_map_gives_the_same_picture() {
        let (ms, mats) = (meshes(), materials());
        let mut kept = Renderer::new(96, 64);
        kept.mesh_generation = Some(1);
        let mut kept_any = false;
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 2, 2] {
            let f = frame(k);
            let fresh = Renderer::new(96, 64).render(&ms, &mats, &f).to_vec();
            assert_eq!(kept.render(&ms, &mats, &f), &fresh[..], "moment {k}");
            kept_any |= kept.shadow_kept;
            // The lamp moves at 4 and a part at 6: the map is drawn again.
            if k == 4 || k == 6 {
                assert!(!kept.shadow_kept);
            }
        }
        assert!(kept_any);
        // Other meshes under another number are never given the old map.
        let other = vec![ms[1].clone(), ms[0].clone()];
        kept.mesh_generation = Some(2);
        let f = frame(2);
        let fresh = Renderer::new(96, 64).render(&other, &mats, &f).to_vec();
        assert_eq!(kept.render(&other, &mats, &f), &fresh[..]);
        assert!(!kept.shadow_kept);
        // Without a number the map is always drawn.
        kept.mesh_generation = None;
        kept.render(&other, &mats, &f);
        kept.render(&other, &mats, &f);
        assert!(!kept.shadow_kept);
    }

    #[test]
    fn any_number_of_threads_draws_the_same_picture() {
        let (ms, mats) = (meshes(), materials());
        // Sizes that end in part of a band, and one band high.
        for (w, h) in [(203, 117), (64, 31)] {
            let mut one = Renderer::new(w, h);
            let mut many: Vec<Renderer> = [2, 3, 8]
                .map(|n| {
                    let mut r = Renderer::new(w, h);
                    r.threads = n;
                    r
                })
                .into();
            for k in [0, 3, 6, 9] {
                let mut f = frame(k);
                if k == 9 {
                    // From under the floor: nothing mirrored, no floor.
                    f.camera.eye.y = -0.5;
                }
                let want = one.render(&ms, &mats, &f).to_vec();
                assert!(want.iter().any(|&b| b > 60), "the scene is in the picture");
                for r in &mut many {
                    let n = r.threads;
                    let got = r.render(&ms, &mats, &f);
                    assert_eq!(got, &want[..], "{w}x{h} moment {k} on {n} threads");
                }
            }
        }
    }
}
