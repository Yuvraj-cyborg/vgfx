//! Draws a frame on the CPU, one thread, no GPU.
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

use super::math::V3;
use super::scene::{Frame, Material, Mesh};

pub mod floor;
pub mod light;
pub mod post;
pub mod raster;
pub mod shade;

pub use raster::NS;
use raster::{Frag, OFFS, Pass, Tiles, Tri};
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
    keys: Vec<u64>,
    tris: Vec<Tri>,
    hv: Vec<[f32; 4]>,
    vn: Vec<V3>,
    vbase: Vec<u32>,
    frags: Vec<Frag>,
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
        Renderer {
            w,
            h,
            out: vec![0; w * h * 3],
            keys: vec![0; w * h * NS],
            tris: Vec::new(),
            hv: Vec::new(),
            vn: Vec::new(),
            vbase: Vec::new(),
            frags: Vec::new(),
            open: vec![0; w * h],
            tiles: Tiles::new(w, h, 5),
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
        }
    }

    /// Bytes held between frames.
    pub fn bytes(&self) -> usize {
        self.out.len()
            + self.keys.len() * 8
            + (self.tris.capacity() + self.mirror.tris.capacity()) * core::mem::size_of::<Tri>()
            + self.hv.capacity() * 16
            + self.vn.capacity() * 12
            + self.frags.capacity() * 12
            + self.open.len()
            + self.hdr.len() * 12
            + self.light.bytes()
            + self.mirror.bytes()
            + self.ao.w * self.ao.h * 12
            + self.post.bytes()
    }

    /// Draws `frame` and returns the picture: `w * h * 3` bytes, red green
    /// blue, rows from the top. One call uses one thread and is a pure
    /// function of its arguments.
    pub fn render(&mut self, meshes: &[Mesh], materials: &[Material], frame: &Frame) -> &[u8] {
        let mut clock = Clock::new(self.timing);
        let (w, h) = (self.w, self.h);

        // ------------------------------------------------------------ setup
        raster::normals(meshes, frame, &mut self.vbase, &mut self.vn);
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
        self.light.draw(meshes, frame, &self.vbase);
        clock.lap(&mut self.ms[1]);

        // ------------------------------------------------------ visibility
        self.tiles.reset();
        let pass = Pass {
            m: rows(fpx, w as f32 * 0.5, h as f32 * 0.5, 1.0),
            near,
            w: w as i32,
            h: h as i32,
            front: -1.0,
        };
        let cut = raster::draw(
            &pass,
            &OFFS,
            meshes,
            frame,
            &self.vbase,
            &self.vn,
            &mut self.hv,
            &mut self.tris,
            &mut self.keys,
            &mut self.tiles,
        );
        raster::resolve(
            &self.keys,
            w,
            h,
            &self.tiles,
            &self.tris,
            &mut self.frags,
            &mut self.open,
        );
        self.counts = [self.tris.len(), self.frags.len(), cut];
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
        self.mirror.tiles.reset();
        if eye.y > 0.0 && frame.floor.mirror > 0.0 {
            let mpass = Pass {
                m: rows(mfp, w as f32 * 0.25, h as f32 * 0.25, -1.0),
                near,
                w: mw as i32,
                h: mh as i32,
                front: 1.0,
            };
            raster::draw(
                &mpass,
                &OFFS,
                meshes,
                frame,
                &self.vbase,
                &self.vn,
                &mut self.hv,
                &mut self.mirror.tris,
                &mut self.mirror.keys,
                &mut self.mirror.tiles,
            );
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
        self.mirror.shade(&under, &self.cent);
        self.bright.copy_from_slice(&self.tiles.f);
        self.mirror.floor(
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
        self.ao.compute(
            &self.keys[..w * h],
            &self.tris,
            &self.tiles,
            w,
            h,
            right,
            up,
            fwd,
            fpx,
            fq,
        );
        clock.lap(&mut self.ms[4]);

        // --------------------------------------------------------- shading
        let inv_ns = 1.0 / NS as f32;
        for f in &self.frags {
            let t = &self.tris[f.tri as usize];
            let (x, y) = (f.px as usize % w, f.px as usize / w);
            let (ox, oy) = self.cent[(f.bits >> 4) as usize & 15];
            let sf = shade::surf(&base, t, x as f32 + ox, y as f32 + oy);
            let ao = self.ao.at(x, y, sf.w);
            let c = compress(
                shade::shade(&base, &sf, t, ao),
                (f.bits & 15) as f32 * inv_ns,
            );
            let o = &mut self.hdr[f.px as usize];
            o[0] += c[0];
            o[1] += c[1];
            o[2] += c[2];
        }
        clock.lap(&mut self.ms[5]);

        // ---------------------------------------------------------- finish
        self.post.finish(
            &mut self.hdr,
            w,
            h,
            frame.exposure,
            frame.bloom,
            &self.open,
            &self.tiles,
            &self.bright,
            NS,
            &mut self.out,
        );
        clock.lap(&mut self.ms[6]);
        &self.out
    }
}
