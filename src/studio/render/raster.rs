//! Which triangle is in front at each sample.
//!
//! Triangles are walked a row of samples at a time. An edge shared by two
//! triangles is always evaluated from its upper end with the same arithmetic,
//! so both sides agree to the bit on where it crosses a row: no sample on a
//! shared edge is taken twice or missed.
//!
//! Depth is `q = 1 / w` (larger is nearer), which is a plane across the screen.

use super::{fmax, fmin};
use crate::studio::math::{M4, V3};
use crate::studio::scene::{Frame, Mesh};

#[inline(always)]
pub fn ceil_i(x: f32) -> i32 {
    (-((-x).floor())) as i32
}

/// Takes a run of samples on one row.
pub trait Sink {
    fn span(&mut self, s: usize, r: usize, c0: usize, n: usize, q: f32, dq: f32);
}

/// Samples in a pixel: a turned square, so a near-level or near-upright edge
/// crosses them one at a time.
pub const NS: usize = 4;
pub const OFFS: [(f32, f32); NS] = [
    (0.375, 0.125),
    (0.875, 0.375),
    (0.125, 0.625),
    (0.625, 0.875),
];

/// A surface in a pixel: the triangle to shade, where, and how much of the
/// pixel it stands for.
pub struct Frag {
    pub px: u32,
    pub tri: u32,
    /// Low bits: samples it stands for. Next four: which samples are its own.
    pub bits: u32,
}

/// Where to shade a surface that owns only some of a pixel's samples: the
/// middle of the ones it owns, by which samples those are.
pub fn cent_table() -> [(f32, f32); 16] {
    let mut cent = [(0.5, 0.5); 16];
    for (m, c) in cent.iter_mut().enumerate().skip(1) {
        let (mut x, mut y, mut n) = (0.0, 0.0, 0.0);
        for (s, o) in OFFS.iter().enumerate() {
            if m >> s & 1 == 1 {
                x += o.0;
                y += o.1;
                n += 1.0;
            }
        }
        *c = (x / n, y / n);
    }
    cent
}

/// Marks every tile within `r` tiles of a marked one.
pub fn dilate(src: &[u8], tw: usize, th: usize, r: usize, tmp: &mut Vec<u8>, dst: &mut Vec<u8>) {
    tmp.clear();
    tmp.resize(tw * th, 0);
    dst.clear();
    dst.resize(tw * th, 0);
    for y in 0..th {
        for x in 0..tw {
            tmp[y * tw + x] =
                (x.saturating_sub(r)..(x + r + 1).min(tw)).fold(0, |m, xx| m | src[y * tw + xx]);
        }
    }
    for y in 0..th {
        for x in 0..tw {
            dst[y * tw + x] =
                (y.saturating_sub(r)..(y + r + 1).min(th)).fold(0, |m, yy| m | tmp[yy * tw + x]);
        }
    }
}

/// Gathers each pixel's samples into surfaces, in the tiles that hold any.
/// Samples of one triangle are one surface; so are samples of neighbouring
/// triangles of the same part that face nearly the same way, since they
/// would shade alike. `open[i]` gets the number of samples nothing covers.
pub fn resolve(
    keys: &[u64],
    w: usize,
    h: usize,
    tiles: &Tiles,
    tris: &[Tri],
    frags: &mut Vec<Frag>,
    open: &mut [u8],
) {
    let n = w * h;
    let (k0, rest) = keys.split_at(n);
    let (k1, rest) = rest.split_at(n);
    let (k2, k3) = rest.split_at(n);
    frags.clear();
    let (tw, sh) = (tiles.tw, tiles.sh);
    // Row by row, but only through the tiles that anything was drawn in.
    for (y, tx) in (0..h).flat_map(|y| (0..tw).map(move |tx| (y, tx))) {
        if tiles.f[(y >> sh) * tw + tx] == 0 {
            continue;
        }
        for i in y * w + (tx << sh)..y * w + ((tx + 1) << sh).min(w) {
            let ids = [k0[i] as u32, k1[i] as u32, k2[i] as u32, k3[i] as u32];
            if ids[0] == ids[1] && ids[0] == ids[2] && ids[0] == ids[3] {
                if ids[0] == 0 {
                    open[i] = NS as u8;
                } else {
                    open[i] = 0;
                    frags.push(Frag {
                        px: i as u32,
                        tri: ids[0] - 1,
                        bits: 4 | 15 << 4,
                    });
                }
                continue;
            }
            let (mut used, mut free) = (0u32, 0u8);
            for s in 0..NS {
                if used >> s & 1 == 1 {
                    continue;
                }
                if ids[s] == 0 {
                    free += 1;
                    continue;
                }
                let t = &tris[ids[s] as usize - 1];
                let mut own = 1u32 << s;
                let mut all = own;
                for (s2, &id2) in ids.iter().enumerate().skip(s + 1) {
                    if used >> s2 & 1 == 1 || id2 == 0 {
                        continue;
                    }
                    if id2 == ids[s] {
                        own |= 1 << s2;
                        all |= 1 << s2;
                    } else {
                        let t2 = &tris[id2 as usize - 1];
                        if t2.inst == t.inst && t2.ng.dot(t.ng) > 0.9 {
                            all |= 1 << s2;
                        }
                    }
                }
                used |= all;
                frags.push(Frag {
                    px: i as u32,
                    tri: ids[s] - 1,
                    bits: all.count_ones() | own << 4,
                });
            }
            open[i] = free;
        }
    }
}

/// The picture cut into square tiles, with a mark on each one that anything
/// was drawn in this frame. A tile with no mark is never cleared, never read
/// and never shaded: an empty stretch of floor costs nothing here.
pub struct Tiles {
    pub tw: usize,
    pub th: usize,
    /// A tile is `1 << sh` pixels across.
    pub sh: usize,
    pub f: Vec<u8>,
}

impl Tiles {
    pub fn new(w: usize, h: usize, sh: usize) -> Tiles {
        let (tw, th) = ((w + (1 << sh) - 1) >> sh, (h + (1 << sh) - 1) >> sh);
        Tiles {
            tw,
            th,
            sh,
            f: vec![0; tw * th],
        }
    }
    pub fn reset(&mut self) {
        self.f.fill(0);
    }
}

/// Depth and triangle packed in one word: the bits of a positive float sort
/// like the float, so one integer comparison is the depth test, and ties go
/// to the later triangle on every machine. A tile's samples are cleared the
/// first time anything lands in it.
pub struct KeySink<'a> {
    pub buf: &'a mut [u64],
    pub plane: usize,
    pub w: usize,
    pub h: usize,
    pub ns: usize,
    pub id: u64,
    pub tiles: &'a mut Tiles,
}

impl Sink for KeySink<'_> {
    #[inline(always)]
    fn span(&mut self, s: usize, r: usize, c0: usize, n: usize, mut q: f32, dq: f32) {
        let t = &mut *self.tiles;
        let sh = t.sh;
        let tr = (r >> sh) * t.tw;
        for tc in (c0 >> sh)..=((c0 + n - 1) >> sh) {
            if t.f[tr + tc] == 0 {
                t.f[tr + tc] = 1;
                let (x0, y0) = (tc << sh, (r >> sh) << sh);
                let (x1, y1) = ((x0 + (1 << sh)).min(self.w), (y0 + (1 << sh)).min(self.h));
                for p in 0..self.ns {
                    for y in y0..y1 {
                        let o = p * self.plane + y * self.w;
                        self.buf[o + x0..o + x1].fill(0);
                    }
                }
            }
        }
        let at = s * self.plane + r * self.w + c0;
        for k in &mut self.buf[at..at + n] {
            let key = ((q.to_bits() as u64) << 32) | self.id;
            if key > *k {
                *k = key;
            }
            q += dq;
        }
    }
}

/// Depth alone, for the lamp's map. Marks the tiles it writes in.
pub struct DepthSink<'a> {
    pub buf: &'a mut [f32],
    pub w: usize,
    pub tiles: &'a mut Tiles,
}

impl Sink for DepthSink<'_> {
    #[inline(always)]
    fn span(&mut self, _s: usize, r: usize, c0: usize, n: usize, mut q: f32, dq: f32) {
        let t = &mut *self.tiles;
        let tr = (r >> t.sh) * t.tw;
        for tc in (c0 >> t.sh)..=((c0 + n - 1) >> t.sh) {
            t.f[tr + tc] = 1;
        }
        let at = r * self.w + c0;
        for d in &mut self.buf[at..at + n] {
            if q > *d {
                *d = q;
            }
            q += dq;
        }
    }
}

/// Fills the samples inside a triangle. `offs` is where the samples sit in a
/// pixel; `pl` is the depth plane: x0, y0, q0, dq/dx, dq/dy.
#[inline(always)]
pub fn scan<S: Sink>(
    sink: &mut S,
    w: i32,
    h: i32,
    offs: &[(f32, f32)],
    v: &[[f32; 2]; 3],
    pl: &[f32; 5],
) {
    let (mut a, mut b, mut c) = (v[0], v[1], v[2]);
    if b[1] < a[1] {
        core::mem::swap(&mut a, &mut b);
    }
    if c[1] < b[1] {
        core::mem::swap(&mut b, &mut c);
        if b[1] < a[1] {
            core::mem::swap(&mut a, &mut b);
        }
    }
    if !(c[1] > a[1]) {
        return;
    }
    let s_ac = (c[0] - a[0]) / (c[1] - a[1]);
    let s_ab = if b[1] > a[1] {
        (b[0] - a[0]) / (b[1] - a[1])
    } else {
        0.0
    };
    let s_bc = if c[1] > b[1] {
        (c[0] - b[0]) / (c[1] - b[1])
    } else {
        0.0
    };
    for (s, &(ox, oy)) in offs.iter().enumerate() {
        let r0 = ceil_i(a[1] - oy).max(0);
        let r1 = ceil_i(c[1] - oy).min(h);
        let rm = ceil_i(b[1] - oy);
        for r in r0..r1 {
            let ys = r as f32 + oy;
            let xl = a[0] + (ys - a[1]) * s_ac;
            let xs = if r < rm {
                a[0] + (ys - a[1]) * s_ab
            } else {
                b[0] + (ys - b[1]) * s_bc
            };
            let (lo, hi) = if xl < xs { (xl, xs) } else { (xs, xl) };
            let c0 = ceil_i(lo - ox).max(0);
            let c1 = ceil_i(hi - ox).min(w);
            if c0 >= c1 {
                continue;
            }
            let q = pl[2] + (c0 as f32 + ox - pl[0]) * pl[3] + (ys - pl[1]) * pl[4];
            sink.span(s, r as usize, c0 as usize, (c1 - c0) as usize, q, pl[3]);
        }
    }
}

/// A drawn triangle, kept so a pixel can ask what it is looking at.
/// `q`, `u`, `v` are planes over the screen (value at x0 y0, slope in x, slope
/// in y): `u / q` and `v / q` are the weights of the second and third corner.
#[derive(Clone, Copy)]
pub struct Tri {
    pub x0: f32,
    pub y0: f32,
    pub q: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
    /// The three corners in the frame's list of normals.
    pub vi: [u32; 3],
    pub inst: u32,
    /// The flat face's direction in the world.
    pub ng: V3,
    /// How fast the shading normal turns across a pixel, squared.
    pub nvar: f32,
}

/// One view: world to pixels. Row 0 over row 2 is x, row 1 over row 2 is y,
/// row 2 is the distance along the view.
pub struct Pass {
    pub m: [[f32; 4]; 3],
    pub near: f32,
    pub w: i32,
    pub h: i32,
    /// The sign of the screen area of a triangle that faces the viewer.
    pub front: f32,
}

fn rows_times(p: &[[f32; 4]; 3], x: &M4) -> [[f32; 4]; 3] {
    let m = &x.m;
    let mut o = [[0.0f32; 4]; 3];
    for i in 0..3 {
        for c in 0..4 {
            o[i][c] = p[i][0] * m[c * 4] + p[i][1] * m[c * 4 + 1] + p[i][2] * m[c * 4 + 2];
        }
        o[i][3] += p[i][3];
    }
    o
}

#[inline(always)]
fn grad(d1: f32, d2: f32, ax: f32, ay: f32, bx: f32, by: f32, inv: f32) -> (f32, f32) {
    ((d1 * by - d2 * ay) * inv, (ax * d2 - bx * d1) * inv)
}

/// World normals for every vertex of every instance, and where each instance
/// starts in that list.
pub fn normals(meshes: &[Mesh], frame: &Frame, vbase: &mut Vec<u32>, vn: &mut Vec<V3>) {
    vbase.clear();
    vn.clear();
    for inst in &frame.instances {
        vbase.push(vn.len() as u32);
        let Some(mesh) = meshes.get(inst.mesh as usize) else {
            continue;
        };
        let x = &inst.xf;
        let s2 = x.m[0] * x.m[0] + x.m[1] * x.m[1] + x.m[2] * x.m[2];
        let k = if s2 > 0.0 { 1.0 / s2.sqrt() } else { 0.0 };
        for i in 0..mesh.pos.len() {
            let n = mesh.nrm.get(i).copied().unwrap_or(V3::Y);
            vn.push(x.dir(n) * k);
        }
    }
}

/// Draws every instance into `buf` and records the triangles that could be
/// seen. A sample's value is its depth and the triangle's place in `tris`
/// plus one.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    pass: &Pass,
    offs: &[(f32, f32)],
    meshes: &[Mesh],
    frame: &Frame,
    vbase: &[u32],
    vn: &[V3],
    hv: &mut Vec<[f32; 4]>,
    tris: &mut Vec<Tri>,
    buf: &mut [u64],
    tiles: &mut Tiles,
) -> usize {
    tris.clear();
    let mut cut = 0;
    let plane = (pass.w * pass.h) as usize;
    let (wf, hf) = (pass.w as f32, pass.h as f32);
    for (ii, inst) in frame.instances.iter().enumerate() {
        let Some(mesh) = meshes.get(inst.mesh as usize) else {
            continue;
        };
        let m = rows_times(&pass.m, &inst.xf);
        hv.clear();
        for p in &mesh.pos {
            let x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
            let y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
            let w = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
            if w > pass.near {
                let q = 1.0 / w;
                hv.push([x * q, y * q, q, w]);
            } else {
                hv.push([x, y, 0.0, w]);
            }
        }
        let base = vbase[ii];
        let nv = hv.len();
        for t in mesh.idx.as_chunks::<3>().0 {
            let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if i0 >= nv || i1 >= nv || i2 >= nv {
                continue;
            }
            let (a, b, c) = (hv[i0], hv[i1], hv[i2]);
            // Up to two triangles once the part behind the camera is cut off.
            let mut poly = [[0.0f32; 5]; 4];
            let mut n = 0;
            if a[2] > 0.0 && b[2] > 0.0 && c[2] > 0.0 {
                poly[0] = [a[0], a[1], a[2], 0.0, 0.0];
                poly[1] = [b[0], b[1], b[2], b[2], 0.0];
                poly[2] = [c[0], c[1], c[2], 0.0, c[2]];
                n = 3;
            } else if a[2] > 0.0 || b[2] > 0.0 || c[2] > 0.0 {
                cut += 1;
                let raw = |v: [f32; 4], b1: f32, b2: f32| {
                    if v[2] > 0.0 {
                        [v[0] * v[3], v[1] * v[3], v[3], b1, b2]
                    } else {
                        [v[0], v[1], v[3], b1, b2]
                    }
                };
                let vs = [raw(a, 0.0, 0.0), raw(b, 1.0, 0.0), raw(c, 0.0, 1.0)];
                for e in 0..3 {
                    let (p, c) = (vs[e], vs[(e + 1) % 3]);
                    let (pin, cin) = (p[2] > pass.near, c[2] > pass.near);
                    let fin = |v: [f32; 5]| {
                        let q = 1.0 / v[2];
                        [v[0] * q, v[1] * q, q, v[3] * q, v[4] * q]
                    };
                    if pin {
                        poly[n] = fin(p);
                        n += 1;
                    }
                    if pin != cin {
                        // Always from the inside end, so a shared edge is cut at the same point.
                        let (i, o) = if pin { (p, c) } else { (c, p) };
                        let k = (pass.near - i[2]) / (o[2] - i[2]);
                        let mut x = [0.0f32; 5];
                        for j in 0..5 {
                            x[j] = i[j] + (o[j] - i[j]) * k;
                        }
                        x[2] = pass.near;
                        poly[n] = fin(x);
                        n += 1;
                    }
                }
            }
            let mut k = 2;
            while k < n {
                let (p0, p1, p2) = (poly[0], poly[k - 1], poly[k]);
                k += 1;
                let (ax, ay, bx, by) = (p1[0] - p0[0], p1[1] - p0[1], p2[0] - p0[0], p2[1] - p0[1]);
                let area2 = ax * by - bx * ay;
                if !(area2 * pass.front > 0.0) {
                    continue;
                }
                let (x0, x1) = (
                    fmin(fmin(p0[0], p1[0]), p2[0]),
                    fmax(fmax(p0[0], p1[0]), p2[0]),
                );
                let (y0, y1) = (
                    fmin(fmin(p0[1], p1[1]), p2[1]),
                    fmax(fmax(p0[1], p1[1]), p2[1]),
                );
                if !(x1 >= 0.0 && x0 < wf && y1 >= 0.0 && y0 < hf) {
                    continue;
                }
                let inv = 1.0 / area2;
                let dq = grad(p1[2] - p0[2], p2[2] - p0[2], ax, ay, bx, by, inv);
                let du = grad(p1[3] - p0[3], p2[3] - p0[3], ax, ay, bx, by, inv);
                let dv = grad(p1[4] - p0[4], p2[4] - p0[4], ax, ay, bx, by, inv);
                let (m0, m1, m2) = (mesh.pos[i0], mesh.pos[i1], mesh.pos[i2]);
                let ng = inst.xf.dir((m1 - m0).cross(m2 - m0)).norm();
                let vi = [base + i0 as u32, base + i1 as u32, base + i2 as u32];
                let (n0, n1, n2) = (vn[vi[0] as usize], vn[vi[1] as usize], vn[vi[2] as usize]);
                let (e1, e2) = (n1 - n0, n2 - n0);
                let mut nvar = 0.0;
                if e1.dot(e1) + e2.dot(e2) > 1e-8 {
                    let iq = 3.0 / (p0[2] + p1[2] + p2[2]);
                    let t = 1.0 / 3.0;
                    let nx = e1 * ((du.0 - dq.0 * t) * iq) + e2 * ((dv.0 - dq.0 * t) * iq);
                    let ny = e1 * ((du.1 - dq.1 * t) * iq) + e2 * ((dv.1 - dq.1 * t) * iq);
                    nvar = nx.dot(nx) + ny.dot(ny);
                }
                tris.push(Tri {
                    x0: p0[0],
                    y0: p0[1],
                    q: [p0[2], dq.0, dq.1],
                    u: [p0[3], du.0, du.1],
                    v: [p0[4], dv.0, dv.1],
                    vi,
                    inst: ii as u32,
                    ng,
                    nvar,
                });
                let mut sink = KeySink {
                    buf: &mut *buf,
                    plane,
                    w: pass.w as usize,
                    h: pass.h as usize,
                    ns: offs.len(),
                    id: tris.len() as u64,
                    tiles: &mut *tiles,
                };
                scan(
                    &mut sink,
                    pass.w,
                    pass.h,
                    offs,
                    &[[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]],
                    &[p0[0], p0[1], p0[2], dq.0, dq.1],
                );
            }
        }
    }
    cut
}
