//! The key lamp: its pool of light and its shadow.
//!
//! The lamp looks at the scene through a square window of `N` by `N` texels.
//! Only the faces turned away from it are drawn into the window, so a lit face
//! is compared with the far side of its own solid and never with itself: no
//! acne, and a bias small enough that light does not creep under a contact.
//!
//! Two readings of the same map:
//! - for meshes, a box filter of depth comparisons whose width follows the
//!   lamp's size and the gap between blocker and receiver, taken on a coarser
//!   level of the map when it is wide;
//! - for the floor, where everything in the map is a blocker, a pyramid of
//!   averaged coverage and depth, read with a few smooth look-ups.
//!
//! The window sits on a fixed grid in the lamp's view and moves in steps of 64
//! texels, so a still shadow has still texels whatever else moves.

use super::par;
use super::raster::{Part, Setup, Sink, Tiles, join, runs, scan_edges};
use super::{clamp01, fmax, fmin};
use crate::studio::math::{M4, V3, v3};
use crate::studio::scene::{Frame, Mesh};
use core::ops::Range;

pub const N: usize = 2048;
/// Levels above the map itself: level l is N >> l across.
const LEVELS: usize = 8;
/// The levels whose tiles are a texel or more across: each row of tiles is
/// built up to there on its own.
const BANDED: usize = 6;
/// For testing only: when set, a mesh's shadow is worked out on the map
/// itself, texel by texel, however wide. Slow; what the levels approximate.
const EXHAUSTIVE: bool = false;
/// The widest half-width of a mesh's penumbra, in texels.
const RCAP: f32 = 96.0;
/// Tangent of the angle from the lamp's axis where its pool starts to fall
/// away, and where it reaches black.
pub const POOL_IN: f32 = 0.22;
pub const POOL_OUT: f32 = 0.80;

pub struct Light {
    d: Vec<f32>,
    qmax: Vec<Vec<f32>>,
    qmin: Vec<Vec<f32>>,
    cov: Vec<Vec<[f32; 2]>>,
    /// The map's own depth at one fixed texel inside each texel of a level.
    dec: Vec<Vec<f32>>,
    uvq: Vec<[f32; 3]>,
    /// Which 64-texel tiles of the map were drawn in, this frame and the
    /// one before: only those are cleared and built into the levels above.
    tiles: Tiles,
    prev: Vec<u8>,
    rx: [f32; 4],
    ry: [f32; 4],
    rw: [f32; 4],
    ax: V3,
    ay: V3,
    pub pos: V3,
    u0: f32,
    v0: f32,
    tau: f32,
    itau: f32,
    /// Lamp radius in texels at unit depth.
    k: f32,
    qglobal: f32,
    /// The faces drawn into the map, in texels, by row of tiles.
    set: Setup,
    /// Which points are in front of the lamp, and the runs of instances
    /// and their faces when several threads set them up.
    seen: Vec<bool>,
    spans: Vec<Range<usize>>,
    parts: Vec<Part<()>>,
}

/// One row of tiles of the map, and of the levels built from it.
struct Band<'a> {
    k: usize,
    d: &'a mut [f32],
    marks: &'a mut [u8],
    prev: &'a mut [u8],
    qmax: [&'a mut [f32]; BANDED],
    qmin: [&'a mut [f32]; BANDED],
    cov: [&'a mut [[f32; 2]]; BANDED],
    dec: [&'a mut [f32]; BANDED],
}

/// Depth alone into a row of tiles of the map, marking the tiles it writes in.
struct DepthBand<'a> {
    buf: &'a mut [f32],
    y0: usize,
    marks: &'a mut [u8],
}

impl Sink for DepthBand<'_> {
    #[inline(always)]
    fn span(&mut self, _s: usize, r: usize, c0: usize, n: usize, mut q: f32, dq: f32) {
        for tc in (c0 >> 6)..=((c0 + n - 1) >> 6) {
            self.marks[tc] = 1;
        }
        let at = (r - self.y0) * N + c0;
        for d in &mut self.buf[at..at + n] {
            if q > *d {
                *d = q;
            }
            q += dq;
        }
    }
}

#[inline(always)]
fn dot4(r: &[f32; 4], p: V3) -> f32 {
    r[0] * p.x + r[1] * p.y + r[2] * p.z + r[3]
}

/// log2 for a blend weight: exact at powers of two, straight between them.
#[inline(always)]
pub fn log2ish(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((b & 0x007f_ffff) | 0x3f80_0000);
    e as f32 + (m - 1.0)
}

impl Default for Light {
    fn default() -> Self {
        Self::new()
    }
}

impl Light {
    pub fn new() -> Light {
        let mut l = Light {
            d: vec![0.0; N * N],
            qmax: Vec::new(),
            qmin: Vec::new(),
            cov: Vec::new(),
            dec: Vec::new(),
            uvq: Vec::new(),
            tiles: Tiles::new(N, N, 6),
            prev: vec![0; (N >> 6) * (N >> 6)],
            rx: [0.0; 4],
            ry: [0.0; 4],
            rw: [0.0; 4],
            ax: V3::X,
            ay: V3::Y,
            pos: V3::ZERO,
            u0: 0.0,
            v0: 0.0,
            tau: 1.0,
            itau: 1.0,
            k: 0.0,
            qglobal: 0.0,
            set: Setup::default(),
            seen: Vec::new(),
            spans: Vec::new(),
            parts: Vec::new(),
        };
        for lv in 1..=LEVELS {
            let n = N >> lv;
            l.qmax.push(vec![0.0; n * n]);
            l.qmin.push(vec![0.0; n * n]);
            l.cov.push(vec![[0.0; 2]; n * n]);
            l.dec.push(vec![0.0; n * n]);
        }
        l
    }

    /// Draws the map for this frame.
    pub fn draw(&mut self, meshes: &[Mesh], frame: &Frame, vbase: &[u32]) {
        self.draw_with(meshes, frame, vbase, 1);
    }

    /// `draw` on up to `threads` threads; the map is the same.
    pub fn draw_with(&mut self, meshes: &[Mesh], frame: &Frame, vbase: &[u32], threads: usize) {
        let key = &frame.key;
        self.pos = key.pos;
        let f = (key.at - key.pos).norm();
        let up = if f.y.abs() > 0.999 { V3::Z } else { V3::Y };
        let view = M4::look_at(key.pos, key.at, up);
        let m = &view.m;
        self.rx = [m[0], m[4], m[8], m[12]];
        self.ry = [m[1], m[5], m[9], m[13]];
        self.rw = [-m[2], -m[6], -m[10], -m[14]];
        self.ax = v3(m[0], m[4], m[8]);
        self.ay = v3(m[1], m[5], m[9]);

        // Where everything is, seen from the lamp: a run of instances to a
        // thread, then the bounds, point by point in order.
        let n = vbase.len().checked_sub(1).map_or(0, |last| {
            let inst = &frame.instances[last];
            vbase[last] as usize + meshes.get(inst.mesh as usize).map_or(0, |m| m.pos.len())
        });
        self.uvq.resize(n, [0.0; 3]);
        self.seen.resize(n, false);
        runs(
            meshes,
            frame,
            if threads > 1 { threads * 2 } else { 1 },
            &mut self.spans,
        );
        {
            let (mut uvq, mut seen) = (&mut self.uvq[..], &mut self.seen[..]);
            let work = self.spans.iter().map(move |run| {
                let end = vbase.get(run.end).map_or(n, |&e| e as usize);
                let len = end - vbase[run.start] as usize;
                let (a, more) = core::mem::take(&mut uvq).split_at_mut(len);
                let (b, more2) = core::mem::take(&mut seen).split_at_mut(len);
                (uvq, seen) = (more, more2);
                (run.clone(), a, b)
            });
            let (rx, ry, rw) = (self.rx, self.ry, self.rw);
            par::each0(threads, work, |(run, uvq, seen)| {
                let mut out = uvq.iter_mut().zip(seen.iter_mut());
                for inst in &frame.instances[run] {
                    let Some(mesh) = meshes.get(inst.mesh as usize) else {
                        continue;
                    };
                    for p in &mesh.pos {
                        let (o, s) = out.next().unwrap();
                        let wp = inst.xf.point(*p);
                        let w = dot4(&rw, wp);
                        *s = w > 0.05;
                        *o = if *s {
                            let q = 1.0 / w;
                            [dot4(&rx, wp) * q, dot4(&ry, wp) * q, q]
                        } else {
                            [0.0, 0.0, 0.0]
                        };
                    }
                }
            });
        }
        let (mut ulo, mut uhi, mut vlo, mut vhi) = (1e30f32, -1e30f32, 1e30f32, -1e30f32);
        for (p, _) in self.uvq.iter().zip(&self.seen).filter(|s| *s.1) {
            ulo = fmin(ulo, p[0]);
            uhi = fmax(uhi, p[0]);
            vlo = fmin(vlo, p[1]);
            vhi = fmax(vhi, p[1]);
        }
        if !(uhi > ulo) {
            (ulo, uhi, vlo, vhi) = (-0.1, 0.1, -0.1, 0.1);
        }
        // A window from a fixed ladder of sizes, on a fixed grid.
        let need = fmax(uhi - ulo, vhi - vlo) * 0.5 * 1.10 + 1e-4;
        let mut half = 1.0 / 4096.0;
        while half < need {
            half *= 1.414_213_5;
        }
        self.tau = 2.0 * half / N as f32;
        self.itau = 1.0 / self.tau;
        let g = self.tau * 64.0;
        let cu = ((ulo + uhi) * 0.5 / g + 0.5).floor() * g;
        let cv = ((vlo + vhi) * 0.5 / g + 0.5).floor() * g;
        self.u0 = cu - half;
        self.v0 = cv - half;
        self.k = key.size * self.itau;

        // The faces turned away from the lamp, in texels, filed by the rows
        // of tiles they may touch.
        let tn = N >> 6;
        let window = (self.u0, self.v0, self.itau);
        if self.spans.len() <= 1 {
            self.set.reset(tn);
            let all = 0..frame.instances.len();
            faces(meshes, frame, vbase, &self.uvq, window, all, &mut self.set);
        } else {
            if self.parts.len() < self.spans.len() {
                self.parts.resize_with(self.spans.len(), Part::default);
            }
            let parts = &mut self.parts[..self.spans.len()];
            let uvq = &self.uvq[..];
            par::each0(
                threads,
                parts.iter_mut().zip(self.spans.iter().cloned()),
                |(p, run)| {
                    p.set.reset(tn);
                    faces(meshes, frame, vbase, uvq, window, run, &mut p.set);
                },
            );
            join(parts, &mut Vec::new(), &mut self.set, tn, threads);
        }

        // Each row of tiles on its own: emptied of what the frame before
        // drew there, drawn, and built into the levels above as far as their
        // tiles are a texel across.
        let texels = |l: usize| (64 >> l) * (N >> l);
        let mut d = self.d.chunks_mut(64 * N);
        let mut marks = self.tiles.f.chunks_mut(tn);
        let mut prev = self.prev.chunks_mut(tn);
        let mut qmax = self
            .qmax
            .iter_mut()
            .enumerate()
            .map(|(i, l)| l.chunks_mut(texels(i + 1)));
        let mut qmax: [_; BANDED] = core::array::from_fn(|_| qmax.next().unwrap());
        let mut qmin = self
            .qmin
            .iter_mut()
            .enumerate()
            .map(|(i, l)| l.chunks_mut(texels(i + 1)));
        let mut qmin: [_; BANDED] = core::array::from_fn(|_| qmin.next().unwrap());
        let mut cov = self
            .cov
            .iter_mut()
            .enumerate()
            .map(|(i, l)| l.chunks_mut(texels(i + 1)));
        let mut cov: [_; BANDED] = core::array::from_fn(|_| cov.next().unwrap());
        let mut dec = self
            .dec
            .iter_mut()
            .enumerate()
            .map(|(i, l)| l.chunks_mut(texels(i + 1)));
        let mut dec: [_; BANDED] = core::array::from_fn(|_| dec.next().unwrap());
        let bands = (0..tn).map(move |k| Band {
            k,
            d: d.next().unwrap(),
            marks: marks.next().unwrap(),
            prev: prev.next().unwrap(),
            qmax: qmax.each_mut().map(|c| c.next().unwrap()),
            qmin: qmin.each_mut().map(|c| c.next().unwrap()),
            cov: cov.each_mut().map(|c| c.next().unwrap()),
            dec: dec.each_mut().map(|c| c.next().unwrap()),
        });
        let set = &self.set;
        par::each0(threads, bands, |b| band(b, set));
        self.top();
    }

    /// The levels whose tiles are under a texel, built whole from the ones
    /// below them, and the nearest depth in the map.
    fn top(&mut self) {
        for lv in BANDED..LEVELS {
            let n = N >> (lv + 1);
            let (lo, hi) = self.qmax.split_at_mut(lv);
            let (lo2, hi2) = self.qmin.split_at_mut(lv);
            let (lo3, hi3) = self.cov.split_at_mut(lv);
            let (smx, smn, scv) = (&lo[lv - 1], &lo2[lv - 1], &lo3[lv - 1]);
            let (dmx, dmn, dcv) = (&mut hi[0], &mut hi2[0], &mut hi3[0]);
            let sn = n * 2;
            for y in 0..n {
                for x in 0..n {
                    let (i, o) = (2 * y * sn + 2 * x, y * n + x);
                    dmx[o] = fmax(fmax(smx[i], smx[i + 1]), fmax(smx[i + sn], smx[i + sn + 1]));
                    dmn[o] = fmin(fmin(smn[i], smn[i + 1]), fmin(smn[i + sn], smn[i + sn + 1]));
                    let (a, b, c, d) = (scv[i], scv[i + 1], scv[i + sn], scv[i + sn + 1]);
                    dcv[o] = [
                        (a[0] + b[0] + c[0] + d[0]) * 0.25,
                        (a[1] + b[1] + c[1] + d[1]) * 0.25,
                    ];
                }
            }
        }
        for lv in BANDED + 1..=LEVELS {
            let (n, half) = (N >> lv, 1usize << (lv - 1));
            let dst = &mut self.dec[lv - 1];
            for y in 0..n {
                let src = &self.d[((y << lv) + half) * N..];
                for x in 0..n {
                    dst[y * n + x] = src[(x << lv) + half];
                }
            }
        }
        let top = &self.qmax[LEVELS - 1];
        self.qglobal = top.iter().fold(0.0f32, |m, &q| fmax(m, q));
    }

    /// A world point in the lamp's view: sideways and upward tangent, and
    /// one over the distance along the axis.
    #[inline(always)]
    pub fn project(&self, p: V3) -> (f32, f32, f32) {
        let q = 1.0 / fmax(dot4(&self.rw, p), 1e-4);
        (dot4(&self.rx, p) * q, dot4(&self.ry, p) * q, q)
    }

    /// The lamp's three rows, for a caller that walks a plane.
    pub fn rows(&self) -> (&[f32; 4], &[f32; 4], &[f32; 4]) {
        (&self.rx, &self.ry, &self.rw)
    }

    /// How much of the lamp's light a direction gets: 1 in the pool, 0 outside.
    #[inline(always)]
    pub fn pool(u: f32, v: f32) -> f32 {
        let r2 = u * u + v * v;
        let f =
            clamp01((POOL_OUT * POOL_OUT - r2) * (1.0 / (POOL_OUT * POOL_OUT - POOL_IN * POOL_IN)));
        f * f * (3.0 - 2.0 * f)
    }

    /// The share of the lamp that reaches a point on a mesh. `ng` is the
    /// face's direction, `c` is `ng . (p - lamp)` (negative for a lit face).
    pub fn shadow(&self, u: f32, v: f32, q: f32, ng: V3, c: f32) -> f32 {
        let nf = N as f32;
        let (s, t) = ((u - self.u0) * self.itau, (v - self.v0) * self.itau);
        if !(s >= 0.0 && s < nf && t >= 0.0 && t < nf) {
            return 1.0;
        }
        let ic = self.tau / c;
        let (gs, gt) = (ng.dot(self.ax) * ic, ng.dot(self.ay) * ic);
        let gabs = gs.abs() + gt.abs();
        let eps0 = 0.004 * q * q + 1e-6 * q;

        let mut rho = fmin(self.k * (self.qglobal - q), RCAP);
        for _ in 0..3 {
            if rho <= 0.0 {
                return 1.0;
            }
            let mut lq = 1;
            while ((1 << lq) as f32) < 2.0 * rho + 2.0 {
                lq += 1;
            }
            let lq = lq.min(LEVELS);
            let (nl, sc) = (N >> lq, 1.0 / (1 << lq) as f32);
            let cl = |x: f32| ((x * sc) as usize).min(nl - 1);
            let (i0, i1, j0, j1) = (
                cl(fmax(s - rho, 0.0)),
                cl(s + rho),
                cl(fmax(t - rho, 0.0)),
                cl(t + rho),
            );
            let (mx, mn) = (&self.qmax[lq - 1], &self.qmin[lq - 1]);
            let (mut qmx, mut qmn) = (0.0f32, 1e30f32);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    qmx = fmax(qmx, mx[j * nl + i]);
                    qmn = fmin(qmn, mn[j * nl + i]);
                }
            }
            let slack = gabs * rho;
            if qmx <= q - slack + eps0 {
                return 1.0;
            }
            if qmn > q + slack + eps0 {
                return 0.0;
            }
            let next = fmin(rho, self.k * (qmx - q));
            let done = next > 0.75 * rho;
            rho = next;
            if done {
                break;
            }
        }
        if rho <= 0.0 {
            return 1.0;
        }

        let rs = rho + 1.0;
        let mut ls = 0;
        while !EXHAUSTIVE && ((1 << ls) as f32) < rs * 0.5 {
            ls += 1;
        }
        let (nl, tex) = (N >> ls, (1 << ls) as f32);
        let lvl = if ls == 0 {
            &self.d[..]
        } else {
            &self.qmax[ls - 1][..]
        };
        let lean = eps0 - if ls > 0 { 0.5 * gabs * tex } else { 0.0 };
        let (taps, step, first) = if EXHAUSTIVE {
            ((2.0 * rs) as usize + 2, 1.0, -rs)
        } else {
            (4, rs * 0.5, -0.75 * rs)
        };
        let (mut nb, mut sum) = (0.0f32, 0.0f32);
        for dj in 0..taps {
            let y = t + first + dj as f32 * step;
            if !(y >= 0.0 && y < nf) {
                continue;
            }
            let j = (y / tex) as usize;
            let py = q + gt * ((j as f32 + 0.5) * tex - t) + lean;
            for di in 0..taps {
                let x = s + first + di as f32 * step;
                if !(x >= 0.0 && x < nf) {
                    continue;
                }
                let i = (x / tex) as usize;
                let qv = lvl[j * nl + i];
                if qv > py + gs * ((i as f32 + 0.5) * tex - s) {
                    nb += 1.0;
                    sum += qv;
                }
            }
        }
        if nb == 0.0 {
            return 1.0;
        }
        let qb = sum / nb;

        let rf = fmin(fmax(self.k * (qb - q), 2.0), rs);
        let mut lf = 0;
        while !EXHAUSTIVE && rf >= (4 << lf) as f32 {
            lf += 1;
        }
        let (nl, tex) = (N >> lf, (1 << lf) as f32);
        let lvl = if lf == 0 {
            &self.d[..]
        } else {
            &self.dec[lf - 1][..]
        };
        let at = if lf == 0 { 0.5 } else { 0.5 * tex + 0.5 };
        let (cs, ct, r) = (s / tex, t / tex, rf / tex);
        let (x0, x1, y0, y1) = (cs - r, cs + r, ct - r, ct + r);
        let (i0, i1) = (x0.floor() as i32, x1.floor() as i32);
        let (j0, j1) = (y0.floor() as i32, y1.floor() as i32);
        let mut acc = 0.0f32;
        for j in j0.max(0)..=j1.min(nl as i32 - 1) {
            let wy = fmin(j as f32 + 1.0, y1) - fmax(j as f32, y0);
            let row = &lvl[j as usize * nl..(j as usize + 1) * nl];
            let py = q + gt * (j as f32 * tex + at - t) + eps0;
            let mut a = 0.0f32;
            for i in i0.max(0)..=i1.min(nl as i32 - 1) {
                if row[i as usize] > py + gs * (i as f32 * tex + at - s) {
                    a += fmin(i as f32 + 1.0, x1) - fmax(i as f32, x0);
                }
            }
            acc += a * wy;
        }
        1.0 - clamp01(acc / (4.0 * r * r))
    }

    /// Coverage and coverage times depth, averaged over about `rho` texels
    /// each way.
    #[inline(always)]
    fn cover(&self, s: f32, t: f32, rho: f32) -> [f32; 2] {
        let lam = log2ish(fmax(rho, 2.0));
        let l = (lam as usize).clamp(1, LEVELS - 1);
        let f = clamp01(lam - l as f32);
        let a = self.cover_at(s, t, l);
        let b = self.cover_at(s, t, l + 1);
        [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]
    }

    #[inline(always)]
    fn cover_at(&self, s: f32, t: f32, l: usize) -> [f32; 2] {
        let n = N >> l;
        let sc = 1.0 / (1 << l) as f32;
        let (x, y) = (s * sc - 0.5, t * sc - 0.5);
        let (xf, yf) = (x.floor(), y.floor());
        let (fx, fy) = (x - xf, y - yf);
        let m = n as i32 - 1;
        let (i0, i1) = (
            (xf as i32).clamp(0, m) as usize,
            (xf as i32 + 1).clamp(0, m) as usize,
        );
        let (j0, j1) = (
            (yf as i32).clamp(0, m) as usize,
            (yf as i32 + 1).clamp(0, m) as usize,
        );
        let c = &self.cov[l - 1];
        let (a, b, d, e) = (
            c[j0 * n + i0],
            c[j0 * n + i1],
            c[j1 * n + i0],
            c[j1 * n + i1],
        );
        let mut o = [0.0f32; 2];
        for k in 0..2 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = d[k] + (e[k] - d[k]) * fx;
            o[k] = top + (bot - top) * fy;
        }
        o
    }

    /// The share of the lamp that reaches a point on the floor, given where
    /// the point is in the map (in texels) and its depth.
    pub fn floor_shadow(&self, s: f32, t: f32, q: f32) -> f32 {
        let nf = N as f32;
        if !(s > -64.0 && s < nf + 64.0 && t > -64.0 && t < nf + 64.0) {
            return 1.0;
        }
        let mut rho = fmin(self.k * (self.qglobal - q), nf * 0.125);
        if rho <= 0.0 {
            return 1.0;
        }
        // Narrow the look to what the blockers found there would need.
        for _ in 0..3 {
            let c = self.cover(s, t, rho);
            if c[0] < 1e-4 {
                return 1.0;
            }
            rho = fmin(fmax(self.k * (c[1] / c[0] - q), 1.5), rho);
        }
        1.0 - clamp01(self.cover(s, t, rho)[0])
    }

    /// Texel coordinates of a tangent pair.
    #[inline(always)]
    pub fn texel(&self, u: f32, v: f32) -> (f32, f32) {
        ((u - self.u0) * self.itau, (v - self.v0) * self.itau)
    }

    pub fn bytes(&self) -> usize {
        let lv: usize = (1..=LEVELS).map(|l| (N >> l) * (N >> l) * 20).sum();
        N * N * 4 + lv + self.uvq.capacity() * 13 + self.set.bytes()
    }
}

/// The faces of the instances of `run` turned away from the lamp, in texels
/// of the window `(u0, v0, 1 / texel)`, with their depth planes.
#[allow(clippy::too_many_arguments)]
fn faces(
    meshes: &[Mesh],
    frame: &Frame,
    vbase: &[u32],
    uvq: &[[f32; 3]],
    (u0, v0, itau): (f32, f32, f32),
    run: Range<usize>,
    set: &mut Setup,
) {
    for (ii, inst) in frame
        .instances
        .iter()
        .enumerate()
        .take(run.end)
        .skip(run.start)
    {
        let Some(mesh) = meshes.get(inst.mesh as usize) else {
            continue;
        };
        let base = vbase[ii] as usize;
        let nv = mesh.pos.len();
        for t in mesh.idx.as_chunks::<3>().0 {
            let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if i0 >= nv || i1 >= nv || i2 >= nv {
                continue;
            }
            let (a, b, c) = (uvq[base + i0], uvq[base + i1], uvq[base + i2]);
            if !(a[2] > 0.0 && b[2] > 0.0 && c[2] > 0.0) {
                continue;
            }
            let (x0, y0) = ((a[0] - u0) * itau, (a[1] - v0) * itau);
            let (x1, y1) = ((b[0] - u0) * itau, (b[1] - v0) * itau);
            let (x2, y2) = ((c[0] - u0) * itau, (c[1] - v0) * itau);
            let (ax, ay, bx, by) = (x1 - x0, y1 - y0, x2 - x0, y2 - y0);
            let area2 = ax * by - bx * ay;
            // Faces turned away from the lamp.
            if !(area2 < 0.0) {
                continue;
            }
            let inv = 1.0 / area2;
            let (d1, d2) = (b[2] - a[2], c[2] - a[2]);
            let dqx = (d1 * by - d2 * ay) * inv;
            let dqy = (ax * d2 - bx * d1) * inv;
            let pl = [x0, y0, a[2], dqx, dqy];
            set.add([[x0, y0], [x1, y1], [x2, y2]], pl, N as i32, 6);
        }
    }
}

/// Draws one row of tiles of the map and builds the levels above it, as far
/// as `BANDED`, from it alone: what `draw` did for those rows when the whole
/// map was drawn at once.
fn band(mut b: Band, set: &Setup) {
    let tn = N >> 6;
    // Empty what the frame before drew; everything else is empty already.
    for tx in 0..tn {
        if b.marks[tx] != 0 {
            for row in b.d.chunks_mut(N) {
                row[tx << 6..(tx + 1) << 6].fill(0.0);
            }
        }
    }
    b.prev.copy_from_slice(&*b.marks);
    b.marks.fill(0);
    let y0 = b.k << 6;
    let mut sink = DepthBand {
        buf: &mut *b.d,
        y0,
        marks: &mut *b.marks,
    };
    let rows = (y0 as i32, y0 as i32 + 64);
    for it in &set.bins[b.k] {
        scan_edges(&mut sink, N as i32, rows, &[(0.5, 0.5)], &it.e, &it.pl);
    }
    // A tile needs building if it holds something now, or did last frame
    // and must go back to empty.
    for (p, m) in b.prev.iter_mut().zip(b.marks.iter()) {
        *p |= *m;
    }
    let d = &*b.d;
    {
        let n1 = N / 2;
        let (mx, mn, cv) = (&mut *b.qmax[0], &mut *b.qmin[0], &mut *b.cov[0]);
        for tx in 0..tn {
            if b.prev[tx] == 0 {
                continue;
            }
            let x0 = tx * 32;
            for y in 0..32 {
                let (r0, r1) = (
                    &d[2 * y * N..(2 * y + 1) * N],
                    &d[(2 * y + 1) * N..(2 * y + 2) * N],
                );
                for x in x0..x0 + 32 {
                    let (a, b, c, d) = (r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]);
                    let o = y * n1 + x;
                    mx[o] = fmax(fmax(a, b), fmax(c, d));
                    mn[o] = fmin(fmin(a, b), fmin(c, d));
                    let cnt =
                        (a > 0.0) as u32 + (b > 0.0) as u32 + (c > 0.0) as u32 + (d > 0.0) as u32;
                    cv[o] = [cnt as f32 * 0.25, (a + b + c + d) * 0.25];
                }
            }
        }
    }
    for lv in 1..BANDED {
        let n = N >> (lv + 1);
        let (lo, hi) = b.qmax.split_at_mut(lv);
        let (lo2, hi2) = b.qmin.split_at_mut(lv);
        let (lo3, hi3) = b.cov.split_at_mut(lv);
        let (smx, smn, scv) = (&*lo[lv - 1], &*lo2[lv - 1], &*lo3[lv - 1]);
        let (dmx, dmn, dcv) = (&mut *hi[0], &mut *hi2[0], &mut *hi3[0]);
        let sn = n * 2;
        let ts = 64 >> (lv + 1);
        for tx in 0..tn {
            if b.prev[tx] == 0 {
                continue;
            }
            let x0 = tx * ts;
            for y in 0..ts {
                for x in x0..x0 + ts {
                    let (i, o) = (2 * y * sn + 2 * x, y * n + x);
                    dmx[o] = fmax(fmax(smx[i], smx[i + 1]), fmax(smx[i + sn], smx[i + sn + 1]));
                    dmn[o] = fmin(fmin(smn[i], smn[i + 1]), fmin(smn[i + sn], smn[i + sn + 1]));
                    let (a, b, c, d) = (scv[i], scv[i + 1], scv[i + sn], scv[i + sn + 1]);
                    dcv[o] = [
                        (a[0] + b[0] + c[0] + d[0]) * 0.25,
                        (a[1] + b[1] + c[1] + d[1]) * 0.25,
                    ];
                }
            }
        }
    }
    for lv in 1..=BANDED {
        let (n, ts, half) = (N >> lv, 64 >> lv, 1usize << (lv - 1));
        let dst = &mut *b.dec[lv - 1];
        for tx in 0..tn {
            if b.prev[tx] == 0 {
                continue;
            }
            let x0 = tx * ts;
            for y in 0..ts {
                let src = &d[((y << lv) + half) * N..];
                for x in x0..x0 + ts {
                    dst[y * n + x] = src[(x << lv) + half];
                }
            }
        }
    }
}
