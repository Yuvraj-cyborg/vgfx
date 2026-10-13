//! Two things done over the whole picture: how open each point is to the
//! room (ambient occlusion, at half size), and the finish (bloom, exposure,
//! tone curve, vignette, dither, sRGB).

use super::raster::{Tiles, Tri, dilate};
use super::{clamp01, expose, fmax, fmin};
use crate::studio::math::{TAU, V3, cos, sin};

const TAPS: usize = 10;
/// How far a surface looks for something that closes it in, in world units.
const REACH: f32 = 0.55;

pub struct Ao {
    pub w: usize,
    pub h: usize,
    /// Depth along the view at every second pixel; the floor where no mesh is.
    pub z: Vec<f32>,
    pub ao: Vec<f32>,
    tmp: Vec<f32>,
    /// Tiles within four, two and one of a tile that holds a mesh.
    m4: Vec<u8>,
    m2: Vec<u8>,
    m1: Vec<u8>,
    ms: Vec<u8>,
    /// Ten points on a spiral, at sixteen turns.
    taps: Vec<[(f32, f32); TAPS]>,
}

impl Ao {
    pub fn new(w: usize, h: usize) -> Ao {
        let mut taps = Vec::new();
        for r in 0..16 {
            let mut t = [(0.0, 0.0); TAPS];
            for (k, t) in t.iter_mut().enumerate() {
                let rad = ((k as f32 + 0.5) / TAPS as f32).sqrt();
                let ang = (k as f32 * 0.381_966 + r as f32 / 16.0) * TAU;
                *t = (cos(ang) * rad, sin(ang) * rad);
            }
            taps.push(t);
        }
        Ao {
            w,
            h,
            z: vec![0.0; w * h],
            ao: vec![1.0; w * h],
            tmp: vec![1.0; w * h],
            m4: Vec::new(),
            m2: Vec::new(),
            m1: Vec::new(),
            ms: Vec::new(),
            taps,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        keys: &[u64],
        tris: &[Tri],
        tiles: &Tiles,
        fw: usize,
        fh: usize,
        right: V3,
        up: V3,
        fwd: V3,
        fpx: f32,
        fq: [f32; 3],
    ) {
        const ORDER: [u8; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
        if !tiles.f.iter().any(|&f| f != 0) {
            return;
        }
        let (w, h) = (self.w, self.h);
        let (tw, th) = (tiles.tw, tiles.th);
        dilate(&tiles.f, tw, th, 4, &mut self.ms, &mut self.m4);
        dilate(&tiles.f, tw, th, 2, &mut self.ms, &mut self.m2);
        dilate(&tiles.f, tw, th, 1, &mut self.ms, &mut self.m1);
        let span = |t: usize, n: usize| (t << 4)..((t + 1) << 4).min(n);
        for (ty, tx) in (0..th).flat_map(|ty| (0..tw).map(move |tx| (ty, tx))) {
            if self.m4[ty * tw + tx] == 0 {
                continue;
            }
            for j in span(ty, h) {
                let y = (2 * j).min(fh - 1);
                for i in span(tx, w) {
                    let x = (2 * i).min(fw - 1);
                    let k = if tiles.f[(y >> 5) * tw + (x >> 5)] != 0 {
                        keys[y * fw + x]
                    } else {
                        0
                    };
                    let o = j * w + i;
                    if k as u32 != 0 {
                        self.z[o] = 1.0 / f32::from_bits((k >> 32) as u32);
                    } else {
                        let q = fq[0] + fq[1] * (x as f32 + 0.375) + fq[2] * (y as f32 + 0.125);
                        self.z[o] = if q > 1e-6 { 1.0 / q } else { 1e6 };
                        self.tmp[o] = 1.0;
                    }
                }
            }
        }
        let inv_f = 1.0 / fpx;
        let (cx, cy) = (fw as f32 * 0.5 - 0.375, fh as f32 * 0.5 - 0.125);
        for (ty, tx) in (0..th).flat_map(|ty| (0..tw).map(move |tx| (ty, tx))) {
            if tiles.f[ty * tw + tx] == 0 {
                continue;
            }
            for j in span(ty, h) {
                let y = (2 * j).min(fh - 1);
                for i in span(tx, w) {
                    let x = (2 * i).min(fw - 1);
                    let o = j * w + i;
                    let id = keys[y * fw + x] as u32;
                    if id == 0 {
                        continue;
                    }
                    let ng = tris[id as usize - 1].ng;
                    let n = V3 {
                        x: ng.dot(right),
                        y: -ng.dot(up),
                        z: ng.dot(fwd),
                    };
                    let z = self.z[o];
                    let p = V3 {
                        x: (x as f32 - cx) * inv_f * z,
                        y: (y as f32 - cy) * inv_f * z,
                        z,
                    };
                    let rp = fmin(fmax(REACH * fpx * 0.5 / z, 2.0), 48.0);
                    let taps = &self.taps[ORDER[(j & 3) * 4 + (i & 3)] as usize];
                    let bias = 0.012 * z;
                    let mut occ = 0.0f32;
                    for &(tx, ty) in taps {
                        let ti =
                            ((i as f32 + tx * rp + 0.5) as i32).clamp(0, w as i32 - 1) as usize;
                        let tj =
                            ((j as f32 + ty * rp + 0.5) as i32).clamp(0, h as i32 - 1) as usize;
                        let zt = self.z[tj * w + ti];
                        let v = V3 {
                            x: ((2 * ti) as f32 - cx) * inv_f * zt,
                            y: ((2 * tj) as f32 - cy) * inv_f * zt,
                            z: zt,
                        } - p;
                        let vv = v.dot(v);
                        let vn = v.dot(n) - bias;
                        if vn > 0.0 && vv < REACH * REACH {
                            let f = 1.0 - vv * (1.0 / (REACH * REACH));
                            occ += vn / (vv + 1e-4) * f * f;
                        }
                    }
                    self.tmp[o] = occ * (REACH * 1.1 / TAPS as f32);
                }
            }
        }
        for pass in 0..2 {
            let (src, dst, step, mask) = if pass == 0 {
                (&self.tmp, &mut self.ao, 1, &self.m2)
            } else {
                (&self.ao, &mut self.tmp, w, &self.m1)
            };
            let lim = if pass == 0 { w } else { h };
            for (ty, tx) in (0..th).flat_map(|ty| (0..tw).map(move |tx| (ty, tx))) {
                if mask[ty * tw + tx] == 0 {
                    continue;
                }
                for j in span(ty, h) {
                    for i in span(tx, w) {
                        let o = j * w + i;
                        let z = self.z[o];
                        let at = if pass == 0 { i } else { j };
                        let iz = 12.0 / z;
                        let (mut s, mut ws) = (0.0f32, 0.0f32);
                        for (k, wk) in [1.0f32, 2.0, 2.0, 2.0, 1.0].iter().enumerate() {
                            let a = at + k;
                            if a < 2 || a >= lim + 2 {
                                continue;
                            }
                            let oo = o + k * step - 2 * step;
                            let wt = wk * fmax(1.0 - (self.z[oo] - z).abs() * iz, 0.0);
                            s += src[oo] * wt;
                            ws += wt;
                        }
                        dst[o] = s / fmax(ws, 1e-6);
                    }
                }
            }
        }
        for (ty, tx) in (0..th).flat_map(|ty| (0..tw).map(move |tx| (ty, tx))) {
            if self.m1[ty * tw + tx] == 0 {
                continue;
            }
            for j in span(ty, h) {
                for i in span(tx, w) {
                    let o = j * w + i;
                    let a = clamp01(1.0 - self.tmp[o]);
                    self.ao[o] = a * a * (3.0 - 2.0 * a) * 0.5 + a * 0.5;
                }
            }
        }
    }

    #[inline(always)]
    pub fn at(&self, x: usize, y: usize, z: f32) -> f32 {
        let (w, h) = (self.w, self.h);
        let (i0, j0) = ((x / 2).min(w - 1), (y / 2).min(h - 1));
        let (i1, j1) = ((i0 + 1).min(w - 1), (j0 + 1).min(h - 1));
        let (fx, fy) = ((x & 1) as f32 * 0.5, (y & 1) as f32 * 0.5);
        let iz = 12.0 / z;
        let (mut s, mut ws) = (0.0f32, 0.0f32);
        for (o, wb) in [
            (j0 * w + i0, (1.0 - fx) * (1.0 - fy)),
            (j0 * w + i1, fx * (1.0 - fy)),
            (j1 * w + i0, (1.0 - fx) * fy),
            (j1 * w + i1, fx * fy),
        ] {
            let wt = wb * fmax(1.0 - (self.z[o] - z).abs() * iz, 0.01);
            s += self.ao[o] * wt;
            ws += wt;
        }
        s / ws
    }
}

const NLUT: usize = 2048;
const UMAX: f32 = 8.0;
const BLEV: usize = 6;

pub struct Post {
    lut: Vec<f32>,
    bl: Vec<Vec<[f32; 3]>>,
    tmp: Vec<[f32; 3]>,
    dims: Vec<(usize, usize)>,
    cols: Vec<(u32, u32, f32, f32)>,
    rows: Vec<[f32; 3]>,
}

fn root_2_4(x: f32) -> f32 {
    let z = x * x * x * x * x;
    let mut y = 1.0f32;
    for _ in 0..64 {
        y = (2.0 * y + z / (y * y)) * (1.0 / 3.0);
    }
    y.sqrt().sqrt()
}

impl Post {
    pub fn new(w: usize, h: usize) -> Post {
        let mut lut = Vec::with_capacity(NLUT + 2);
        for i in 0..NLUT + 2 {
            let u = i as f32 * (UMAX / NLUT as f32);
            let x = u * u;
            let y = clamp01((x * (0.399 * x + 0.105)) / (x * (0.3874 * x + 0.1926) + 0.14));
            let s = if y <= 0.003_130_8 {
                12.92 * y
            } else {
                1.055 * root_2_4(y) - 0.055
            };
            lut.push(s * 255.0);
        }
        let (mut bl, mut dims) = (Vec::new(), Vec::new());
        let (mut lw, mut lh) = (w.div_ceil(4), h.div_ceil(4));
        for _ in 0..BLEV {
            bl.push(vec![[0.0; 3]; lw * lh]);
            dims.push((lw, lh));
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
        }
        Post {
            lut,
            tmp: vec![[0.0; 3]; dims[0].0 * dims[0].1],
            bl,
            dims,
            cols: Vec::new(),
            rows: Vec::new(),
        }
    }

    pub fn bytes(&self) -> usize {
        self.bl.iter().map(|l| l.len() * 12).sum::<usize>()
            + self.tmp.len() * 12
            + self.lut.len() * 4
    }

    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        hdr: &mut [[f32; 3]],
        w: usize,
        h: usize,
        exposure: f32,
        bloom: f32,
        open: &[u8],
        tiles: &Tiles,
        bright: &[u8],
        ns: usize,
        out: &mut [u8],
    ) {
        let bloom = clamp01(bloom);
        let (bw, bh) = self.dims[0];
        let (tw, sh) = (tiles.tw, tiles.sh);
        for (y, tx) in (0..h).flat_map(|y| (0..tw).map(move |tx| (y, tx))) {
            if tiles.f[(y >> sh) * tw + tx] == 0 {
                continue;
            }
            for i in y * w + (tx << sh)..y * w + ((tx + 1) << sh).min(w) {
                if open[i] as usize != ns {
                    hdr[i] = expose(hdr[i], exposure);
                }
            }
        }
        let mut has_bloom = false;
        if bloom > 0.0 && bright.iter().any(|&f| f != 0) {
            self.bl[0].fill([0.0; 3]);
            for y in 0..h {
                let brow = &mut self.bl[0][(y / 4) * bw..(y / 4 + 1) * bw];
                for tx in 0..tw {
                    if bright[(y >> sh) * tw + tx] == 0 {
                        continue;
                    }
                    for x in (tx << sh)..((tx + 1) << sh).min(w) {
                        let c = hdr[y * w + x];
                        let b = fmax(c[0], fmax(c[1], c[2]));
                        if b > 1.0 {
                            let t = fmin((b - 1.0) * 0.5, 1.0) * fmin(1.0, 5.0 / b);
                            let o = &mut brow[x / 4];
                            o[0] += c[0] * t;
                            o[1] += c[1] * t;
                            o[2] += c[2] * t;
                            has_bloom = true;
                        }
                    }
                }
            }
        }
        if has_bloom {
            for l in 1..BLEV {
                let (lo, hi) = self.bl.split_at_mut(l);
                down(&lo[l - 1], self.dims[l - 1], &mut hi[0], self.dims[l]);
            }
            let spread = 0.45 + 0.75 * bloom;
            let mut wt = [0.0f32; BLEV];
            let (mut k, mut sum) = (1.0f32, 0.0f32);
            for w in wt.iter_mut() {
                *w = k;
                sum += k;
                k *= spread;
            }
            let gain = (0.05 + 0.35 * bloom) / (16.0 * sum);
            for l in (0..BLEV - 1).rev() {
                let (lo, hi) = self.bl.split_at_mut(l + 1);
                let (dst, src) = (&mut lo[l], &hi[0]);
                let ((dw, dh), (sw, sh)) = (self.dims[l], self.dims[l + 1]);
                let top = if l == BLEV - 2 { wt[BLEV - 1] } else { 1.0 };
                for y in 0..dh {
                    let fy = y as f32 * 0.5 - 0.25;
                    let (y0, ty) = (fy.floor(), fy - fy.floor());
                    let (ya, yb) = (
                        (y0 as i32).clamp(0, sh as i32 - 1) as usize,
                        (y0 as i32 + 1).clamp(0, sh as i32 - 1) as usize,
                    );
                    for x in 0..dw {
                        let fx = x as f32 * 0.5 - 0.25;
                        let (x0, tx) = (fx.floor(), fx - fx.floor());
                        let (xa, xb) = (
                            (x0 as i32).clamp(0, sw as i32 - 1) as usize,
                            (x0 as i32 + 1).clamp(0, sw as i32 - 1) as usize,
                        );
                        let d = &mut dst[y * dw + x];
                        for c in 0..3 {
                            let a = src[ya * sw + xa][c]
                                + (src[ya * sw + xb][c] - src[ya * sw + xa][c]) * tx;
                            let b = src[yb * sw + xa][c]
                                + (src[yb * sw + xb][c] - src[yb * sw + xa][c]) * tx;
                            d[c] = d[c] * wt[l] + (a + (b - a) * ty) * top;
                        }
                    }
                }
            }
            let b0 = &mut self.bl[0];
            for y in 0..bh {
                for x in 0..bw {
                    let (l, r) = (x.saturating_sub(1), (x + 1).min(bw - 1));
                    for c in 0..3 {
                        self.tmp[y * bw + x][c] =
                            (b0[y * bw + l][c] + 2.0 * b0[y * bw + x][c] + b0[y * bw + r][c])
                                * 0.25;
                    }
                }
            }
            for y in 0..bh {
                let (u, d) = (y.saturating_sub(1), (y + 1).min(bh - 1));
                for x in 0..bw {
                    for c in 0..3 {
                        b0[y * bw + x][c] = (self.tmp[u * bw + x][c]
                            + 2.0 * self.tmp[y * bw + x][c]
                            + self.tmp[d * bw + x][c])
                            * 0.25
                            * gain;
                    }
                }
            }
        }
        const BAYER: [f32; 16] = [
            0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.,
        ];
        let b0 = &self.bl[0];
        let lut = &self.lut[..];
        let (iw, ih) = (2.0 / w as f32, 2.0 / h as f32);
        let us = NLUT as f32 / UMAX;
        self.cols.clear();
        for x in 0..w {
            let fx = (x as f32 + 0.5) * 0.25 - 0.5;
            let (x0, tx) = (fx.floor(), fx - fx.floor());
            let (xa, xb) = (
                (x0 as i32).clamp(0, bw as i32 - 1) as usize,
                (x0 as i32 + 1).clamp(0, bw as i32 - 1) as usize,
            );
            let ux = (x as f32 + 0.5) * iw - 1.0;
            self.cols.push((xa as u32, xb as u32, tx, ux * ux * 0.8));
        }
        self.rows.resize(3 * w, [0.0; 3]);
        let (ra, rest) = self.rows.split_at_mut(w);
        let (rb, ru) = rest.split_at_mut(w);
        let mut held = (usize::MAX, usize::MAX);
        for y in 0..h {
            let fy = (y as f32 + 0.5) * 0.25 - 0.5;
            let (y0, ty) = (fy.floor(), fy - fy.floor());
            let (ya, yb) = (
                (y0 as i32).clamp(0, bh as i32 - 1) as usize * bw,
                (y0 as i32 + 1).clamp(0, bh as i32 - 1) as usize * bw,
            );
            let uy = (y as f32 + 0.5) * ih - 1.0;
            let uy2 = uy * uy * 0.6;
            if has_bloom && held != (ya, yb) {
                held = (ya, yb);
                for (x, &(xa, xb, tx, _)) in self.cols.iter().enumerate() {
                    let (xa, xb) = (xa as usize, xb as usize);
                    for k in 0..3 {
                        ra[x][k] = b0[ya + xa][k] + (b0[ya + xb][k] - b0[ya + xa][k]) * tx;
                        rb[x][k] = b0[yb + xa][k] + (b0[yb + xb][k] - b0[yb + xa][k]) * tx;
                    }
                }
            }
            let src = &hdr[y * w..(y + 1) * w];
            for x in 0..w {
                let mut c = src[x];
                if has_bloom {
                    for k in 0..3 {
                        c[k] += ra[x][k] + (rb[x][k] - ra[x][k]) * ty;
                    }
                }
                let vig = 1.0 - 0.20 * (self.cols[x].3 + uy2);
                for k in 0..3 {
                    ru[x][k] = fmin(fmax(c[k] * vig, 0.0).sqrt() * us, NLUT as f32);
                }
            }
            let orow = &mut out[y * w * 3..(y + 1) * w * 3];
            for (x, (u3, o)) in ru.iter().zip(orow.as_chunks_mut::<3>().0).enumerate() {
                let dith = (BAYER[(y & 3) * 4 + (x & 3)] + 0.5) * (1.0 / 16.0);
                for k in 0..3 {
                    let u = u3[k];
                    let i = u as usize;
                    let v = lut[i] + (lut[i + 1] - lut[i]) * (u - i as f32);
                    o[k] = (v + dith) as u8;
                }
            }
        }
    }
}

fn down(
    src: &[[f32; 3]],
    (sw, sh): (usize, usize),
    dst: &mut [[f32; 3]],
    (dw, dh): (usize, usize),
) {
    const K: [f32; 4] = [0.125, 0.375, 0.375, 0.125];
    for y in 0..dh {
        for x in 0..dw {
            let mut o = [0.0f32; 3];
            for (j, kj) in K.iter().enumerate() {
                let yy = ((2 * y + j) as i32 - 1).clamp(0, sh as i32 - 1) as usize;
                for (i, ki) in K.iter().enumerate() {
                    let xx = ((2 * x + i) as i32 - 1).clamp(0, sw as i32 - 1) as usize;
                    let (s, k) = (&src[yy * sw + xx], kj * ki);
                    o[0] += s[0] * k;
                    o[1] += s[1] * k;
                    o[2] += s[2] * k;
                }
            }
            dst[y * dw + x] = o;
        }
    }
}
