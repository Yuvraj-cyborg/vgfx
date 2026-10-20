//! Two things done over the whole picture: how open each point is to the
//! room (ambient occlusion, at half size), and the finish (bloom, exposure,
//! tone curve, vignette, dither, sRGB).

use super::par;
use super::raster::{Tiles, Tri, dilate};
use super::{clamp01, expose, fmax, fmin};
use crate::studio::math::{TAU, V3, cos, sin};
use std::sync::atomic::{AtomicBool, Ordering};

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
        self.compute_with(keys, fw, tris, tiles, fw, fh, right, up, fwd, fpx, fq, 1);
    }

    /// `compute` on up to `threads` threads, a row of tiles at a time; the
    /// result is the same. The key of pixel (x, y) is `keys[y * stride + x]`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_with(
        &mut self,
        keys: &[u64],
        stride: usize,
        tris: &[Tri],
        tiles: &Tiles,
        fw: usize,
        fh: usize,
        right: V3,
        up: V3,
        fwd: V3,
        fpx: f32,
        fq: [f32; 3],
        threads: usize,
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
        // A row of tiles is sixteen rows here; `rows` of the buffers.
        let rows = w << 4;
        let span = |t: usize, n: usize| (t << 4)..((t + 1) << 4).min(n);
        // Depth: what each point is, from the picture's keys.
        let m4 = &self.m4;
        let bands = self
            .z
            .chunks_mut(rows)
            .zip(self.tmp.chunks_mut(rows))
            .enumerate();
        par::each0(threads, bands, |(ty, (z, tmp))| {
            for tx in 0..tw {
                if m4[ty * tw + tx] == 0 {
                    continue;
                }
                for j in span(ty, h) {
                    let y = (2 * j).min(fh - 1);
                    for i in span(tx, w) {
                        let x = (2 * i).min(fw - 1);
                        let k = if tiles.f[(y >> 5) * tw + (x >> 5)] != 0 {
                            keys[y * stride + x]
                        } else {
                            0
                        };
                        let o = (j - (ty << 4)) * w + i;
                        if k as u32 != 0 {
                            z[o] = 1.0 / f32::from_bits((k >> 32) as u32);
                        } else {
                            let q = fq[0] + fq[1] * (x as f32 + 0.375) + fq[2] * (y as f32 + 0.125);
                            z[o] = if q > 1e-6 { 1.0 / q } else { 1e6 };
                            tmp[o] = 1.0;
                        }
                    }
                }
            }
        });
        // How closed in each point is, then that blurred along the row.
        let inv_f = 1.0 / fpx;
        let (cx, cy) = (fw as f32 * 0.5 - 0.375, fh as f32 * 0.5 - 0.125);
        let (zs, taps, m2) = (&self.z[..], &self.taps, &self.m2);
        let bands = self
            .tmp
            .chunks_mut(rows)
            .zip(self.ao.chunks_mut(rows))
            .enumerate();
        par::each0(threads, bands, |(ty, (tmp, ao))| {
            let base = (ty << 4) * w;
            for tx in 0..tw {
                if tiles.f[ty * tw + tx] == 0 {
                    continue;
                }
                for j in span(ty, h) {
                    let y = (2 * j).min(fh - 1);
                    for i in span(tx, w) {
                        let x = (2 * i).min(fw - 1);
                        let o = j * w + i;
                        let id = keys[y * stride + x] as u32;
                        if id == 0 {
                            continue;
                        }
                        let ng = tris[id as usize - 1].ng;
                        let n = V3 {
                            x: ng.dot(right),
                            y: -ng.dot(up),
                            z: ng.dot(fwd),
                        };
                        let z = zs[o];
                        let p = V3 {
                            x: (x as f32 - cx) * inv_f * z,
                            y: (y as f32 - cy) * inv_f * z,
                            z,
                        };
                        let rp = fmin(fmax(REACH * fpx * 0.5 / z, 2.0), 48.0);
                        let taps = &taps[ORDER[(j & 3) * 4 + (i & 3)] as usize];
                        let bias = 0.012 * z;
                        let mut occ = 0.0f32;
                        for &(tx, ty) in taps {
                            let ti =
                                ((i as f32 + tx * rp + 0.5) as i32).clamp(0, w as i32 - 1) as usize;
                            let tj =
                                ((j as f32 + ty * rp + 0.5) as i32).clamp(0, h as i32 - 1) as usize;
                            let zt = zs[tj * w + ti];
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
                        tmp[o - base] = occ * (REACH * 1.1 / TAPS as f32);
                    }
                }
            }
            blur(tmp, base, ao, base, zs, ty, tw, w, h, 1, m2);
        });
        // Blurred down the columns, from `ao` back into `tmp`: a band reads
        // two rows either side of its own.
        let (aos, m1) = (&self.ao[..], &self.m1);
        let bands = self.tmp.chunks_mut(rows).enumerate();
        par::each0(threads, bands, |(ty, tmp)| {
            blur(aos, 0, tmp, (ty << 4) * w, zs, ty, tw, w, h, w, m1);
        });
        let tmps = &self.tmp[..];
        par::each0(threads, self.ao.chunks_mut(rows).enumerate(), |(ty, ao)| {
            let base = (ty << 4) * w;
            for tx in 0..tw {
                if m1[ty * tw + tx] == 0 {
                    continue;
                }
                for j in span(ty, h) {
                    let r = span(tx, w);
                    let (src, dst) = (
                        &tmps[j * w + r.start..j * w + r.end],
                        &mut ao[j * w - base + r.start..j * w - base + r.end],
                    );
                    for (o, t) in dst.iter_mut().zip(src) {
                        let a = clamp01(1.0 - t);
                        *o = a * a * (3.0 - 2.0 * a) * 0.5 + a * 0.5;
                    }
                }
            }
        });
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

/// One pass of the depth-aware blur, over the tiles of row `ty` marked in
/// `mask`: along the rows when `step` is 1, down the columns when it is `w`.
/// `src` holds the picture from index `sbase` on, `dst` the band's rows from
/// `dbase`.
#[allow(clippy::too_many_arguments)]
fn blur(
    src: &[f32],
    sbase: usize,
    dst: &mut [f32],
    dbase: usize,
    z: &[f32],
    ty: usize,
    tw: usize,
    w: usize,
    h: usize,
    step: usize,
    mask: &[u8],
) {
    const WK: [f32; 5] = [1.0, 2.0, 2.0, 2.0, 1.0];
    let span = |t: usize, n: usize| (t << 4)..((t + 1) << 4).min(n);
    let lim = if step == 1 { w } else { h };
    // A pixel whose five taps all fall in the picture.
    let whole = |at: usize| at >= 2 && at + 2 < lim;
    for tx in 0..tw {
        if mask[ty * tw + tx] == 0 {
            continue;
        }
        for j in span(ty, h) {
            let cols = span(tx, w);
            // Along the row, the pixels with every tap: value by value, so
            // the compiler does several at once.
            let (i0, i1) = if step == 1 {
                (
                    cols.start.max(2),
                    cols.end.min(w.saturating_sub(2)).max(cols.start.max(2)),
                )
            } else if whole(j) {
                (cols.start, cols.end)
            } else {
                (cols.start, cols.start)
            };
            if i1 > i0 {
                let (o0, n) = (j * w + i0, i1 - i0);
                let zc = &z[o0..o0 + n];
                let at = |k: usize| o0 + k * step - 2 * step;
                let zt: [&[f32]; 5] = core::array::from_fn(|k| &z[at(k)..at(k) + n]);
                let st: [&[f32]; 5] =
                    core::array::from_fn(|k| &src[at(k) - sbase..at(k) - sbase + n]);
                let out = &mut dst[o0 - dbase..o0 - dbase + n];
                for i in 0..n {
                    let zo = zc[i];
                    let iz = 12.0 / zo;
                    let (mut s, mut ws) = (0.0f32, 0.0f32);
                    for k in 0..5 {
                        let wt = WK[k] * fmax(1.0 - (zt[k][i] - zo).abs() * iz, 0.0);
                        s += st[k][i] * wt;
                        ws += wt;
                    }
                    out[i] = s / fmax(ws, 1e-6);
                }
            }
            // The rest, tap by tap.
            for i in cols.filter(|&i| i < i0 || i >= i1) {
                let o = j * w + i;
                let zo = z[o];
                let at = if step == 1 { i } else { j };
                let iz = 12.0 / zo;
                let (mut s, mut ws) = (0.0f32, 0.0f32);
                for (k, wk) in WK.iter().enumerate() {
                    let a = at + k;
                    if a < 2 || a >= lim + 2 {
                        continue;
                    }
                    let oo = o + k * step - 2 * step;
                    let wt = wk * fmax(1.0 - (z[oo] - zo).abs() * iz, 0.0);
                    s += src[oo - sbase] * wt;
                    ws += wt;
                }
                dst[o - dbase] = s / fmax(ws, 1e-6);
            }
        }
    }
}

const NLUT: usize = 2048;
const UMAX: f32 = 8.0;
const BLEV: usize = 6;

pub struct Post {
    /// The tone curve at `NLUT + 1` points, each with the step to the next.
    lut: Box<[[f32; 2]; NLUT + 1]>,
    bl: Vec<Vec<[f32; 3]>>,
    tmp: Vec<[f32; 3]>,
    dims: Vec<(usize, usize)>,
    cols: Vec<(u32, u32, f32)>,
    /// For each value of a row of pixels (three to a pixel): its column's
    /// share of the vignette, and its dither on each of four rows.
    vig: Vec<f32>,
    dith: [Vec<f32>; 4],
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
            lut: Box::new(core::array::from_fn(|i| [lut[i], lut[i + 1] - lut[i]])),
            tmp: vec![[0.0; 3]; dims[0].0 * dims[0].1],
            bl,
            dims,
            cols: Vec::new(),
            vig: Vec::new(),
            dith: Default::default(),
            rows: Vec::new(),
        }
    }

    pub fn bytes(&self) -> usize {
        self.bl.iter().map(|l| l.len() * 12).sum::<usize>()
            + self.tmp.len() * 12
            + self.lut.len() * 8
    }

    /// The whole finish, on one thread: `gather` every band, `spread` the
    /// bloom if there is any, `tone` every band.
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
        let has_bloom = self.gather_all(hdr, w, exposure, bloom, open, tiles, bright, ns, 1);
        if has_bloom {
            self.spread(bloom, 1);
        }
        self.columns(w, h);
        let mut rows = [core::mem::take(&mut self.rows)];
        self.tone_all(hdr, w, h, tiles.sh, has_bloom, out, 1, &mut rows);
        let [rows] = rows;
        self.rows = rows;
    }

    /// `gather` for every band of rows (one row of `tiles` high), on up to
    /// `threads` threads. Says whether anything was bright.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_all(
        &mut self,
        hdr: &mut [[f32; 3]],
        w: usize,
        exposure: f32,
        bloom: f32,
        open: &[u8],
        tiles: &Tiles,
        bright: &[u8],
        ns: usize,
        threads: usize,
    ) -> bool {
        let (bw, sh) = (self.dims[0].0, tiles.sh);
        let rows = w << sh;
        let any = AtomicBool::new(false);
        let bands = hdr
            .chunks_mut(rows)
            .zip(open.chunks(rows))
            .zip(tiles.f.chunks(tiles.tw).zip(bright.chunks(tiles.tw)))
            .zip(self.bl[0].chunks_mut(bw << (sh - 2)))
            .enumerate();
        par::each0(
            threads,
            bands,
            |(k, (((hdr, open), (marks, bright)), b0))| {
                if gather(
                    hdr,
                    k << sh,
                    w,
                    exposure,
                    bloom,
                    open,
                    marks,
                    bright,
                    sh,
                    ns,
                    b0,
                    bw,
                ) {
                    any.store(true, Ordering::Relaxed);
                }
            },
        );
        any.into_inner()
    }

    /// `tone` for every band of `1 << sh` rows, on up to `threads` threads,
    /// each with one of `rows` for scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn tone_all(
        &self,
        hdr: &[[f32; 3]],
        w: usize,
        h: usize,
        sh: usize,
        has_bloom: bool,
        out: &mut [u8],
        threads: usize,
        rows: &mut [Vec<[f32; 3]>],
    ) {
        let n = w << sh;
        let bands = hdr.chunks(n).zip(out.chunks_mut(n * 3)).enumerate();
        par::each(threads, bands, rows, |rows, (k, (hdr, out))| {
            self.tone(hdr, k << sh, w, h, has_bloom, out, rows);
        });
    }

    /// Spreads what `gather` found bright over the levels of the bloom, the
    /// larger levels on up to `threads` threads, some rows to each.
    pub fn spread(&mut self, bloom: f32, threads: usize) {
        const ROWS: usize = 8;
        let bloom = clamp01(bloom);
        let (bw, bh) = self.dims[0];
        // Only a level this large is worth sharing out.
        let share = |(w, h): (usize, usize)| if w * h >= 1 << 15 { threads } else { 1 };
        for l in 1..BLEV {
            let (lo, hi) = self.bl.split_at_mut(l);
            let (src, sd, dd) = (&lo[l - 1][..], self.dims[l - 1], self.dims[l]);
            let pieces = hi[0].chunks_mut(dd.0 * ROWS).enumerate();
            par::each0(share(dd), pieces, |(k, rows)| {
                down(src, sd, rows, dd, k * ROWS)
            });
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
            let (dst, src) = (&mut lo[l], &hi[0][..]);
            let ((dw, dh), (sw, sh)) = (self.dims[l], self.dims[l + 1]);
            let top = if l == BLEV - 2 { wt[BLEV - 1] } else { 1.0 };
            let wl = wt[l];
            let pieces = dst.chunks_mut(dw * ROWS).enumerate();
            par::each0(share((dw, dh)), pieces, |(k, rows)| {
                for (yl, drow) in rows.chunks_mut(dw).enumerate() {
                    let y = k * ROWS + yl;
                    let fy = y as f32 * 0.5 - 0.25;
                    let (y0, ty) = (fy.floor(), fy - fy.floor());
                    let (ya, yb) = (
                        (y0 as i32).clamp(0, sh as i32 - 1) as usize,
                        (y0 as i32 + 1).clamp(0, sh as i32 - 1) as usize,
                    );
                    for (x, d) in drow.iter_mut().enumerate() {
                        let fx = x as f32 * 0.5 - 0.25;
                        let (x0, tx) = (fx.floor(), fx - fx.floor());
                        let (xa, xb) = (
                            (x0 as i32).clamp(0, sw as i32 - 1) as usize,
                            (x0 as i32 + 1).clamp(0, sw as i32 - 1) as usize,
                        );
                        for c in 0..3 {
                            let a = src[ya * sw + xa][c]
                                + (src[ya * sw + xb][c] - src[ya * sw + xa][c]) * tx;
                            let b = src[yb * sw + xa][c]
                                + (src[yb * sw + xb][c] - src[yb * sw + xa][c]) * tx;
                            d[c] = d[c] * wl + (a + (b - a) * ty) * top;
                        }
                    }
                }
            });
        }
        let n = share((bw, bh));
        let b0 = &self.bl[0][..];
        par::each0(
            n,
            self.tmp.chunks_mut(bw * ROWS).enumerate(),
            |(k, rows)| {
                for (yl, trow) in rows.chunks_mut(bw).enumerate() {
                    let y = k * ROWS + yl;
                    for (x, t) in trow.iter_mut().enumerate() {
                        let (l, r) = (x.saturating_sub(1), (x + 1).min(bw - 1));
                        for c in 0..3 {
                            t[c] =
                                (b0[y * bw + l][c] + 2.0 * b0[y * bw + x][c] + b0[y * bw + r][c])
                                    * 0.25;
                        }
                    }
                }
            },
        );
        let tmp = &self.tmp[..];
        par::each0(
            n,
            self.bl[0].chunks_mut(bw * ROWS).enumerate(),
            |(k, rows)| {
                for (yl, brow) in rows.chunks_mut(bw).enumerate() {
                    let y = k * ROWS + yl;
                    let (u, d) = (y.saturating_sub(1), (y + 1).min(bh - 1));
                    for (x, b) in brow.iter_mut().enumerate() {
                        for c in 0..3 {
                            b[c] = (tmp[u * bw + x][c]
                                + 2.0 * tmp[y * bw + x][c]
                                + tmp[d * bw + x][c])
                                * 0.25
                                * gain;
                        }
                    }
                }
            },
        );
    }

    /// Where each column of the picture reads the bloom, and its share of
    /// the vignette: what `tone` needs, worked out once a frame.
    pub fn columns(&mut self, w: usize, _h: usize) {
        const BAYER: [f32; 16] = [
            0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.,
        ];
        let bw = self.dims[0].0;
        let iw = 2.0 / w as f32;
        self.cols.clear();
        self.vig.clear();
        for x in 0..w {
            let fx = (x as f32 + 0.5) * 0.25 - 0.5;
            let (x0, tx) = (fx.floor(), fx - fx.floor());
            let (xa, xb) = (
                (x0 as i32).clamp(0, bw as i32 - 1) as usize,
                (x0 as i32 + 1).clamp(0, bw as i32 - 1) as usize,
            );
            let ux = (x as f32 + 0.5) * iw - 1.0;
            self.cols.push((xa as u32, xb as u32, tx));
            self.vig.extend([ux * ux * 0.8; 3]);
        }
        for (r, d) in self.dith.iter_mut().enumerate() {
            d.clear();
            d.extend((0..w).flat_map(|x| [(BAYER[r * 4 + (x & 3)] + 0.5) * (1.0 / 16.0); 3]));
        }
    }

    /// Bloom, vignette, tone curve, dither and sRGB for the rows of `hdr`,
    /// which start at row `y0`, into the same rows of `out`. `rows` is
    /// scratch. After `columns` (and `spread`, when there is bloom).
    #[allow(clippy::too_many_arguments)]
    pub fn tone(
        &self,
        hdr: &[[f32; 3]],
        y0: usize,
        w: usize,
        h: usize,
        has_bloom: bool,
        out: &mut [u8],
        rows: &mut Vec<[f32; 3]>,
    ) {
        let (bw, bh) = self.dims[0];
        let b0 = &self.bl[0];
        let lut = &*self.lut;
        let ih = 2.0 / h as f32;
        let us = NLUT as f32 / UMAX;
        rows.resize(3 * w, [0.0; 3]);
        let (ra, rest) = rows.split_at_mut(w);
        let (rb, ru) = rest.split_at_mut(w);
        let mut held = (usize::MAX, usize::MAX);
        for (yl, src) in hdr.chunks(w).enumerate() {
            let y = y0 + yl;
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
                for (x, &(xa, xb, tx)) in self.cols.iter().enumerate() {
                    let (xa, xb) = (xa as usize, xb as usize);
                    for k in 0..3 {
                        ra[x][k] = b0[ya + xa][k] + (b0[ya + xb][k] - b0[ya + xa][k]) * tx;
                        rb[x][k] = b0[yb + xa][k] + (b0[yb + xb][k] - b0[yb + xa][k]) * tx;
                    }
                }
            }
            // Value by value along the row, each the same steps as the
            // others: the compiler does several at once.
            let vig = |v: f32| 1.0 - 0.20 * (v + uy2);
            let tone = |c: f32, v: f32| fmin(fmax(c * vig(v), 0.0).sqrt() * us, NLUT as f32);
            let src = src.as_flattened();
            let ru = ru.as_flattened_mut();
            if has_bloom {
                let (ra, rb) = (ra.as_flattened(), rb.as_flattened());
                let bloom = ra.iter().zip(rb).map(|(a, b)| a + (b - a) * ty);
                for (u, ((c, v), add)) in ru.iter_mut().zip(src.iter().zip(&self.vig).zip(bloom)) {
                    *u = tone(c + add, *v);
                }
            } else {
                for (u, (c, v)) in ru.iter_mut().zip(src.iter().zip(&self.vig)) {
                    *u = tone(*c, *v);
                }
            }
            let orow = &mut out[yl * w * 3..(yl + 1) * w * 3];
            for (o, (&u, d)) in orow.iter_mut().zip(ru.iter().zip(&self.dith[y & 3])) {
                // `u` is at most NLUT.
                let i = u as usize;
                let [at, step] = lut[i.min(NLUT)];
                let v = at + step * (u - i as f32);
                *o = (v + d) as u8;
            }
        }
    }
}

/// The first part of the finish for one band of rows, from row `y0` and one
/// row of tiles (`1 << sh` rows) high: surfaces' pixels exposed, and what is
/// bright enough to bloom gathered into `b0`, the band's rows of the
/// quarter-size picture, `bw` across. `open` and `hdr` are the band's rows,
/// `marks` and `bright` its row of tiles. Says whether anything was bright.
#[allow(clippy::too_many_arguments)]
pub fn gather(
    hdr: &mut [[f32; 3]],
    y0: usize,
    w: usize,
    exposure: f32,
    bloom: f32,
    open: &[u8],
    marks: &[u8],
    bright: &[u8],
    sh: usize,
    ns: usize,
    b0: &mut [[f32; 3]],
    bw: usize,
) -> bool {
    for (row, open) in hdr.chunks_mut(w).zip(open.chunks(w)) {
        for (tx, _) in marks.iter().enumerate().filter(|m| *m.1 != 0) {
            for x in (tx << sh)..((tx + 1) << sh).min(w) {
                if open[x] as usize != ns {
                    row[x] = expose(row[x], exposure);
                }
            }
        }
    }
    let mut has_bloom = false;
    if clamp01(bloom) > 0.0 {
        b0.fill([0.0; 3]);
        for (yl, row) in hdr.chunks(w).enumerate() {
            let by = (y0 + yl) / 4 - y0 / 4;
            let brow = &mut b0[by * bw..(by + 1) * bw];
            for (tx, _) in bright.iter().enumerate().filter(|m| *m.1 != 0) {
                for x in (tx << sh)..((tx + 1) << sh).min(w) {
                    let c = row[x];
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
    has_bloom
}

/// Rows of a level from the level below it, from row `y0`: `dst` holds
/// those rows of a level `dw` across.
fn down(
    src: &[[f32; 3]],
    (sw, sh): (usize, usize),
    dst: &mut [[f32; 3]],
    (dw, _dh): (usize, usize),
    y0: usize,
) {
    const K: [f32; 4] = [0.125, 0.375, 0.375, 0.125];
    for (yl, row) in dst.chunks_mut(dw).enumerate() {
        let y = y0 + yl;
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
            row[x] = o;
        }
    }
}
