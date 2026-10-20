//! The floor: the whole plane y = 0, worked out for each pixel, and the
//! mirror image of what stands on it.
//!
//! The scene is drawn a second time upside down at half size, with the same
//! four samples to a pixel as the picture itself, shaded once per surface,
//! and kept as a pyramid of smoothly blurred copies. A floor pixel reads the
//! copy whose blur suits the height of what it mirrors, and fades it with
//! that height. All of it is done only in the tiles that hold an image.
//!
//! Everything about the floor that changes slowly (the lamp's pool, the
//! sheen of the lamps, how mirror-like the floor is at that angle, the
//! backdrop) is worked out at every eighth pixel and blended between. The
//! shadow and the mirror image are read at every pixel.
//!
//! Each step works a band of rows at a time, and a band writes only its own
//! rows, so the bands may be shared out over threads.

use super::light::{Light, log2ish};
use super::par;
use super::raster::{
    Frag, KeyBand, NS, Pass, Setup, Tiles, Tri, dilate, draw_band, key_bands, resolve_band,
};
use super::shade::{Ctx, lamp_spec, shade, sky, surf};
use super::{clamp01, compress, expose, fmax, fmin};
use crate::studio::math::V3;
use crate::studio::scene::Floor;

const G: usize = 8;
const MLEV: usize = 6;
/// Values on the floor's coarse grid: lit floor colour, its reflection,
/// how strongly it mirrors, and the backdrop.
pub const CH: usize = 10;
/// Rows of the coarse grid worked out together.
const GRID_ROWS: usize = 8;

pub struct Mirror {
    pub w: usize,
    pub h: usize,
    /// Laid out by band, as `raster::KeyBand` says.
    pub keys: Vec<u64>,
    pub tris: Vec<Tri>,
    /// Tiles of sixteen pixels: one texel of level 4. A band of the image
    /// is one row of them.
    pub tiles: Tiles,
    /// The triangles of the upside-down view, by band.
    pub set: Setup,
    /// Whether the frame before left an image here.
    dirty: bool,
    /// The surfaces of each band.
    frags: Vec<Vec<Frag>>,
    open: Vec<u8>,
    /// For each of the first five levels, the tiles that may hold something
    /// now, and those that did last frame and must be emptied.
    nz: Vec<Vec<u8>>,
    pz: Vec<Vec<u8>>,
    scratch: Vec<u8>,
    /// Colour, cover, cover times height; level 0 is w by h.
    lv: Vec<Vec<[f32; 5]>>,
    dims: Vec<(usize, usize)>,
    /// Whether there is any image near each cell of level 4.
    near: Vec<u8>,
    grid: Vec<[f32; CH]>,
    row: Vec<[f32; CH]>,
}

impl Mirror {
    pub fn new(w: usize, h: usize) -> Mirror {
        let (mut lv, mut dims) = (Vec::new(), Vec::new());
        let (mut lw, mut lh) = (w, h);
        for _ in 0..MLEV {
            lv.push(vec![[0.0; 5]; lw * lh]);
            dims.push((lw, lh));
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
        }
        let tiles = Tiles::new(w, h, 4);
        let flags = vec![vec![0u8; tiles.f.len()]; 5];
        Mirror {
            w,
            h,
            keys: vec![0; w * h * NS],
            tris: Vec::new(),
            frags: (0..tiles.th).map(|_| Vec::new()).collect(),
            tiles,
            set: Setup::default(),
            dirty: false,
            open: vec![0; w * h],
            nz: flags.clone(),
            pz: flags,
            scratch: Vec::new(),
            near: vec![0; dims[4].0 * dims[4].1],
            lv,
            dims,
            grid: Vec::new(),
            row: Vec::new(),
        }
    }

    /// Shades the upside-down picture already drawn into `keys` and `tiles`
    /// (a band at a time, by `raster::draw_band`) and builds its blurred
    /// copies. `cx` looks from the eye's reflection
    /// under the floor; `cent` says where in a pixel to shade a surface that
    /// owns only some of its samples.
    pub fn shade(&mut self, cx: &Ctx, cent: &[(f32, f32); 16]) {
        self.draw_and_shade(None, cx, cent, 1);
    }

    /// `shade`, on up to `threads` threads; with `pass`, each band is first
    /// drawn from the triangles `raster::setup` filed in `set`, with its
    /// tile marks cleared before.
    pub fn draw_and_shade(
        &mut self,
        pass: Option<&Pass>,
        cx: &Ctx,
        cent: &[(f32, f32); 16],
        threads: usize,
    ) {
        let (w, h) = (self.w, self.h);
        let (tw, th) = (self.tiles.tw, self.tiles.th);
        // Level 0: each pixel the sum of its surfaces by the share of the
        // pixel they cover, so edges come out soft.
        {
            let (tris, set, pz0) = (&self.tris, &self.set, &self.pz[0]);
            let rows = w << 4;
            let mut open = self.open.chunks_mut(rows);
            let mut frags = self.frags.iter_mut();
            let mut nz0 = self.nz[0].chunks_mut(tw);
            let mut l0 = self.lv[0].chunks_mut(rows);
            let bands = key_bands(&mut self.keys, w, h, &mut self.tiles).map(move |(k, band)| {
                let (open, frags) = (open.next().unwrap(), frags.next().unwrap());
                (
                    k,
                    band,
                    open,
                    frags,
                    nz0.next().unwrap(),
                    l0.next().unwrap(),
                )
            });
            par::each0_by_cost(
                threads,
                bands,
                |b| set.bins[b.0].len(),
                |(k, mut band, open, frags, nz0, l0): (usize, KeyBand, _, _, _, _)| {
                    if let Some(pass) = pass {
                        band.marks.fill(0);
                        draw_band(pass, 4, set, k, &mut band);
                    }
                    frags.clear();
                    resolve_band(&band, w, 4, tris, frags, open);
                    nz0.copy_from_slice(&*band.marks);
                    let pz0 = &pz0[k * tw..(k + 1) * tw];
                    for tx in 0..tw {
                        if nz0[tx] | pz0[tx] != 0 {
                            let (x0, x1) = (tx << 4, ((tx + 1) << 4).min(w));
                            for row in l0.chunks_mut(w) {
                                row[x0..x1].fill([0.0; 5]);
                            }
                        }
                    }
                    let base = band.y0 * w;
                    for f in frags.iter() {
                        let t = &tris[f.tri as usize];
                        let (x, y) = (f.px as usize % w, f.px as usize / w);
                        let (ox, oy) = cent[(f.bits >> 4) as usize & 15];
                        let sf = surf(cx, t, x as f32 + ox, y as f32 + oy);
                        let c = shade(cx, &sf, t, 0.85);
                        let share = (f.bits & 15) as f32 * (1.0 / NS as f32);
                        // Keep one glint from swamping the blur.
                        let k = share / (1.0 + 0.25 * fmax(c[0], fmax(c[1], c[2])));
                        let o = &mut l0[f.px as usize - base];
                        o[0] += c[0] * k;
                        o[1] += c[1] * k;
                        o[2] += c[2] * k;
                        o[3] += share;
                        o[4] += share * fmax(sf.p.y, 0.0);
                    }
                },
            );
        }
        // Nothing mirrored now and nothing left over: every copy is empty
        // already, and level 0 was left as it was.
        let any = self.tiles.f.iter().any(|&f| f != 0);
        if !any && !self.dirty {
            return;
        }
        self.dirty = any;
        const K: [f32; 4] = [0.125, 0.375, 0.375, 0.125];
        for l in 1..MLEV {
            if l < 5 {
                let (lo, hi) = self.nz.split_at_mut(l);
                dilate(&lo[l - 1], tw, th, 1, &mut self.scratch, &mut hi[0]);
            }
            let (lo, hi) = self.lv.split_at_mut(l);
            let (src, dst) = (&lo[l - 1], &mut hi[0]);
            let ((sw, sh), (dw, dh)) = (self.dims[l - 1], self.dims[l]);
            let ts = 16 >> l;
            let (cells_x, span) = if ts >= 1 { (tw, ts) } else { (1, dw.max(dh)) };
            let (nz, pz) = (&self.nz, &self.pz);
            // A row of cells at a time: a row reads the rows of the level
            // below it and a row either side, and writes only its own.
            par::each0(
                threads,
                dst.chunks_mut(span * dw).enumerate(),
                |(cy, rows)| {
                    for cx_ in 0..cells_x {
                        let k = cy * cells_x + cx_;
                        if ts >= 1 && nz[l][k] | pz[l][k] == 0 {
                            continue;
                        }
                        let (x0, y0) = (cx_ * span, cy * span);
                        for y in y0..(y0 + span).min(dh) {
                            for x in x0..(x0 + span).min(dw) {
                                let mut o = [0.0f32; 5];
                                for (j, kj) in K.iter().enumerate() {
                                    let yy =
                                        ((2 * y + j) as i32 - 1).clamp(0, sh as i32 - 1) as usize;
                                    for (i, ki) in K.iter().enumerate() {
                                        let xx = ((2 * x + i) as i32 - 1).clamp(0, sw as i32 - 1)
                                            as usize;
                                        let (s, kk) = (&src[yy * sw + xx], kj * ki);
                                        for c in 0..5 {
                                            o[c] += s[c] * kk;
                                        }
                                    }
                                }
                                rows[(y - y0) * dw + x] = o;
                            }
                        }
                    }
                },
            );
        }
        core::mem::swap(&mut self.nz, &mut self.pz);
        // Level 3 is read first; it reaches under 20 pixels past the image.
        dilate(&self.tiles.f, tw, th, 2, &mut self.scratch, &mut self.near);
    }

    #[inline(always)]
    fn at(&self, l: usize, x: f32, y: f32) -> [f32; 5] {
        let (w, h) = self.dims[l];
        let sc = 1.0 / (1 << l) as f32;
        let (x, y) = (x * sc - 0.5, y * sc - 0.5);
        let (xf, yf) = (x.floor(), y.floor());
        let (fx, fy) = (x - xf, y - yf);
        let (mx, my) = (w as i32 - 1, h as i32 - 1);
        let (i0, i1) = (
            (xf as i32).clamp(0, mx) as usize,
            (xf as i32 + 1).clamp(0, mx) as usize,
        );
        let (j0, j1) = (
            (yf as i32).clamp(0, my) as usize,
            (yf as i32 + 1).clamp(0, my) as usize,
        );
        let s = &self.lv[l];
        let (a, b, c, d) = (
            s[j0 * w + i0],
            s[j0 * w + i1],
            s[j1 * w + i0],
            s[j1 * w + i1],
        );
        let mut o = [0.0f32; 5];
        for k in 0..5 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            o[k] = top + (bot - top) * fy;
        }
        o
    }

    /// The image blurred over about `r` of its own pixels each way.
    #[inline(always)]
    fn blurred(&self, x: f32, y: f32, r: f32) -> [f32; 5] {
        let lam = log2ish(fmax(r, 1.0));
        let l = (lam as usize).min(MLEV - 2);
        let f = clamp01(lam - l as f32);
        let a = self.at(l, x, y);
        let b = self.at(l + 1, x, y);
        let mut o = [0.0f32; 5];
        for k in 0..5 {
            o[k] = a[k] + (b[k] - a[k]) * f;
        }
        o
    }

    pub fn bytes(&self) -> usize {
        self.lv.iter().map(|l| l.len() * 20).sum::<usize>()
            + self.keys.len() * 8
            + self.tris.capacity() * core::mem::size_of::<Tri>()
            + self.set.bytes()
            + self.frags.iter().map(|f| f.capacity() * 12).sum::<usize>()
            + self.open.len()
    }

    /// Fills every pixel of `hdr` with the floor and backdrop it shows.
    #[allow(clippy::too_many_arguments)]
    pub fn floor(
        &mut self,
        cx: &Ctx,
        fl: &Floor,
        w: usize,
        h: usize,
        ns: usize,
        open: &[u8],
        tiles: &Tiles,
        fpx: f32,
        exposure: f32,
        hdr: &mut [[f32; 3]],
        bright: &mut [u8],
    ) {
        let mut rows = [core::mem::take(&mut self.row)];
        self.floor_with(
            cx, fl, w, h, ns, open, tiles, fpx, exposure, hdr, bright, 1, &mut rows,
        );
        let [row] = rows;
        self.row = row;
    }

    /// `floor` on up to `threads` threads, a row of `tiles` at a time, each
    /// thread with one of `rows` for scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn floor_with(
        &mut self,
        cx: &Ctx,
        fl: &Floor,
        w: usize,
        h: usize,
        ns: usize,
        open: &[u8],
        tiles: &Tiles,
        fpx: f32,
        exposure: f32,
        hdr: &mut [[f32; 3]],
        bright: &mut [u8],
        threads: usize,
        rows: &mut [Vec<[f32; CH]>],
    ) {
        self.grid(cx, fl, w, h, threads);
        let me = &*self;
        let (n, sh, tw) = (w << tiles.sh, tiles.sh, tiles.tw);
        let bands = hdr
            .chunks_mut(n)
            .zip(open.chunks(n))
            .zip(tiles.f.chunks(tw).zip(bright.chunks_mut(tw)))
            .enumerate();
        // The dearest bands first: those in the lamp's pool, where the
        // shadow is read, and near the mirror image.
        let (gw, (nw, nh)) = (w / G + 2, me.dims[4]);
        let cost = |k: usize| {
            let (y0, y1) = (k << sh, ((k + 1) << sh).min(h));
            let lit: usize = (y0 / G..=(y1 - 1) / G + 1)
                .map(|gy| {
                    me.grid[gy * gw..(gy + 1) * gw]
                        .iter()
                        .filter(|g| g[0] + g[1] + g[2] > 0.0)
                        .count()
                })
                .sum();
            let near: usize = ((y0 >> 5)..=((y1 - 1) >> 5))
                .map(|r| {
                    me.near[r.min(nh - 1) * nw..][..nw]
                        .iter()
                        .filter(|&&f| f != 0)
                        .count()
                })
                .sum();
            lit + 8 * near
        };
        par::each_by_cost(
            threads,
            bands,
            rows,
            |b| cost(b.0),
            |row, (k, ((hdr, open), (marks, bright)))| {
                me.floor_rows(
                    cx,
                    fl,
                    w,
                    h,
                    ns,
                    fpx,
                    exposure,
                    k << sh,
                    open,
                    marks,
                    sh,
                    hdr,
                    bright,
                    row,
                );
            },
        );
    }

    /// The slow parts of the floor, on a coarse grid, on up to `threads`
    /// threads: what `floor_rows` blends between.
    pub fn grid(&mut self, cx: &Ctx, fl: &Floor, w: usize, h: usize, threads: usize) {
        let (gw, gh) = (w / G + 2, h / G + 2);
        let rough = fmin(fmax(fl.rough, 0.045), 1.0);
        let a = rough * rough;
        let mirror = clamp01(fl.mirror);
        let eye = cx.eye;
        self.grid.resize(gw * gh, [0.0; CH]);
        let rows = self.grid.chunks_mut(gw * GRID_ROWS).enumerate();
        par::each0(threads, rows, |(k, cells)| {
            for (i, cell) in cells.iter_mut().enumerate() {
                let (gx, gy) = (i % gw, k * GRID_ROWS + i / gw);
                let d = cx.ray0 + cx.rdx * (gx * G) as f32 + cx.rdy * (gy * G) as f32;
                let il = 1.0 / d.dot(d).sqrt();
                let mut n = [0.0f32; CH];
                // Backdrop: the far wall, dim.
                let s = 0.5 * fmin(sky(fmax(d.y * il, 0.0), 0.0), 0.9);
                for c in 0..3 {
                    n[7 + c] = cx.amb[c] * s;
                }
                if eye.y > 0.0 {
                    let dy = fmin(d.y, -1e-4);
                    let x = eye
                        + V3 {
                            x: d.x,
                            y: dy,
                            z: d.z,
                        } * (eye.y / -dy);
                    let v = V3 {
                        x: -d.x,
                        y: -dy,
                        z: -d.z,
                    } * il;
                    let nv = fmax(v.y, 1e-3);
                    let r = V3 {
                        x: -v.x,
                        y: v.y,
                        z: -v.z,
                    };
                    let p5 = (1.0 - nv) * (1.0 - nv) * (1.0 - nv) * (1.0 - nv) * (1.0 - nv);
                    let refl = mirror * (0.45 + 0.55 * p5);
                    let sk = sky(r.y, rough);
                    let mut er = [cx.amb[0] * sk, cx.amb[1] * sk, cx.amb[2] * sk];
                    {
                        let lamp = &cx.lamps[0];
                        let lv = lamp.pos - x;
                        let inv = 1.0 / lv.dot(lv).sqrt();
                        let (pu, pv, _) = cx.light.project(x);
                        let pool = Light::pool(pu, pv);
                        let sp = fmin(
                            lamp_spec(V3::Y, v, r, nv, lv, inv, lamp.size, a, a * a, &[1.0; 3])[0],
                            4.0,
                        ) * pool;
                        let k = fmax(lv.y * inv, 0.0) * pool;
                        for c in 0..3 {
                            er[c] += lamp.col[c] * sp;
                            n[c] = fl.base[c] * lamp.col[c] * k;
                        }
                    }
                    for c in 0..3 {
                        n[3 + c] = er[c] * refl;
                    }
                    n[6] = refl;
                }
                *cell = n;
            }
        });
    }

    /// The floor and backdrop of one band of rows, from row `y0`, one row
    /// of the picture's tiles (`1 << sh` rows) high: `open`, `hdr` are the
    /// band's rows, `marks` and `bright` its row of tiles. `row` is scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn floor_rows(
        &self,
        cx: &Ctx,
        fl: &Floor,
        w: usize,
        h: usize,
        ns: usize,
        fpx: f32,
        exposure: f32,
        y0: usize,
        open: &[u8],
        marks: &[u8],
        sh: usize,
        hdr: &mut [[f32; 3]],
        bright: &mut [u8],
        row: &mut Vec<[f32; CH]>,
    ) {
        let gw = w / G + 2;
        let rough = fmin(fmax(fl.rough, 0.045), 1.0);
        let eye = cx.eye;
        let (rx, ry, rw) = cx.light.rows();
        let plane = |r: &[f32; 4]| {
            let e = r[0] * eye.x + r[1] * eye.y + r[2] * eye.z + r[3];
            let m = V3 {
                x: r[0],
                y: r[1],
                z: r[2],
            };
            [
                e * cx.ray0.y - eye.y * m.dot(cx.ray0),
                e * cx.rdx.y - eye.y * m.dot(cx.rdx),
                e * cx.rdy.y - eye.y * m.dot(cx.rdy),
            ]
        };
        let (px_, py_, pw_) = (plane(rx), plane(ry), plane(rw));
        let hz = 1.0 / fmax((cx.rdx.y * cx.rdx.y + cx.rdy.y * cx.rdy.y).sqrt(), 1e-9);
        let (nw, nh) = self.dims[4];
        let inv_ns = 1.0 / ns as f32;
        let amb = [
            fl.base[0] * cx.amb[0],
            fl.base[1] * cx.amb[1],
            fl.base[2] * cx.amb[2],
        ];
        row.resize(gw, [0.0; CH]);
        const FX: [f32; G] = [
            0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375,
        ];
        let floor_on = eye.y > 0.0;
        for y in y0..(y0 + (1 << sh)).min(h) {
            let (gy, fy) = (y / G, ((y % G) as f32 + 0.5) / G as f32);
            // Value by value, so the compiler does several at once.
            let a = self.grid[gy * gw..(gy + 1) * gw].as_flattened();
            let b = self.grid[(gy + 1) * gw..(gy + 2) * gw].as_flattened();
            for (r, (a, b)) in row.as_flattened_mut().iter_mut().zip(a.iter().zip(b)) {
                *r = a + (b - a) * fy;
            }
            let yc = y as f32 + 0.5;
            let at = (y - y0) * w;
            let nrow = (y >> 5).min(nh - 1) * nw;
            let mut x0 = 0;
            while x0 < w {
                let x1 = (x0 + G).min(w);
                let tk = x0 >> sh;
                let meshes = marks[tk] != 0;
                let (a, b) = (row[x0 / G], row[x0 / G + 1]);
                let near = self.near[nrow + (x0 >> 5).min(nw - 1)] != 0;
                if x1 - x0 == G
                    && !near
                    && a[0] == 0.0
                    && a[1] == 0.0
                    && a[2] == 0.0
                    && b[0] == 0.0
                    && b[1] == 0.0
                    && b[2] == 0.0
                {
                    let mut wt = [1.0f32; G];
                    let mut whole = [true; G];
                    if meshes {
                        for j in 0..G {
                            let n0 = open[at + x0 + j] as usize;
                            wt[j] = n0 as f32 * inv_ns;
                            whole[j] = n0 == ns;
                        }
                    }
                    let mut res = [[0.0f32; 3]; G];
                    let mut top = [0.0f32; G];
                    for j in 0..G {
                        let xc = (x0 + j) as f32 + 0.5;
                        let dy = cx.ray0.y + cx.rdx.y * xc + cx.rdy.y * yc;
                        let cov = if floor_on {
                            clamp01(0.5 - dy * hz)
                        } else {
                            0.0
                        };
                        let mut col = [0.0f32; 3];
                        for c in 0..3 {
                            let sky = a[7 + c] + (b[7 + c] - a[7 + c]) * FX[j];
                            let fl = a[3 + c] + (b[3 + c] - a[3 + c]) * FX[j];
                            col[c] = sky * (1.0 - cov) + cov * (amb[c] + fl);
                        }
                        let st = compress(col, wt[j]);
                        let m = fmax(st[0], fmax(st[1], st[2]));
                        let k = exposure / (1.0 - fmin(m, 0.999));
                        let done = whole[j] && m > 0.0;
                        res[j] = if done {
                            [st[0] * k, st[1] * k, st[2] * k]
                        } else {
                            st
                        };
                        top[j] = if done { m * k } else { 0.0 };
                    }
                    hdr[at + x0..at + x1].copy_from_slice(&res);
                    if top.iter().any(|&t| t > 1.0) {
                        bright[tk] = 1;
                    }
                    x0 = x1;
                    continue;
                }
                for x in x0..x1 {
                    let i = at + x;
                    let n0 = if meshes { open[i] as usize } else { ns };
                    if n0 == 0 {
                        hdr[i] = [0.0; 3];
                        continue;
                    }
                    let xc = x as f32 + 0.5;
                    let fx = FX[x - x0];
                    let dy = cx.ray0.y + cx.rdx.y * xc + cx.rdy.y * yc;
                    let cov = if floor_on {
                        clamp01(0.5 - dy * hz)
                    } else {
                        0.0
                    };
                    let mut g = [0.0f32; CH];
                    if cov < 1.0 {
                        for k in 7..CH {
                            g[k] = a[k] + (b[k] - a[k]) * fx;
                        }
                    }
                    if cov > 0.0 {
                        for k in 0..7 {
                            g[k] = a[k] + (b[k] - a[k]) * fx;
                        }
                    }
                    let mut col = [g[7] * (1.0 - cov), g[8] * (1.0 - cov), g[9] * (1.0 - cov)];
                    if cov > 0.0 {
                        let mut lit = 0.0;
                        if g[0] + g[1] + g[2] > 1e-5 {
                            let iw = 1.0 / (pw_[0] + pw_[1] * xc + pw_[2] * yc);
                            let (s, t) = cx.light.texel(
                                (px_[0] + px_[1] * xc + px_[2] * yc) * iw,
                                (py_[0] + py_[1] * xc + py_[2] * yc) * iw,
                            );
                            lit = cx.light.floor_shadow(s, t, dy * iw);
                        }
                        let mut see = 1.0;
                        let mut img = [0.0f32; 3];
                        let (hx, hy) = (xc * 0.5, yc * 0.5);
                        if near {
                            let c = self.at(3, hx, hy);
                            if c[3] > 1e-3 {
                                let dyc = fmin(dy, -1e-4);
                                let d = cx.ray0 + cx.rdx * xc + cx.rdy * yc;
                                let tf = (eye.y / -dyc) * d.dot(d).sqrt();
                                let ht = c[4] / c[3];
                                let r = 0.2 * rough * ht * fpx / (tf + ht);
                                let m = self.blurred(hx, hy, r);
                                if m[3] > 1e-3 {
                                    let ht = m[4] / m[3];
                                    let k = ht * (0.25 + 1.2 * rough);
                                    let fade = 1.0 / (1.0 + k * k);
                                    see = 1.0 - m[3];
                                    let top = fmax(m[0], fmax(m[1], m[2])) / m[3];
                                    let k = fade / fmax(1.0 - 0.25 * top, 0.2);
                                    for c in 0..3 {
                                        img[c] = m[c] * k;
                                    }
                                }
                            }
                        }
                        for c in 0..3 {
                            col[c] += cov * (g[c] * lit + amb[c] + g[3 + c] * see + g[6] * img[c]);
                        }
                    }
                    let st = compress(col, n0 as f32 * inv_ns);
                    if n0 == ns {
                        let f = expose(st, exposure);
                        if fmax(f[0], fmax(f[1], f[2])) > 1.0 {
                            bright[tk] = 1;
                        }
                        hdr[i] = f;
                    } else {
                        hdr[i] = st;
                    }
                }
                x0 = x1;
            }
        }
    }
}
