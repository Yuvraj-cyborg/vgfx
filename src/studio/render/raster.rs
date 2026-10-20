//! Which triangle is in front at each sample.
//!
//! Triangles are walked a row of samples at a time. An edge shared by two
//! triangles is always evaluated from its upper end with the same arithmetic,
//! so both sides agree to the bit on where it crosses a row: no sample on a
//! shared edge is taken twice or missed.
//!
//! Depth is `q = 1 / w` (larger is nearer), which is a plane across the screen.

use super::{fmax, fmin, par};
use crate::studio::math::{M4, V3};
use crate::studio::scene::{Frame, Instance, Mesh};
use core::ops::Range;

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
    scan_rows(sink, w, (0, h), offs, v, pl);
}

/// The corners top to bottom, as `scan` takes them.
#[inline(always)]
fn order(v: &[[f32; 2]; 3]) -> ([f32; 2], [f32; 2], [f32; 2]) {
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
    (a, b, c)
}

/// `scan` for the rows from `rows.0` up to `rows.1` only: those rows get the
/// very samples, with the very values, that `scan` gives them. `rows` must
/// lie within the picture.
#[inline(always)]
pub fn scan_rows<S: Sink>(
    sink: &mut S,
    w: i32,
    rows: (i32, i32),
    offs: &[(f32, f32)],
    v: &[[f32; 2]; 3],
    pl: &[f32; 5],
) {
    if let Some(e) = Edges::of(v) {
        scan_edges(sink, w, rows, offs, &e, pl);
    }
}

/// A triangle made ready for `scan`: its corners top to bottom and the
/// slopes of its edges, worked out once however many bands it is drawn in.
/// The default covers no rows.
#[derive(Clone, Copy, Default)]
pub struct Edges {
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    s_ac: f32,
    s_ab: f32,
    s_bc: f32,
}

impl Edges {
    /// None for a triangle that covers no rows.
    #[inline(always)]
    pub fn of(v: &[[f32; 2]; 3]) -> Option<Edges> {
        let (a, b, c) = order(v);
        if !(c[1] > a[1]) {
            return None;
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
        Some(Edges {
            a,
            b,
            c,
            s_ac,
            s_ab,
            s_bc,
        })
    }
}

/// `scan_rows` for a triangle already made ready.
#[inline(always)]
pub fn scan_edges<S: Sink>(
    sink: &mut S,
    w: i32,
    rows: (i32, i32),
    offs: &[(f32, f32)],
    e: &Edges,
    pl: &[f32; 5],
) {
    let (a, b, c) = (e.a, e.b, e.c);
    let (s_ac, s_ab, s_bc) = (e.s_ac, e.s_ab, e.s_bc);
    for (s, &(ox, oy)) in offs.iter().enumerate() {
        let r0 = ceil_i(a[1] - oy).max(rows.0);
        let r1 = ceil_i(c[1] - oy).min(rows.1);
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

/// `normals` on up to `threads` threads, a run of instances to each; the
/// normals are the same. `spans` is scratch.
pub fn normals_with(
    meshes: &[Mesh],
    frame: &Frame,
    vbase: &mut Vec<u32>,
    vn: &mut Vec<V3>,
    threads: usize,
    spans: &mut Vec<Range<usize>>,
) {
    if threads <= 1 || par::SERIAL {
        return normals(meshes, frame, vbase, vn);
    }
    vbase.clear();
    let mut n = 0;
    for inst in &frame.instances {
        vbase.push(n as u32);
        n += meshes.get(inst.mesh as usize).map_or(0, |m| m.pos.len());
    }
    vn.resize(n, V3::ZERO);
    runs(meshes, frame, threads * 2, spans);
    let mut rest = &mut vn[..];
    let mut parts = Vec::with_capacity(spans.len());
    for run in spans.iter() {
        let end = vbase.get(run.end).map_or(n, |&e| e as usize);
        let (mine, more) = core::mem::take(&mut rest).split_at_mut(end - vbase[run.start] as usize);
        rest = more;
        parts.push((run.clone(), mine));
    }
    par::each0(threads, parts.into_iter(), |(run, out)| {
        let mut o = out.iter_mut();
        for inst in &frame.instances[run] {
            let Some(mesh) = meshes.get(inst.mesh as usize) else {
                continue;
            };
            let x = &inst.xf;
            let s2 = x.m[0] * x.m[0] + x.m[1] * x.m[1] + x.m[2] * x.m[2];
            let k = if s2 > 0.0 { 1.0 / s2.sqrt() } else { 0.0 };
            for i in 0..mesh.pos.len() {
                let n = mesh.nrm.get(i).copied().unwrap_or(V3::Y);
                *o.next().unwrap() = x.dir(n) * k;
            }
        }
    });
}

/// Draws every instance into `buf` and records the triangles that could be
/// seen. A sample's value is its depth and the triangle's place in `tris`
/// plus one. All in one go; `setup` and `draw_band` draw the same samples a
/// band of rows at a time.
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
    let mut set = Setup::default();
    let cut = setup(pass, meshes, frame, vbase, vn, hv, tris, &mut set, tiles.sh);
    // Every triangle, in order, from wherever it was filed.
    let mut each = vec![None; set.count as usize];
    for it in set.bins.iter().flatten() {
        each[it.id as usize] = Some(*it);
    }
    let plane = (pass.w * pass.h) as usize;
    for it in each.iter().flatten() {
        let mut sink = KeySink {
            buf: &mut *buf,
            plane,
            w: pass.w as usize,
            h: pass.h as usize,
            ns: offs.len(),
            id: it.id as u64 + 1,
            tiles: &mut *tiles,
        };
        scan_edges(&mut sink, pass.w, (0, pass.h), offs, &it.e, &it.pl);
    }
    cut
}

/// A triangle filed under a band: its place in the frame's list, its edges
/// and its depth plane (x0, y0, q0, dq/dx, dq/dy), all a band needs to draw
/// it, side by side in the order it is drawn.
#[derive(Clone, Copy, Default)]
pub struct Item {
    pub id: u32,
    pub e: Edges,
    pub pl: [f32; 5],
}

/// The triangles of a view, ready to be drawn a band of rows at a time.
#[derive(Default)]
pub struct Setup {
    /// How many triangles were added.
    pub count: u32,
    /// For each band of rows, the triangles whose samples may fall in it.
    pub bins: Vec<Vec<Item>>,
}

impl Setup {
    /// Bytes held.
    pub fn bytes(&self) -> usize {
        let item = core::mem::size_of::<Item>();
        self.bins.iter().map(|b| b.capacity() * item).sum()
    }

    /// Empties it for a picture of `bands` bands.
    pub fn reset(&mut self, bands: usize) {
        self.count = 0;
        self.bins.resize_with(bands, Vec::new);
        for b in &mut self.bins {
            b.clear();
        }
    }

    /// Adds the next triangle, with corners `v` and depth plane `pl`, and
    /// files it under every band of `1 << sh` rows, of a picture `h` rows
    /// high, that a sample anywhere in a pixel inside it can lie in.
    pub fn add(&mut self, v: [[f32; 2]; 3], pl: [f32; 5], h: i32, sh: usize) {
        let id = self.count;
        self.count += 1;
        let Some(e) = Edges::of(&v) else {
            return;
        };
        // `scan_rows` takes the rows from ceil(a - oy) up to ceil(c - oy),
        // with oy from 0 to 1.
        let r0 = ceil_i(e.a[1] - 1.0).max(0);
        let r1 = ceil_i(e.c[1]).min(h);
        if r0 >= r1 {
            return;
        }
        for b in &mut self.bins[(r0 as usize >> sh)..=((r1 - 1) as usize >> sh)] {
            b.push(Item { id, e, pl });
        }
    }
}

/// Sets up every instance's triangles for one view: what each is, where it
/// lies on the screen, and which bands of `1 << sh` rows it may touch.
/// Returns how many were cut by the near plane.
#[allow(clippy::too_many_arguments)]
pub fn setup(
    pass: &Pass,
    meshes: &[Mesh],
    frame: &Frame,
    vbase: &[u32],
    vn: &[V3],
    hv: &mut Vec<[f32; 4]>,
    tris: &mut Vec<Tri>,
    set: &mut Setup,
    sh: usize,
) -> usize {
    tris.clear();
    set.reset((pass.h as usize).div_ceil(1 << sh));
    let all = 0..frame.instances.len();
    setup_run(pass, meshes, frame, vbase, vn, all, hv, tris, set, sh)
}

/// A run of instances set up on its own: its triangles, with their places
/// counted from the start of the run.
pub struct Part<T> {
    pub items: Vec<T>,
    pub set: Setup,
    pub hv: Vec<[f32; 4]>,
    pub cut: usize,
}

impl<T> Default for Part<T> {
    fn default() -> Self {
        Part {
            items: Vec::new(),
            set: Setup::default(),
            hv: Vec::new(),
            cut: 0,
        }
    }
}

/// The instances cut into at most `n` runs, in order, of about the same
/// amount of work.
pub fn runs(meshes: &[Mesh], frame: &Frame, n: usize, out: &mut Vec<Range<usize>>) {
    out.clear();
    let work = |i: &Instance| {
        meshes
            .get(i.mesh as usize)
            .map_or(1, |m| m.idx.len() / 3 + m.pos.len() / 4 + 1)
    };
    let total: usize = frame.instances.iter().map(work).sum();
    let each = total.div_ceil(n.max(1)).max(1);
    let (mut start, mut acc) = (0, 0);
    for (i, inst) in frame.instances.iter().enumerate() {
        acc += work(inst);
        if acc >= each {
            out.push(start..i + 1);
            (start, acc) = (i + 1, 0);
        }
    }
    if start < frame.instances.len() {
        out.push(start..frame.instances.len());
    }
}

/// Joins the parts' triangles, in order, into `all` and `set` (of `bands`
/// bands): what setting them all up in one run would have made.
pub fn join<T: Copy + Send + Sync>(
    parts: &[Part<T>],
    all: &mut Vec<T>,
    set: &mut Setup,
    bands: usize,
    threads: usize,
) {
    all.clear();
    for p in parts {
        all.extend_from_slice(&p.items);
    }
    set.count = parts.iter().map(|p| p.set.count).sum();
    set.bins.resize_with(bands, Vec::new);
    par::each0(threads, set.bins.iter_mut().enumerate(), |(k, b)| {
        b.clear();
        let mut at = 0;
        for p in parts {
            b.extend(p.set.bins[k].iter().map(|&it| Item {
                id: it.id + at,
                ..it
            }));
            at += p.set.count;
        }
    });
}

/// `setup` on up to `threads` threads, a run of instances to each, with the
/// same result. `parts` and `spans` are scratch.
#[allow(clippy::too_many_arguments)]
pub fn setup_with(
    pass: &Pass,
    meshes: &[Mesh],
    frame: &Frame,
    vbase: &[u32],
    vn: &[V3],
    hv: &mut Vec<[f32; 4]>,
    tris: &mut Vec<Tri>,
    set: &mut Setup,
    sh: usize,
    threads: usize,
    parts: &mut Vec<Part<Tri>>,
    spans: &mut Vec<Range<usize>>,
) -> usize {
    if threads <= 1 || par::SERIAL {
        return setup(pass, meshes, frame, vbase, vn, hv, tris, set, sh);
    }
    runs(meshes, frame, threads * 2, spans);
    if parts.len() < spans.len() {
        parts.resize_with(spans.len(), Part::default);
    }
    let parts = &mut parts[..spans.len()];
    let bands = (pass.h as usize).div_ceil(1 << sh);
    par::each0(
        threads,
        parts.iter_mut().zip(spans.iter().cloned()),
        |(p, run)| {
            p.items.clear();
            p.set.reset(bands);
            p.cut = setup_run(
                pass,
                meshes,
                frame,
                vbase,
                vn,
                run,
                &mut p.hv,
                &mut p.items,
                &mut p.set,
                sh,
            );
        },
    );
    join(parts, tris, set, bands, threads);
    parts.iter().map(|p| p.cut).sum()
}

/// Sets up the instances of `run`, adding their triangles to `tris` and
/// `set`.
#[allow(clippy::too_many_arguments)]
fn setup_run(
    pass: &Pass,
    meshes: &[Mesh],
    frame: &Frame,
    vbase: &[u32],
    vn: &[V3],
    run: Range<usize>,
    hv: &mut Vec<[f32; 4]>,
    tris: &mut Vec<Tri>,
    set: &mut Setup,
    sh: usize,
) -> usize {
    let mut cut = 0;
    let (wf, hf) = (pass.w as f32, pass.h as f32);
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
                let corners = [[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]];
                set.add(corners, [p0[0], p0[1], p0[2], dq.0, dq.1], pass.h, sh);
            }
        }
    }
    cut
}

/// One band of `1 << sh` rows of a view, from row `y0`: its keys, a row of
/// each sample in turn for each row of pixels (row `y`, sample `s` at
/// `(y - y0) * NS + s` rows in), and its row of tile marks.
pub struct KeyBand<'a> {
    pub y0: usize,
    pub keys: &'a mut [u64],
    pub marks: &'a mut [u8],
}

struct BandSink<'a> {
    keys: &'a mut [u64],
    marks: &'a mut [u8],
    y0: usize,
    w: usize,
    sh: usize,
    id: u64,
}

impl Sink for BandSink<'_> {
    #[inline(always)]
    fn span(&mut self, s: usize, r: usize, c0: usize, n: usize, mut q: f32, dq: f32) {
        let (w, sh) = (self.w, self.sh);
        for tc in (c0 >> sh)..=((c0 + n - 1) >> sh) {
            if self.marks[tc] == 0 {
                self.marks[tc] = 1;
                let (x0, x1) = (tc << sh, ((tc + 1) << sh).min(w));
                for row in self.keys.chunks_mut(w) {
                    row[x0..x1].fill(0);
                }
            }
        }
        let at = ((r - self.y0) * NS + s) * w + c0;
        for k in &mut self.keys[at..at + n] {
            let key = ((q.to_bits() as u64) << 32) | self.id;
            if key > *k {
                *k = key;
            }
            q += dq;
        }
    }
}

/// Draws the triangles filed under band `k` into it, four samples to a
/// pixel at `OFFS`: its samples come out as `draw` makes them. A tile's
/// samples are cleared the first time anything lands in it.
pub fn draw_band(pass: &Pass, sh: usize, set: &Setup, k: usize, band: &mut KeyBand) {
    let rows = (
        band.y0 as i32,
        (band.y0 + (1 << sh)).min(pass.h as usize) as i32,
    );
    let mut sink = BandSink {
        keys: &mut *band.keys,
        marks: &mut *band.marks,
        y0: band.y0,
        w: pass.w as usize,
        sh,
        id: 0,
    };
    for it in &set.bins[k] {
        sink.id = it.id as u64 + 1;
        scan_edges(&mut sink, pass.w, rows, &OFFS, &it.e, &it.pl);
    }
}

/// `resolve` for one band of `1 << sh` rows: appends the band's surfaces to
/// `frags`, in the order `resolve` finds them, and fills its rows of `open`.
pub fn resolve_band(
    band: &KeyBand,
    w: usize,
    sh: usize,
    tris: &[Tri],
    frags: &mut Vec<Frag>,
    open: &mut [u8],
) {
    for ((y, row), k) in open.chunks_mut(w).enumerate().zip(band.keys.chunks(NS * w)) {
        for (tx, _) in band.marks.iter().enumerate().filter(|m| *m.1 != 0) {
            for x in tx << sh..((tx + 1) << sh).min(w) {
                let px = ((band.y0 + y) * w + x) as u32;
                let ids = [
                    k[x] as u32,
                    k[w + x] as u32,
                    k[2 * w + x] as u32,
                    k[3 * w + x] as u32,
                ];
                if ids[0] == ids[1] && ids[0] == ids[2] && ids[0] == ids[3] {
                    if ids[0] == 0 {
                        row[x] = NS as u8;
                    } else {
                        row[x] = 0;
                        frags.push(Frag {
                            px,
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
                        px,
                        tri: ids[s] - 1,
                        bits: all.count_ones() | own << 4,
                    });
                }
                row[x] = free;
            }
        }
    }
}

/// A view's keys (`NS * w` to a row of pixels, laid out as `KeyBand` says;
/// `draw` and `resolve` take a plane a sample instead) and its tile marks,
/// cut into one `KeyBand` per row of tiles.
pub fn key_bands<'a>(
    keys: &'a mut [u64],
    w: usize,
    h: usize,
    tiles: &'a mut Tiles,
) -> impl Iterator<Item = (usize, KeyBand<'a>)> + Send + 'a {
    let (sh, tw) = (tiles.sh, tiles.tw);
    let bands = keys[..w * h * NS].chunks_mut((NS * w) << sh);
    let marks = tiles.f.chunks_mut(tw);
    bands.zip(marks).enumerate().map(move |(k, (keys, marks))| {
        (
            k,
            KeyBand {
                y0: k << sh,
                keys,
                marks,
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::studio::math::{M4, v3};
    use crate::studio::scene::Instance;
    use crate::studio::shapes::{ball, bbox};

    #[test]
    fn bands_draw_and_gather_what_one_pass_does() {
        let meshes = vec![
            bbox(v3(0.0, 0.5, 0.0), v3(0.5, 0.5, 0.5), 0.06),
            ball(v3(0.0, 0.4, 0.0), 0.4, 1.0, 20),
        ];
        // A row of parts, the last one through the near plane.
        let frame = Frame {
            instances: (0..13)
                .map(|i| {
                    let z = if i == 12 { 5.8 } else { (i % 4) as f32 * -0.5 };
                    let p = v3(i as f32 * 0.37 - 2.2, (i % 3) as f32 * 0.4, z);
                    Instance {
                        mesh: i % 2,
                        material: 0,
                        xf: M4::translate(p),
                        glow: 1.0,
                    }
                })
                .collect(),
            ..Frame::default()
        };
        let (mut vbase, mut vn) = (Vec::new(), Vec::new());
        normals(&meshes, &frame, &mut vbase, &mut vn);
        for (w, h) in [(157usize, 99usize), (40, 20)] {
            // From (0, 1, 6) toward -z.
            let (f, cx, cy) = (w as f32 * 0.6, w as f32 * 0.5, h as f32 * 0.5);
            let rw = [0.0, 0.0, -1.0, 6.0];
            let pass = Pass {
                m: [[f, 0.0, -cx, 6.0 * cx], [0.0, -f, -cy, f + 6.0 * cy], rw],
                near: 0.1,
                w: w as i32,
                h: h as i32,
                front: -1.0,
            };
            let mut hv = Vec::new();
            let (mut tris, mut keys, mut tiles) =
                (Vec::new(), vec![0; w * h * NS], Tiles::new(w, h, 5));
            let (mut frags, mut open) = (Vec::new(), vec![0; w * h]);
            let cut = draw(
                &pass, &OFFS, &meshes, &frame, &vbase, &vn, &mut hv, &mut tris, &mut keys,
                &mut tiles,
            );
            resolve(&keys, w, h, &tiles, &tris, &mut frags, &mut open);
            assert!(cut > 0 && frags.len() > 100);

            let (mut tris2, mut keys2, mut tiles2) =
                (Vec::new(), vec![0; w * h * NS], Tiles::new(w, h, 5));
            let (mut frags2, mut open2, mut set) = (Vec::new(), vec![0; w * h], Setup::default());
            let cut2 = setup(
                &pass, &meshes, &frame, &vbase, &vn, &mut hv, &mut tris2, &mut set, 5,
            );
            let mut rows = open2.chunks_mut(w << 5);
            for (k, mut band) in key_bands(&mut keys2, w, h, &mut tiles2) {
                band.marks.fill(0);
                draw_band(&pass, 5, &set, k, &mut band);
                resolve_band(&band, w, 5, &tris2, &mut frags2, rows.next().unwrap());
            }
            assert_eq!(cut, cut2);
            // The bands hold each row's samples in turn; `draw` a plane each.
            let planes: Vec<u64> = (0..NS)
                .flat_map(|s| (0..h).flat_map(move |y| (0..w).map(move |x| (s, y, x))))
                .map(|(s, y, x)| keys2[(y * NS + s) * w + x])
                .collect();
            assert!(keys == planes && tiles.f == tiles2.f && open == open2);
            let list = |f: &[Frag]| f.iter().map(|f| (f.px, f.tri, f.bits)).collect::<Vec<_>>();
            assert_eq!(list(&frags), list(&frags2));
        }
    }
}
