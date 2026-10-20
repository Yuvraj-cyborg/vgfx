//! Forms the shape builders do not make from a profile or an outline: a
//! solid swept through sections, for the bodies of vehicles and products,
//! whose width, height and roundness all change from front to back; and a
//! smooth surface through a grid of points, for anything a formula draws.
//! Only +, -, *, /, sqrt and the polynomial sin and cos are used, so the
//! meshes are the same bits on every machine.

use super::math::{PI, V3, cos, sin, v3};
use super::scene::Mesh;

/// One section of a loft, across z: how far it reaches each side of x = 0,
/// its bottom and its top, and how round its corners are, from 0 (square) to
/// 1 (as round as its smaller half allows).
#[derive(Clone, Copy, Debug)]
pub struct Section {
    pub z: f32,
    pub w: f32,
    pub y0: f32,
    pub y1: f32,
    pub round: f32,
}

/// The points of a section and their normals across it: a rectangle with
/// quarter circles for corners, `arc` points to each, counter-clockwise from
/// the middle of the right side.
fn ring(s: &Section, arc: usize) -> Vec<(f32, f32, f32, f32)> {
    let hw = s.w.max(1e-4);
    let hh = ((s.y1 - s.y0) * 0.5).max(1e-4);
    let mid = (s.y0 + s.y1) * 0.5;
    let r = hw.min(hh) * s.round.clamp(0.04, 1.0);
    let corners = [
        (hw - r, hh - r),
        (-(hw - r), hh - r),
        (-(hw - r), -(hh - r)),
        (hw - r, -(hh - r)),
    ];
    let mut out = Vec::with_capacity(4 * (arc + 1));
    for (q, (cx, cy)) in corners.iter().enumerate() {
        for k in 0..=arc {
            let a = (q as f32 + k as f32 / arc as f32) * PI * 0.5;
            let (c, sn) = (cos(a), sin(a));
            out.push((cx + r * c, mid + cy + r * sn, c, sn));
        }
    }
    out
}

/// A solid swept along z through `sections`, given in order of z, with
/// `around` points round each (a multiple of four is best). The ends are
/// closed flat. Neighbouring sections shade as one surface, so a body reads
/// as smooth; a section repeated at the same z makes a step.
pub fn loft(sections: &[Section], around: usize) -> Mesh {
    let mut m = Mesh::default();
    let n = sections.len();
    if n < 2 {
        return m;
    }
    let arc = (around / 4).clamp(1, 24);
    let rings: Vec<_> = sections.iter().map(|s| ring(s, arc)).collect();
    let per = rings[0].len();
    // Positions and normals, ring by ring. A normal leans along z by how
    // the surface narrows or widens there.
    for i in 0..n {
        let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
        let dz = sections[b].z - sections[a].z;
        for k in 0..per {
            let (x, y, nx, ny) = rings[i][k];
            let (dx, dy) = if dz.abs() > 1e-6 {
                (
                    (rings[b][k].0 - rings[a][k].0) / dz,
                    (rings[b][k].1 - rings[a][k].1) / dz,
                )
            } else {
                (0.0, 0.0)
            };
            m.pos.push(v3(x, y, sections[i].z));
            m.nrm.push(v3(nx, ny, -(nx * dx + ny * dy)).norm());
        }
    }
    let at = |i: usize, k: usize| (i * per + k % per) as u32;
    for i in 0..n - 1 {
        for k in 0..per {
            let q = [at(i, k), at(i, k + 1), at(i + 1, k + 1), at(i + 1, k)];
            for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
                let (p0, p1, p2) = (
                    m.pos[t[0] as usize],
                    m.pos[t[1] as usize],
                    m.pos[t[2] as usize],
                );
                let g = (p1 - p0).cross(p2 - p0);
                if g.len() < 1e-12 {
                    continue;
                }
                let nm = m.nrm[t[0] as usize] + m.nrm[t[1] as usize] + m.nrm[t[2] as usize];
                if g.dot(nm) >= 0.0 {
                    m.idx.extend_from_slice(&t);
                } else {
                    m.idx.extend_from_slice(&[t[0], t[2], t[1]]);
                }
            }
        }
    }
    // The two ends: a fan from the middle of each, facing out along z.
    for (i, nz) in [(0usize, -1.0f32), (n - 1, 1.0)] {
        let s = &sections[i];
        let base = m.pos.len() as u32;
        let nn = v3(0.0, 0.0, nz);
        m.pos.push(v3(0.0, (s.y0 + s.y1) * 0.5, s.z));
        m.nrm.push(nn);
        for &(x, y, _, _) in &rings[i] {
            m.pos.push(v3(x, y, s.z));
            m.nrm.push(nn);
        }
        for k in 0..per as u32 {
            let (b, c) = (base + 1 + k, base + 1 + (k + 1) % per as u32);
            let g = (m.pos[b as usize] - m.pos[base as usize])
                .cross(m.pos[c as usize] - m.pos[base as usize]);
            if g.len() < 1e-12 {
                continue;
            }
            if g.dot(nn) >= 0.0 {
                m.idx.extend_from_slice(&[base, b, c]);
            } else {
                m.idx.extend_from_slice(&[base, c, b]);
            }
        }
    }
    m
}

/// How a grid of points on a surface closes on itself, one way across it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seam {
    /// The first and the last row are edges of the surface, or poles: a row
    /// whose points all meet in one, as at the top of a sphere.
    Open,
    /// The row after the last is the first: round a sphere, a tube, a torus.
    Closed,
    /// The row after the last is the first turned over, its points the other
    /// way round, so that the surface comes back on its other side: a Möbius
    /// band. The other way across must be open.
    Twisted,
}

/// A grid of points and how it closes, read in the frame of one point.
struct Grid<'a> {
    pts: &'a [V3],
    n: [usize; 2],
    seams: [Seam; 2],
    /// For each way and each end (first row, last row): whether that row is
    /// a pole.
    pole: [[bool; 2]; 2],
}

impl Grid<'_> {
    fn p(&self, i: usize, j: usize) -> V3 {
        self.pts[i * self.n[1] + j]
    }

    /// The point `(i, j)` steps from the first, past either end as the seams
    /// carry it: the point it is, and whether the surface has come back on
    /// its other side to get there. Nothing past an open edge.
    fn at(&self, i: isize, j: isize) -> Option<(usize, usize, bool)> {
        let (nu, nv) = (self.n[0] as isize, self.n[1] as isize);
        let (mut i, mut j, mut flip) = (i, j, false);
        if !(0..nu).contains(&i) {
            match self.seams[0] {
                Seam::Open => return None,
                Seam::Closed => i = i.rem_euclid(nu),
                Seam::Twisted => {
                    i = i.rem_euclid(nu);
                    j = nv - 1 - j;
                    flip = true;
                }
            }
        }
        if !(0..nv).contains(&j) {
            match self.seams[1] {
                Seam::Open => return None,
                Seam::Closed => j = j.rem_euclid(nv),
                Seam::Twisted => {
                    j = j.rem_euclid(nv);
                    i = nu - 1 - i;
                    flip = !flip;
                }
            }
        }
        Some((i as usize, j as usize, flip))
    }

    /// Whether `(i, j)` is on a pole: a row whose points all meet.
    fn on_pole(&self, i: usize, j: usize) -> bool {
        (i == 0 && self.pole[0][0])
            || (i == self.n[0] - 1 && self.pole[0][1])
            || (j == 0 && self.pole[1][0])
            || (j == self.n[1] - 1 && self.pole[1][1])
    }

    /// The point that stands for `(i, j)`: the first of its row if that row
    /// is a pole, so that a pole is one point.
    fn canon(&self, i: usize, j: usize) -> usize {
        let last = [self.n[0] - 1, self.n[1] - 1];
        if (i == 0 && self.pole[0][0]) || (i == last[0] && self.pole[0][1]) {
            return i * self.n[1];
        }
        if (j == 0 && self.pole[1][0]) || (j == last[1] && self.pole[1][1]) {
            return j;
        }
        i * self.n[1] + j
    }

    /// The way the surface runs at `(i, j)` along `way`, from the points
    /// either side, or from the point itself at an open edge.
    fn tangent(&self, i: usize, j: usize, way: usize) -> V3 {
        let (i, j) = (i as isize, j as isize);
        let (a, b) = if way == 0 {
            (self.at(i - 1, j), self.at(i + 1, j))
        } else {
            (self.at(i, j - 1), self.at(i, j + 1))
        };
        let here = self.p(i as usize, j as usize);
        let of = |q: Option<(usize, usize, bool)>| q.map_or(here, |(i, j, _)| self.p(i, j));
        of(b) - of(a)
    }

    /// The points' quads, each `[a, b, c, d]` from `(i, j)` round to
    /// `(i, j + 1)`, as grid points and whether each is on the other side.
    fn cells(&self) -> Vec<[(usize, usize, bool); 4]> {
        let cells = |k: usize| self.n[k] - if self.seams[k] == Seam::Open { 1 } else { 0 };
        let mut out = Vec::with_capacity(cells(0) * cells(1));
        for i in 0..cells(0) as isize {
            for j in 0..cells(1) as isize {
                let q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
                if let [Some(a), Some(b), Some(c), Some(d)] = q.map(|(i, j)| self.at(i, j)) {
                    out.push([a, b, c, d]);
                }
            }
        }
        out
    }
}

/// A smooth surface through a grid of points: `pts[i * nv + j]` is the point
/// `i` steps along the first way across it (u) and `j` along the second (v),
/// `nu` by `nv` of them, and `seams` says how each way closes. Neighbouring
/// points shade as one surface: each point's normal is u × v, from the points
/// either side of it. A closed seam is welded; an open end whose points all
/// meet (the top of a sphere) is one point, its normal from the triangles
/// round it.
///
/// A surface that closes all round (a sphere, a torus) is a solid, turned to
/// face out. One with an edge, or a twist, is a sheet: it is made a solid
/// `thick` through, half each side of the points, its edges closed with a
/// rim; with no thickness it is the sheet alone, seen from the side u × v
/// points to, which the renderer shows from that side only.
///
/// An open way takes two points or more, a closed or twisted one three; a
/// twisted way needs the other open. A grid that does not fit gives an empty
/// mesh.
pub fn surface(pts: &[V3], nu: usize, nv: usize, seams: [Seam; 2], thick: f32) -> Mesh {
    let n = [nu, nv];
    for k in 0..2 {
        let least = if seams[k] == Seam::Open { 2 } else { 3 };
        if n[k] < least || (seams[k] == Seam::Twisted && seams[1 - k] != Seam::Open) {
            return Mesh::default();
        }
    }
    if nu.checked_mul(nv) != Some(pts.len())
        || !pts
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
    {
        return Mesh::default();
    }
    // How near points must be to be one: a hundred-thousandth of the
    // surface's size.
    let (mut lo, mut hi) = (pts[0], pts[0]);
    for p in pts {
        lo = v3(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = v3(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let size = (hi - lo).len();
    if !(size > 0.0 && size.is_finite()) {
        return Mesh::default();
    }
    let near = size * 1e-5;
    let mut g = Grid {
        pts,
        n,
        seams,
        pole: [[false; 2]; 2],
    };
    for k in 0..2 {
        if seams[k] != Seam::Open {
            continue;
        }
        for (e, row) in [0, n[k] - 1].into_iter().enumerate() {
            let point = |m: usize| if k == 0 { g.p(row, m) } else { g.p(m, row) };
            let first = point(0);
            g.pole[k][e] = (1..n[1 - k]).all(|m| (point(m) - first).len() <= near);
        }
    }
    let cells = g.cells();

    // Each point's normal, in its own frame: u × v from its neighbours; where
    // that fails (a pole, points that meet) the triangles round it, summed.
    let mut round = vec![V3::ZERO; nu * nv];
    let corners = |q: &[(usize, usize, bool); 4]| -> [[usize; 3]; 2] {
        let pos = q.map(|(i, j, _)| g.p(i, j));
        if (pos[2] - pos[0]).dot(pos[2] - pos[0]) <= (pos[3] - pos[1]).dot(pos[3] - pos[1]) {
            [[0, 1, 2], [0, 2, 3]]
        } else {
            [[0, 1, 3], [1, 2, 3]]
        }
    };
    for q in &cells {
        for t in corners(q) {
            let [a, b, c] = t.map(|k| g.p(q[k].0, q[k].1));
            let f = (b - a).cross(c - a);
            for k in t {
                let (i, j, flip) = q[k];
                round[g.canon(i, j)] += if flip { -f } else { f };
            }
        }
    }
    let mut nrm = vec![V3::ZERO; nu * nv];
    for i in 0..nu {
        for j in 0..nv {
            let c = g.canon(i, j);
            let (pu, pv) = (g.tangent(i, j, 0), g.tangent(i, j, 1));
            let x = pu.cross(pv);
            let k = x.dot(x);
            nrm[i * nv + j] = if !g.on_pole(i, j) && k > 0.0 && k > 1e-12 * pu.dot(pu) * pv.dot(pv)
            {
                x.norm()
            } else if round[c].dot(round[c]) > 0.0 {
                round[c].norm()
            } else {
                V3::Y
            };
        }
    }

    let solid = seams.iter().all(|&s| s != Seam::Twisted)
        && (0..2).all(|k| seams[k] == Seam::Closed || (g.pole[k][0] && g.pole[k][1]));
    let half = if solid { 0.0 } else { thick.max(0.0) * 0.5 };
    let mut m = Mesh::default();
    // A vertex for each point on each side, made when first used.
    let mut made = vec![u32::MAX; 2 * nu * nv];
    let mut vertex = |m: &mut Mesh, i: usize, j: usize, side: bool| -> u32 {
        let c = g.canon(i, j);
        let slot = &mut made[2 * c + side as usize];
        if *slot == u32::MAX {
            *slot = m.pos.len() as u32;
            let nn = if side { -nrm[c] } else { nrm[c] };
            m.pos.push(pts[c] + nn * half);
            m.nrm.push(nn);
        }
        *slot
    };
    let sides: &[bool] = if half > 0.0 { &[false, true] } else { &[false] };
    for &side in sides {
        for q in &cells {
            for t in corners(q) {
                let mut v = t.map(|k| vertex(&mut m, q[k].0, q[k].1, side != q[k].2));
                if v[0] == v[1] || v[1] == v[2] || v[2] == v[0] {
                    continue;
                }
                if side {
                    v.swap(1, 2);
                }
                m.idx.extend_from_slice(&v);
            }
        }
    }
    if solid {
        // Turned to face out: the volume it encloses comes out positive.
        let vol: f32 = m
            .idx
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| m.pos[t[0] as usize].dot(m.pos[t[1] as usize].cross(m.pos[t[2] as usize])))
            .sum();
        if vol < 0.0 {
            for t in m.idx.as_chunks_mut::<3>().0 {
                t.swap(1, 2);
            }
            for nn in &mut m.nrm {
                *nn = -*nn;
            }
        }
        return m;
    }
    if half > 0.0 {
        rims(&g, &nrm, half, &mut m);
    }
    m
}

/// The rims that close a thick sheet's edges: along each open end that is
/// not a pole, a strip from one face to the other, facing out across it.
fn rims(g: &Grid, nrm: &[V3], half: f32, m: &mut Mesh) {
    let n = g.n;
    for k in 0..2 {
        if g.seams[k] != Seam::Open {
            continue;
        }
        let o = 1 - k;
        let edges = n[o] - if g.seams[o] == Seam::Open { 1 } else { 0 };
        for e in 0..2 {
            if g.pole[k][e] {
                continue;
            }
            let row = if e == 0 { 0 } else { n[k] as isize - 1 };
            let mut ends = Vec::with_capacity(edges + 1);
            for s in 0..=edges as isize {
                let (i, j) = if k == 0 { (row, s) } else { (s, row) };
                let Some((i, j, flip)) = g.at(i, j) else {
                    continue;
                };
                // Past a twist the point is at the other end, in a frame of its own.
                let e = if flip { 1 - e } else { e };
                let c = g.canon(i, j);
                let nn = nrm[c];
                // Out across the edge: the edge's own way crossed with the
                // face's normal, turned away from the rest of the sheet; where
                // that fails, straight on along the way the sheet ends.
                let sign = if (k == 1) == (e == 0) { 1.0 } else { -1.0 };
                let mut out = g.tangent(i, j, o).cross(nn) * sign;
                if !(out.dot(out) > 0.0) {
                    let way = g.tangent(i, j, k);
                    out = (way - nn * way.dot(nn)) * if e == 0 { -1.0 } else { 1.0 };
                }
                let out = if out.dot(out) > 0.0 { out.norm() } else { nn };
                // Its two corners, on the face of the first point's side and on the other.
                let base = m.pos.len() as u32;
                let (front, back) = if flip { (-nn, nn) } else { (nn, -nn) };
                for f in [front, back] {
                    m.pos.push(g.pts[c] + f * half);
                    m.nrm.push(out);
                }
                ends.push(base);
            }
            for w in ends.windows(2) {
                let (fp, bp, fq, bq) = (w[0], w[0] + 1, w[1], w[1] + 1);
                // Wound to face out, as for the plane z = 0 seen from beyond its edge.
                let turned = (k == 1) == (e == 0);
                if turned {
                    m.idx.extend_from_slice(&[fp, bq, fq, fp, bp, bq]);
                } else {
                    m.idx.extend_from_slice(&[fp, fq, bq, fp, bq, bp]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sec(z: f32, w: f32, y0: f32, y1: f32, round: f32) -> Section {
        Section {
            z,
            w,
            y0,
            y1,
            round,
        }
    }

    #[test]
    fn a_loft_is_closed_and_faces_out() {
        let m = loft(
            &[
                sec(-1.0, 0.4, 0.2, 0.6, 0.5),
                sec(0.0, 0.6, 0.1, 0.9, 0.8),
                sec(1.2, 0.3, 0.2, 0.5, 1.0),
            ],
            24,
        );
        assert!(!m.idx.is_empty());
        // Every triangle faces away from the middle of the body.
        let mid = v3(0.0, 0.5, 0.0);
        for t in m.idx.as_chunks::<3>().0 {
            let (a, b, c) = (
                m.pos[t[0] as usize],
                m.pos[t[1] as usize],
                m.pos[t[2] as usize],
            );
            let g = (b - a).cross(c - a);
            let centre = (a + b + c) * (1.0 / 3.0);
            assert!(g.dot(centre - mid) > -1e-6, "a triangle faces in");
        }
        // It reaches as far as its widest section, and no further.
        let wide = m.pos.iter().map(|p| p.x.abs()).fold(0.0f32, f32::max);
        assert!((wide - 0.6).abs() < 1e-4);
    }

    #[test]
    fn a_loft_is_the_same_bits_every_time() {
        let s = [sec(0.0, 0.5, 0.0, 1.0, 0.3), sec(2.0, 0.5, 0.0, 1.0, 0.3)];
        let (a, b) = (loft(&s, 16), loft(&s, 16));
        assert_eq!(
            a.pos.iter().map(|p| p.x.to_bits()).collect::<Vec<_>>(),
            b.pos.iter().map(|p| p.x.to_bits()).collect::<Vec<_>>()
        );
    }

    use crate::studio::math::{PI, TAU};

    /// The points of `f` over a grid: a closed way steps short of its end,
    /// which is its start again; an open one reaches it.
    fn grid(
        n: [usize; 2],
        u: [f32; 2],
        v: [f32; 2],
        shut: [bool; 2],
        f: impl Fn(f32, f32) -> V3,
    ) -> Vec<V3> {
        let at = |k: usize, s: usize, r: [f32; 2]| {
            let steps = if shut[k] { n[k] } else { n[k] - 1 };
            r[0] + (r[1] - r[0]) * (s as f32 / steps as f32)
        };
        let mut out = Vec::new();
        for i in 0..n[0] {
            for j in 0..n[1] {
                out.push(f(at(0, i, u), at(1, j, v)));
            }
        }
        out
    }

    fn sphere(r: f32, u: f32, v: f32) -> V3 {
        v3(r * sin(v) * cos(u), r * cos(v), r * sin(v) * sin(u))
    }

    fn torus(u: f32, v: f32) -> V3 {
        let (big, small) = (1.0, 0.3);
        v3(
            (big + small * cos(v)) * cos(u),
            small * sin(v),
            (big + small * cos(v)) * sin(u),
        )
    }

    fn tri(m: &Mesh, t: &[u32]) -> (V3, V3, V3) {
        (
            m.pos[t[0] as usize],
            m.pos[t[1] as usize],
            m.pos[t[2] as usize],
        )
    }

    /// Every edge met by the same edge running the other way, by where its
    /// ends are: a solid with no cracks, nothing turned against its
    /// neighbours.
    fn closed(m: &Mesh) -> bool {
        use std::collections::HashMap;
        let key = |p: V3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
        let mut count = HashMap::new();
        for t in m.idx.as_chunks::<3>().0 {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *count
                    .entry((key(m.pos[a as usize]), key(m.pos[b as usize])))
                    .or_insert(0) += 1;
            }
        }
        !count.is_empty()
            && count
                .iter()
                .all(|(&(a, b), n)| count.get(&(b, a)) == Some(n))
    }

    /// Each triangle is wound the way its corners' normals point.
    fn wound_as_shaded(m: &Mesh) -> bool {
        m.idx.as_chunks::<3>().0.iter().all(|t| {
            let (a, b, c) = tri(m, t);
            let nn = m.nrm[t[0] as usize] + m.nrm[t[1] as usize] + m.nrm[t[2] as usize];
            (b - a).cross(c - a).dot(nn) > 0.0
        })
    }

    #[test]
    fn a_sphere_meets_at_its_poles_and_faces_out() {
        let (nu, nv) = (24, 13);
        let pts = grid([nu, nv], [0.0, TAU], [0.0, PI], [true, false], |u, v| {
            sphere(0.5, u, v)
        });
        let m = surface(&pts, nu, nv, [Seam::Closed, Seam::Open], 0.0);
        // Each pole is one point; the rows between are welded round the seam.
        assert_eq!(m.pos.len(), nu * (nv - 2) + 2);
        assert_eq!(m.idx.len() / 3, 2 * nu * (nv - 2));
        assert!(closed(&m) && wound_as_shaded(&m));
        for t in m.idx.as_chunks::<3>().0 {
            let (a, b, c) = tri(&m, t);
            assert!(
                (b - a).cross(c - a).dot(a + b + c) > 0.0,
                "a triangle faces in"
            );
        }
        // Normals run straight out, at the poles as everywhere else.
        for (p, n) in m.pos.iter().zip(&m.nrm) {
            assert!(n.dot(p.norm()) > 0.999, "{n:?} at {p:?}");
        }
        let top = m.nrm.iter().filter(|n| n.y > 0.9999).count();
        assert_eq!(top, 1, "one point at the top");
        // Drawn the other way round, it still faces out.
        let back = grid([nu, nv], [0.0, TAU], [PI, 0.0], [true, false], |u, v| {
            sphere(0.5, u, v)
        });
        let b = surface(&back, nu, nv, [Seam::Closed, Seam::Open], 0.0);
        for (p, n) in b.pos.iter().zip(&b.nrm) {
            assert!(n.dot(p.norm()) > 0.999);
        }
        assert!(closed(&b) && wound_as_shaded(&b));
    }

    #[test]
    fn a_torus_is_welded_round_both_ways() {
        let (nu, nv) = (32, 16);
        let pts = grid([nu, nv], [0.0, TAU], [0.0, TAU], [true, true], torus);
        let m = surface(&pts, nu, nv, [Seam::Closed, Seam::Closed], 0.05);
        // A solid has no rim and no thickness: one point a vertex.
        assert_eq!(m.pos.len(), nu * nv);
        assert_eq!(m.idx.len() / 3, 2 * nu * nv);
        assert!(closed(&m) && wound_as_shaded(&m));
        for (p, n) in m.pos.iter().zip(&m.nrm) {
            let ring = v3(p.x, 0.0, p.z).norm();
            assert!(n.dot((*p - ring).norm()) > 0.999, "{n:?} at {p:?}");
        }
        // No two vertices in one place: the seams are one, not two.
        let mut seen: Vec<_> = m
            .pos
            .iter()
            .map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), m.pos.len());
    }

    #[test]
    fn a_sheet_is_made_a_solid_as_thick_as_asked() {
        let (nu, nv) = (9, 7);
        let wave = |x: f32, z: f32| v3(x, 0.2 * sin(2.0 * x) * cos(3.0 * z), z);
        let pts = grid([nu, nv], [-1.0, 1.0], [-0.5, 0.5], [false, false], wave);
        let m = surface(&pts, nu, nv, [Seam::Open, Seam::Open], 0.04);
        // Two faces, and a rim of two corners at each point round the edge.
        let round = 2 * (nu + nv);
        assert_eq!(m.pos.len(), 2 * nu * nv + 2 * round);
        assert_eq!(
            m.idx.len() / 3,
            4 * (nu - 1) * (nv - 1) + 2 * 2 * (nu - 1 + nv - 1)
        );
        assert!(closed(&m) && wound_as_shaded(&m));
        // The faces stand 0.02 off the points, either side.
        for (k, p) in pts.iter().enumerate() {
            let near = m
                .pos
                .iter()
                .filter(|q| ((**q - *p).len() - 0.02).abs() < 1e-5)
                .count();
            assert!(near >= 2, "point {k} has {near} corners 0.02 from it");
        }
        // With no thickness it is the sheet alone.
        let thin = surface(&pts, nu, nv, [Seam::Open, Seam::Open], 0.0);
        assert_eq!(
            (thin.pos.len(), thin.idx.len() / 3),
            (nu * nv, 2 * (nu - 1) * (nv - 1))
        );
        assert!(wound_as_shaded(&thin));
    }

    #[test]
    fn a_twisted_band_comes_back_on_its_other_side_whole() {
        let (nu, nv) = (40, 5);
        let mobius = |u: f32, v: f32| {
            let r = 1.0 + v * cos(u * 0.5);
            v3(r * cos(u), v * sin(u * 0.5), r * sin(u))
        };
        let pts = grid([nu, nv], [0.0, TAU], [-0.3, 0.3], [true, false], mobius);
        let m = surface(&pts, nu, nv, [Seam::Twisted, Seam::Open], 0.03);
        assert!(closed(&m) && wound_as_shaded(&m));
        // Its one edge goes round twice: every point of the two edge rows
        // carries a rim.
        assert_eq!(m.pos.len(), 2 * nu * nv + 2 * 2 * (nu + 1));
        // Taken as merely closed, the seam would cross the band.
        let wrong = surface(&pts, nu, nv, [Seam::Closed, Seam::Open], 0.03);
        let long = |m: &Mesh| {
            m.idx
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| {
                    let (a, b, c) = tri(m, t);
                    (b - a).len().max((c - b).len()).max((a - c).len())
                })
                .fold(0.0f32, f32::max)
        };
        assert!(
            long(&m) < 0.35 && long(&wrong) > 0.5,
            "{} {}",
            long(&m),
            long(&wrong)
        );
    }

    #[test]
    fn a_cup_has_a_pole_at_the_bottom_and_a_rim_at_the_top() {
        let (nu, nv) = (24, 8);
        let cup = |u: f32, v: f32| v3(v * cos(u), v * v, v * sin(u));
        let pts = grid([nu, nv], [0.0, TAU], [0.0, 1.0], [true, false], cup);
        let m = surface(&pts, nu, nv, [Seam::Closed, Seam::Open], 0.02);
        assert!(closed(&m) && wound_as_shaded(&m));
        // The bottom is one point each side, its normal straight down the axis.
        let bottom: Vec<_> = m
            .pos
            .iter()
            .zip(&m.nrm)
            .filter(|(p, _)| p.x.abs() < 1e-6 && p.z.abs() < 1e-6)
            .collect();
        assert_eq!(bottom.len(), 2);
        assert!(bottom.iter().all(|(_, n)| n.y.abs() > 0.9999));
    }

    #[test]
    fn a_surface_is_the_same_bits_every_time() {
        let pts = grid([20, 10], [0.0, TAU], [0.0, PI], [true, false], |u, v| {
            sphere(1.0, u, v)
        });
        let (a, b) = (
            surface(&pts, 20, 10, [Seam::Closed, Seam::Open], 0.0),
            surface(&pts, 20, 10, [Seam::Closed, Seam::Open], 0.0),
        );
        let bits = |m: &Mesh| {
            m.pos
                .iter()
                .chain(&m.nrm)
                .flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(a.idx, b.idx);
    }

    #[test]
    fn a_grid_that_does_not_fit_makes_nothing() {
        let pts = grid([6, 4], [0.0, 1.0], [0.0, 1.0], [false, false], |u, v| {
            v3(u, 0.0, v)
        });
        let open = [Seam::Open, Seam::Open];
        assert!(!surface(&pts, 6, 4, open, 0.0).idx.is_empty());
        assert!(
            surface(&pts, 5, 4, open, 0.0).pos.is_empty(),
            "the wrong count"
        );
        assert!(
            surface(&pts[..4], 4, 1, open, 0.0).pos.is_empty(),
            "one row"
        );
        assert!(
            surface(&pts, 6, 4, [Seam::Twisted, Seam::Closed], 0.0)
                .pos
                .is_empty()
        );
        assert!(
            surface(&[V3::ZERO; 24], 6, 4, open, 0.0).pos.is_empty(),
            "all in one place"
        );
        let mut bad = pts.clone();
        bad[5].y = f32::NAN;
        assert!(
            surface(&bad, 6, 4, open, 0.0).pos.is_empty(),
            "not a number"
        );
    }
}
