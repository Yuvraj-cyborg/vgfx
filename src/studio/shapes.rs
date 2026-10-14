//! Shape builders. Every one returns a closed solid wound counter-clockwise
//! seen from outside. Only +, -, *, /, sqrt and the polynomial sin and cos are
//! used, so the meshes are the same bits on every machine.

use super::math::{M4, TAU, V3, cos, sin, v3};
use super::scene::Mesh;

/// A polygon (a fan from its first corner) with a normal at each corner,
/// wound so that it faces `hint`. Slivers with no area are dropped.
fn poly(m: &mut Mesh, pts: &[(V3, V3)], hint: V3) {
    let n = pts.len();
    let mut g = V3::ZERO;
    for i in 0..n {
        let (a, b) = (pts[i].0, pts[(i + 1) % n].0);
        g += v3(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    let rev = g.dot(hint) < 0.0;
    let base = m.pos.len() as u32;
    for (p, nn) in pts {
        m.pos.push(*p);
        m.nrm.push(nn.norm());
    }
    for i in 1..n - 1 {
        let (a, b, c) = (pts[0].0, pts[i].0, pts[i + 1].0);
        if (b - a).cross(c - a).len() < 1e-10 {
            continue;
        }
        let (i, j) = (base + i as u32, base + i as u32 + 1);
        if rev {
            m.idx.extend_from_slice(&[base, j, i]);
        } else {
            m.idx.extend_from_slice(&[base, i, j]);
        }
    }
}

/// A convex ring of corners in one plane, counter-clockwise seen from the
/// side `n` points to, filled from its middle so that every corner is used,
/// even those that lie along a side. Which way each triangle faces comes
/// from the order of the ring, never from its own tiny area, so a sliver
/// one rounding step thick still faces the way its neighbours do.
fn fan(m: &mut Mesh, ring: &[V3], n: V3) {
    let k = ring.len();
    if k < 3 {
        return;
    }
    let mut c = V3::ZERO;
    for p in ring {
        c += *p;
    }
    let c = c * (1.0 / k as f32);
    let base = m.pos.len() as u32;
    m.pos.push(c);
    m.nrm.push(n);
    for p in ring {
        m.pos.push(*p);
        m.nrm.push(n);
    }
    for i in 0..k {
        let j = (i + 1) % k;
        if ring[i] == ring[j] {
            continue;
        }
        m.idx
            .extend_from_slice(&[base, base + 1 + i as u32, base + 1 + j as u32]);
    }
}

/// Four corners, counter-clockwise seen from the side they face, each with
/// its normal; facing comes from the order, as in `fan`.
fn quad(m: &mut Mesh, q: [(V3, V3); 4]) {
    let base = m.pos.len() as u32;
    for (p, n) in q {
        m.pos.push(p);
        m.nrm.push(n.norm());
    }
    for (a, b, c) in [(0usize, 1usize, 2usize), (0, 2, 3)] {
        if q[a].0 != q[b].0 && q[b].0 != q[c].0 && q[c].0 != q[a].0 {
            m.idx
                .extend_from_slice(&[base + a as u32, base + b as u32, base + c as u32]);
        }
    }
}

/// Appends `src`, moved by `xf` (a turn and a move only), to `dst`. With
/// `mirror` the result is reflected in the plane x = 0 after `xf`.
pub fn add(dst: &mut Mesh, src: &Mesh, xf: &M4, mirror: bool) {
    let base = dst.pos.len() as u32;
    for (p, n) in src.pos.iter().zip(&src.nrm) {
        let (mut p, mut n) = (xf.point(*p), xf.dir(*n));
        if mirror {
            p.x = -p.x;
            n.x = -n.x;
        }
        dst.pos.push(p);
        dst.nrm.push(n);
    }
    for t in src.idx.as_chunks::<3>().0 {
        if mirror {
            dst.idx
                .extend_from_slice(&[base + t[0], base + t[2], base + t[1]]);
        } else {
            dst.idx
                .extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
        }
    }
}

/// A frame whose y axis runs from `a` toward `b`, with its origin at `a`.
pub fn frame_y(a: V3, b: V3) -> M4 {
    let y = (b - a).norm();
    let h = if y.x.abs() < 0.9 { V3::X } else { V3::Z };
    let x = (h - y * h.dot(y)).norm();
    let z = x.cross(y);
    M4 {
        m: [
            x.x, x.y, x.z, 0.0, y.x, y.y, y.z, 0.0, z.x, z.y, z.z, 0.0, a.x, a.y, a.z, 1.0,
        ],
    }
}

/// A box with every edge cut back by `b`. The cut faces carry the normals of
/// the faces they join, so an edge shades as if rounded and catches the lamp.
pub fn bbox(c: V3, h: V3, b: f32) -> Mesh {
    let b = b.min(h.x * 0.8).min(h.y * 0.8).min(h.z * 0.8);
    let mut m = Mesh::default();
    let hh = [h.x, h.y, h.z];
    let cv = |a: usize, sg: [f32; 3]| -> (V3, V3) {
        let mut p = [0.0f32; 3];
        let mut n = [0.0f32; 3];
        for k in 0..3 {
            p[k] = sg[k] * if k == a { hh[k] } else { hh[k] - b };
        }
        n[a] = sg[a];
        (c + v3(p[0], p[1], p[2]), v3(n[0], n[1], n[2]))
    };
    let s = [-1.0f32, 1.0];
    for a in 0..3 {
        let (u, w) = ((a + 1) % 3, (a + 2) % 3);
        for &sa in &s {
            let mut q = Vec::new();
            for (su, sw) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let mut sg = [0.0; 3];
                sg[a] = sa;
                sg[u] = su;
                sg[w] = sw;
                q.push(cv(a, sg));
            }
            let mut n = [0.0; 3];
            n[a] = sa;
            poly(&mut m, &q, v3(n[0], n[1], n[2]));
            for &su in &s {
                let mk = |ax: usize, sw: f32| {
                    let mut sg = [0.0; 3];
                    sg[a] = sa;
                    sg[u] = su;
                    sg[w] = sw;
                    cv(ax, sg)
                };
                let q = [mk(a, -1.0), mk(a, 1.0), mk(u, 1.0), mk(u, -1.0)];
                let mut n = [0.0; 3];
                n[a] = sa;
                n[u] = su;
                poly(&mut m, &q, v3(n[0], n[1], n[2]));
            }
        }
    }
    for &sx in &s {
        for &sy in &s {
            for &sz in &s {
                let sg = [sx, sy, sz];
                poly(&mut m, &[cv(0, sg), cv(1, sg), cv(2, sg)], v3(sx, sy, sz));
            }
        }
    }
    m
}

/// Cuts a simple polygon (counter-clockwise) into triangles by clipping ears.
fn triangulate(p: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let cross = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let mut out = Vec::new();
    while idx.len() > 3 {
        let m = idx.len();
        let mut cut = false;
        for i in 0..m {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            if cross(p[a], p[b], p[c]) <= 1e-9 {
                continue;
            }
            let inside = idx.iter().any(|&k| {
                k != a
                    && k != b
                    && k != c
                    && cross(p[a], p[b], p[k]) >= 0.0
                    && cross(p[b], p[c], p[k]) >= 0.0
                    && cross(p[c], p[a], p[k]) >= 0.0
            });
            if !inside {
                out.push([a, b, c]);
                idx.remove(i);
                cut = true;
                break;
            }
        }
        if !cut {
            out.push([idx[m - 1], idx[0], idx[1]]);
            idx.remove(0);
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

fn area2(o: &[[f32; 2]]) -> f32 {
    let n = o.len();
    (0..n)
        .map(|i| o[i][0] * o[(i + 1) % n][1] - o[(i + 1) % n][0] * o[i][1])
        .sum()
}

/// An outline in the xy plane pushed along z from `z0` to `z1`, its rims cut
/// back by `b`. The outline may be concave.
pub fn prism(outline: &[[f32; 2]], z0: f32, z1: f32, b: f32) -> Mesh {
    if outline.len() < 3 {
        return Mesh::default();
    }
    let mut o = outline.to_vec();
    if area2(&o) < 0.0 {
        o.reverse();
    }
    let n = o.len();
    let (z0, z1) = if z0 < z1 { (z0, z1) } else { (z1, z0) };
    let b = b.min((z1 - z0) * 0.45);
    let unit = |a: [f32; 2], c: [f32; 2]| {
        let (dx, dy) = (c[0] - a[0], c[1] - a[1]);
        let l = (dx * dx + dy * dy).sqrt().max(1e-9);
        [dx / l, dy / l]
    };
    let ins: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let (e0, e1) = (unit(o[(i + n - 1) % n], o[i]), unit(o[i], o[(i + 1) % n]));
            let (n0, n1) = ([-e0[1], e0[0]], [-e1[1], e1[0]]);
            let d = (1.0 + n0[0] * n1[0] + n0[1] * n1[1]).max(0.3);
            [
                o[i][0] + (n0[0] + n1[0]) * b / d,
                o[i][1] + (n0[1] + n1[1]) * b / d,
            ]
        })
        .collect();
    let mut m = Mesh::default();
    let tris = triangulate(&ins);
    for (z, nz) in [(z1, 1.0f32), (z0, -1.0)] {
        let nn = v3(0.0, 0.0, nz);
        for t in &tris {
            let q: Vec<(V3, V3)> = t
                .iter()
                .map(|&k| (v3(ins[k][0], ins[k][1], z), nn))
                .collect();
            poly(&mut m, &q, nn);
        }
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let e = unit(o[i], o[j]);
        let ne = v3(e[1], -e[0], 0.0);
        let p = |k: usize, z: f32| v3(o[k][0], o[k][1], z);
        let q = |k: usize, z: f32| v3(ins[k][0], ins[k][1], z);
        poly(
            &mut m,
            &[
                (p(i, z0 + b), ne),
                (p(j, z0 + b), ne),
                (p(j, z1 - b), ne),
                (p(i, z1 - b), ne),
            ],
            ne,
        );
        if b > 0.0 {
            for (zr, zc, nz) in [(z1 - b, z1, 1.0f32), (z0 + b, z0, -1.0)] {
                let nc = v3(0.0, 0.0, nz);
                poly(
                    &mut m,
                    &[
                        (p(i, zr), ne),
                        (p(j, zr), ne),
                        (q(j, zc), nc),
                        (q(i, zc), nc),
                    ],
                    ne + nc,
                );
            }
        }
    }
    m
}

/// An outline drawn in the side view (z across, y up) pushed along x.
pub fn prism_x(outline: &[[f32; 2]], x0: f32, x1: f32, b: f32) -> Mesh {
    let xf = M4 {
        m: [
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };
    let mut m = Mesh::default();
    add(&mut m, &prism(outline, -x1, -x0, b), &xf, false);
    m
}

/// An outline drawn in the top view (x across, z up the page) pushed along y.
pub fn prism_y(outline: &[[f32; 2]], y0: f32, y1: f32, b: f32) -> Mesh {
    let xf = M4 {
        m: [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };
    let mut m = Mesh::default();
    add(&mut m, &prism(outline, -y1, -y0, b), &xf, false);
    m
}

/// Outlines in the xy plane pushed along z from `z0` to `z1`, with the
/// non-zero rule and a soft rim: the solid a word or a mark makes.
pub fn extrude_polys(contours: &[Vec<[f32; 2]>], z0: f32, z1: f32) -> Mesh {
    extrude_outlines(contours, z0, z1, false, (z1 - z0).abs() * 0.12)
}

/// Outlines in the xy plane pushed along z from `z0` to `z1`. Any outlines:
/// holes, outlines that overlap one another or cross themselves, either
/// winding. What is solid follows the non-zero rule, or the even-odd rule
/// with `even_odd`, as the same outlines would fill in 2D.
///
/// The outlines are cut into bands at every corner and every crossing, so
/// inside a band the solid is a row of trapezoids. The faces are those
/// trapezoids; the walls stand on the edges where solid meets empty, facing
/// the empty side, so the inside of an O faces into the O. Every corner a
/// face has is a corner of the walls beside it and of the face across a band
/// line, so the solid has no cracks. Over `rim` at each end the walls'
/// normals turn halfway to the face, so light catches the edges as it would
/// a small bevel; 0 leaves them sharp.
pub fn extrude_outlines(
    contours: &[Vec<[f32; 2]>],
    z0: f32,
    z1: f32,
    even_odd: bool,
    rim: f32,
) -> Mesh {
    let (z0, z1) = if z0 < z1 { (z0, z1) } else { (z1, z0) };
    let rim = if rim > 0.0 {
        rim.min((z1 - z0) * 0.45)
    } else {
        0.0
    };
    let outlines: Vec<Vec<[f32; 2]>> = contours
        .iter()
        .map(|c| clean_outline(c))
        .filter(|c| c.len() >= 3)
        .collect();
    let mut m = Mesh::default();
    // Outlines whose boxes do not touch are cut on their own, so one letter's
    // corners do not cut the bands of the next.
    for island in islands(&outlines) {
        extrude_island(&mut m, &outlines, &island, z0, z1, even_odd, rim);
    }
    m
}

/// An outline with no repeated corners, no closing repeat and nothing that
/// is not a number.
fn clean_outline(c: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut o: Vec<[f32; 2]> = Vec::with_capacity(c.len());
    for &p in c {
        if p[0].is_finite() && p[1].is_finite() && o.last() != Some(&p) {
            o.push(p);
        }
    }
    while o.len() > 1 && o.first() == o.last() {
        o.pop();
    }
    o
}

/// Groups of outlines whose boxes overlap, each in the order given.
fn islands(outlines: &[Vec<[f32; 2]>]) -> Vec<Vec<usize>> {
    let boxes: Vec<[f32; 4]> = outlines
        .iter()
        .map(|c| {
            c.iter()
                .fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, p| {
                    [
                        b[0].min(p[0]),
                        b[1].min(p[1]),
                        b[2].max(p[0]),
                        b[3].max(p[1]),
                    ]
                })
        })
        .collect();
    let mut group: Vec<usize> = (0..outlines.len()).collect();
    fn root(g: &mut [usize], mut i: usize) -> usize {
        while g[i] != i {
            g[i] = g[g[i]];
            i = g[i];
        }
        i
    }
    for i in 0..boxes.len() {
        for j in i + 1..boxes.len() {
            let (a, b) = (boxes[i], boxes[j]);
            if a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3] {
                let (ri, rj) = (root(&mut group, i), root(&mut group, j));
                if ri != rj {
                    group[ri.max(rj)] = ri.min(rj);
                }
            }
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut slot: Vec<usize> = vec![usize::MAX; outlines.len()];
    for i in 0..outlines.len() {
        let r = root(&mut group, i);
        if slot[r] == usize::MAX {
            slot[r] = out.len();
            out.push(Vec::new());
        }
        out[slot[r]].push(i);
    }
    out
}

/// One edge of an outline that is not level, stored low end first.
#[derive(Clone, Copy)]
struct Edge {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    /// +1 if the outline runs up it, -1 if down.
    w: i32,
    /// The normal at the low end and at the high end, along the outline's
    /// own right hand, smoothed across gentle corners.
    n0: V3,
    n1: V3,
}

impl Edge {
    /// Where the edge is at height `y`. The ends are exact, so two pieces of
    /// one edge, and the faces beside them, meet on the same bits.
    fn x(&self, y: f32) -> f32 {
        if y == self.y0 {
            self.x0
        } else if y == self.y1 {
            self.x1
        } else {
            self.x0 + (self.x1 - self.x0) * ((y - self.y0) / (self.y1 - self.y0))
        }
    }
    fn normal(&self, y: f32) -> V3 {
        let t = (y - self.y0) / (self.y1 - self.y0);
        self.n0.lerp(self.n1, t).norm()
    }
}

/// A run of solid across one band: the edges at its two ends.
#[derive(Clone, Copy)]
struct Span {
    left: usize,
    right: usize,
}

fn extrude_island(
    m: &mut Mesh,
    outlines: &[Vec<[f32; 2]>],
    island: &[usize],
    z0: f32,
    z1: f32,
    even_odd: bool,
    rim: f32,
) {
    // Corners whose two edges turn by less than this share a normal, so a
    // curve reads as a curve and a corner stays a corner.
    const SMOOTH: f32 = 0.82;
    let mut edges: Vec<Edge> = Vec::new();
    for &ci in island {
        let c = &outlines[ci];
        let n = c.len();
        let side = |k: usize| -> V3 {
            let (a, b) = (c[k % n], c[(k + 1) % n]);
            v3(b[1] - a[1], -(b[0] - a[0]), 0.0).norm()
        };
        for k in 0..n {
            let (a, b) = (c[k], c[(k + 1) % n]);
            if a[1] == b[1] {
                continue;
            }
            let here = side(k);
            let (before, after) = (side(k + n - 1), side(k + 1));
            let at_a = if here.dot(before) > SMOOTH {
                (here + before).norm()
            } else {
                here
            };
            let at_b = if here.dot(after) > SMOOTH {
                (here + after).norm()
            } else {
                here
            };
            edges.push(if a[1] < b[1] {
                Edge {
                    x0: a[0],
                    y0: a[1],
                    x1: b[0],
                    y1: b[1],
                    w: 1,
                    n0: at_a,
                    n1: at_b,
                }
            } else {
                Edge {
                    x0: b[0],
                    y0: b[1],
                    x1: a[0],
                    y1: a[1],
                    w: -1,
                    n0: at_b,
                    n1: at_a,
                }
            });
        }
    }
    if edges.is_empty() {
        return;
    }
    edges.sort_by(|a, b| a.y0.total_cmp(&b.y0).then(a.x0.total_cmp(&b.x0)));

    // Band lines: every corner, and every height where two edges cross.
    let mut ys: Vec<f32> = edges.iter().flat_map(|e| [e.y0, e.y1]).collect();
    let mut live: Vec<usize> = Vec::new();
    for i in 0..edges.len() {
        let e = edges[i];
        live.retain(|&j| edges[j].y1 > e.y0);
        for &j in &live {
            let o = edges[j];
            let (lo, hi) = (e.y0.max(o.y0), e.y1.min(o.y1));
            if lo >= hi {
                continue;
            }
            let (d0, d1) = (e.x(lo) - o.x(lo), e.x(hi) - o.x(hi));
            if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
                let y = lo + (hi - lo) * (d0 / (d0 - d1));
                if y > lo && y < hi {
                    ys.push(y);
                }
            }
        }
        live.push(i);
    }
    ys.sort_by(f32::total_cmp);
    ys.dedup();
    if ys.len() < 2 {
        return;
    }

    // The runs of solid in each band.
    let solid = |w: i32| if even_odd { w & 1 != 0 } else { w != 0 };
    let mut bands: Vec<Vec<Span>> = Vec::with_capacity(ys.len() - 1);
    let mut next = 0usize;
    live.clear();
    for b in 0..ys.len() - 1 {
        let (ya, yb) = (ys[b], ys[b + 1]);
        while next < edges.len() && edges[next].y0 <= ya {
            live.push(next);
            next += 1;
        }
        live.retain(|&j| edges[j].y1 > ya);
        let mid = (ya + yb) * 0.5;
        live.sort_by(|&i, &j| edges[i].x(mid).total_cmp(&edges[j].x(mid)).then(i.cmp(&j)));
        let mut w = 0;
        let mut start = usize::MAX;
        let mut runs = Vec::new();
        for &j in &live {
            let was = solid(w);
            w += edges[j].w;
            match (was, solid(w)) {
                (false, true) => start = j,
                (true, false) => runs.push(Span {
                    left: start,
                    right: j,
                }),
                _ => {}
            }
        }
        bands.push(runs);
    }

    // Where the runs of one band meet a band line, from below or above.
    let ends = |b: usize, top: bool| -> Vec<[f32; 2]> {
        let y = if top { ys[b + 1] } else { ys[b] };
        bands[b]
            .iter()
            .map(|s| [edges[s.left].x(y), edges[s.right].x(y)])
            .collect()
    };
    // Every x on band line `l` where some run starts or ends, from either side.
    let cuts = |l: usize| -> Vec<f32> {
        let mut xs: Vec<f32> = Vec::new();
        if l > 0 {
            xs.extend(ends(l - 1, true).iter().flatten());
        }
        if l + 1 < ys.len() {
            xs.extend(ends(l, false).iter().flatten());
        }
        xs.sort_by(f32::total_cmp);
        xs.dedup();
        xs
    };

    let zs: Vec<f32> = if rim > 0.0 {
        vec![z0, z0 + rim, z1 - rim, z1]
    } else {
        vec![z0, z1]
    };
    let up = v3(0.0, 0.0, 1.0);
    // A wall's normal at height z: turned halfway to the face at each end.
    let turn = |n: V3, z: f32| -> V3 {
        if rim <= 0.0 {
            n
        } else if z == z1 {
            (n + up).norm()
        } else if z == z0 {
            (n - up).norm()
        } else {
            n
        }
    };
    // A wall standing on the line from a to b, with the solid on its right
    // seen from above (walking from a to b), so it faces left of that walk;
    // `na` and `nb` are its normals at the two ends.
    let wall = |m: &mut Mesh, a: [f32; 2], b: [f32; 2], na: V3, nb: V3| {
        for k in 0..zs.len() - 1 {
            let (za, zb) = (zs[k], zs[k + 1]);
            quad(
                m,
                [
                    (v3(b[0], b[1], za), turn(nb, za)),
                    (v3(a[0], a[1], za), turn(na, za)),
                    (v3(a[0], a[1], zb), turn(na, zb)),
                    (v3(b[0], b[1], zb), turn(nb, zb)),
                ],
            );
        }
    };

    let line_cuts: Vec<Vec<f32>> = (0..ys.len()).map(cuts).collect();
    for b in 0..bands.len() {
        let (ya, yb) = (ys[b], ys[b + 1]);
        for s in &bands[b] {
            let (l, r) = (edges[s.left], edges[s.right]);
            let (la, lb, ra, rb) = (l.x(ya), l.x(yb), r.x(ya), r.x(yb));
            // The faces, with every cut on their two level sides as a corner.
            let mut ring: Vec<[f32; 2]> = vec![[la, ya]];
            ring.extend(
                line_cuts[b]
                    .iter()
                    .filter(|&&x| x > la && x < ra)
                    .map(|&x| [x, ya]),
            );
            ring.push([ra, ya]);
            ring.push([rb, yb]);
            ring.extend(
                line_cuts[b + 1]
                    .iter()
                    .rev()
                    .filter(|&&x| x > lb && x < rb)
                    .map(|&x| [x, yb]),
            );
            ring.push([lb, yb]);
            // The ring runs counter-clockwise seen from above.
            let top: Vec<V3> = ring.iter().map(|p| v3(p[0], p[1], z1)).collect();
            let bottom: Vec<V3> = ring.iter().rev().map(|p| v3(p[0], p[1], z0)).collect();
            fan(m, &top, up);
            fan(m, &bottom, -up);
            // The walls at the run's two ends face away from it. Which way the
            // smoothed normal points is settled on the whole edge, not on
            // this piece of it, which may be a sliver.
            let out = |e: &Edge, left: bool, y: f32| {
                let g = if left {
                    v3(-(e.y1 - e.y0), e.x1 - e.x0, 0.0)
                } else {
                    v3(e.y1 - e.y0, -(e.x1 - e.x0), 0.0)
                };
                let n = e.normal(y);
                if n.dot(g) < 0.0 { -n } else { n }
            };
            // Solid on the right of a walk up the left end, and of a walk down the right end.
            wall(m, [la, ya], [lb, yb], out(&l, true, ya), out(&l, true, yb));
            wall(
                m,
                [rb, yb],
                [ra, ya],
                out(&r, false, yb),
                out(&r, false, ya),
            );
        }
    }

    // The level walls: on each band line, where solid lies on one side only.
    for l in 0..ys.len() {
        let y = ys[l];
        let below: Vec<[f32; 2]> = if l > 0 { ends(l - 1, true) } else { Vec::new() };
        let above: Vec<[f32; 2]> = if l + 1 < ys.len() {
            ends(l, false)
        } else {
            Vec::new()
        };
        let inside = |runs: &[[f32; 2]], x: f32| runs.iter().any(|r| x > r[0] && x < r[1]);
        let xs = &line_cuts[l];
        for k in 0..xs.len().saturating_sub(1) {
            let (xa, xb) = (xs[k], xs[k + 1]);
            let mid = (xa + xb) * 0.5;
            match (inside(&below, mid), inside(&above, mid)) {
                // Solid below: walk rightwards. Solid above: walk leftwards.
                (true, false) => wall(m, [xa, y], [xb, y], v3(0.0, 1.0, 0.0), v3(0.0, 1.0, 0.0)),
                (false, true) => wall(m, [xb, y], [xa, y], v3(0.0, -1.0, 0.0), v3(0.0, -1.0, 0.0)),
                _ => {}
            }
        }
    }
}

/// A profile of (radius, height) turned about the y axis from angle `a0` to
/// `a1` in `seg` steps. The profile runs counter-clockwise with radius to the
/// right and height up, so the solid is on its left. A full turn needs no
/// ends; a part turn needs `closed` and gets flat ends. Neighbouring bands
/// shade as one when their normals agree better than `soft` (a cosine).
pub fn lathe_arc(prof: &[[f32; 2]], seg: usize, closed: bool, soft: f32, a0: f32, a1: f32) -> Mesh {
    let n = prof.len();
    if n < 2 || seg < 3 {
        return Mesh::default();
    }
    let segs = if closed { n } else { n - 1 };
    let full = (a1 - a0 - TAU).abs() < 1e-4;
    let nrm = |k: usize| -> Option<[f32; 2]> {
        let (p, q) = (prof[k % n], prof[(k + 1) % n]);
        let (dr, dy) = (q[0] - p[0], q[1] - p[1]);
        let l = (dr * dr + dy * dy).sqrt();
        if l < 1e-7 {
            None
        } else {
            Some([dy / l, -dr / l])
        }
    };
    let blend = |a: [f32; 2], o: Option<[f32; 2]>| -> [f32; 2] {
        match o {
            Some(b) if a[0] * b[0] + a[1] * b[1] > soft => [a[0] + b[0], a[1] + b[1]],
            _ => a,
        }
    };
    let mut m = Mesh::default();
    let ang: Vec<(f32, f32)> = (0..=seg)
        .map(|j| {
            let a = a0
                + (a1 - a0)
                    * (if full && j == seg {
                        0.0
                    } else {
                        j as f32 / seg as f32
                    });
            (cos(a), sin(a))
        })
        .collect();
    for k in 0..segs {
        let Some(nk) = nrm(k) else { continue };
        let prev = if k > 0 || closed {
            nrm((k + n - 1) % n)
        } else {
            None
        };
        let next = if k + 1 < segs || closed {
            nrm(k + 1)
        } else {
            None
        };
        let (ns, ne) = (blend(nk, prev), blend(nk, next));
        let (p, q) = (prof[k], prof[(k + 1) % n]);
        for j in 0..seg {
            let ((c0, s0), (c1, s1)) = (ang[j], ang[j + 1]);
            let pt = |r: [f32; 2], c: f32, s: f32, nn: [f32; 2]| {
                (
                    v3(r[0] * c, r[1], r[0] * s),
                    v3(nn[0] * c, nn[1], nn[0] * s),
                )
            };
            let quad = [
                pt(p, c0, s0, ns),
                pt(p, c1, s1, ns),
                pt(q, c1, s1, ne),
                pt(q, c0, s0, ne),
            ];
            let cm = (c0 + c1) * 0.5;
            let sm = (s0 + s1) * 0.5;
            poly(&mut m, &quad, v3(nk[0] * cm, nk[1], nk[0] * sm));
        }
    }
    if !full && closed {
        let mut o = prof.to_vec();
        if area2(&o) < 0.0 {
            o.reverse();
        }
        for (j, sg) in [(0usize, -1.0f32), (seg, 1.0)] {
            let (c, s) = ang[j];
            let nn = v3(-s * sg, 0.0, c * sg);
            for t in triangulate(&o) {
                let q: Vec<(V3, V3)> = t
                    .iter()
                    .map(|&k| (v3(o[k][0] * c, o[k][1], o[k][0] * s), nn))
                    .collect();
                poly(&mut m, &q, nn);
            }
        }
    }
    m
}

/// A full turn of a profile that starts and ends on the axis.
pub fn lathe(prof: &[[f32; 2]], seg: usize) -> Mesh {
    lathe_arc(prof, seg, false, 0.82, 0.0, TAU)
}

/// A full turn of a closed profile: a ring or a tube.
pub fn ring(prof: &[[f32; 2]], seg: usize) -> Mesh {
    lathe_arc(prof, seg, true, 0.82, 0.0, TAU)
}

fn placed(mesh: &Mesh, a: V3, b: V3) -> Mesh {
    let mut m = Mesh::default();
    add(&mut m, mesh, &frame_y(a, b), false);
    m
}

/// A rod from `a` to `b`, radius `r0` at `a` and `r1` at `b`, ends cut back by `bv`.
pub fn cone(a: V3, b: V3, r0: f32, r1: f32, seg: usize, bv: f32) -> Mesh {
    let l = (b - a).len();
    let bv = bv.min(l * 0.4).min(r0 * 0.6).min(r1 * 0.6);
    placed(
        &lathe(
            &[
                [0.0, 0.0],
                [r0 - bv, 0.0],
                [r0, bv],
                [r1, l - bv],
                [r1 - bv, l],
                [0.0, l],
            ],
            seg,
        ),
        a,
        b,
    )
}

pub fn cyl(a: V3, b: V3, r: f32, seg: usize, bv: f32) -> Mesh {
    cone(a, b, r, r, seg, bv)
}

/// A pipe from `a` to `b` with a bore.
pub fn tube(a: V3, b: V3, ro: f32, ri: f32, seg: usize) -> Mesh {
    let l = (b - a).len();
    let bv = ((ro - ri) * 0.25).min(l * 0.3);
    placed(
        &ring(
            &[
                [ri, 0.0],
                [ro - bv, 0.0],
                [ro, bv],
                [ro, l - bv],
                [ro - bv, l],
                [ri, l],
            ],
            seg,
        ),
        a,
        b,
    )
}

/// A six-sided bolt head standing on `p`, pointing along `n`.
pub fn bolt(p: V3, n: V3, r: f32, h: f32) -> Mesh {
    let n = n.norm();
    placed(
        &lathe_arc(
            &[[0.0, -h], [r, -h], [r, h * 0.7], [r * 0.7, h], [0.0, h]],
            6,
            false,
            2.0,
            0.0,
            TAU,
        ),
        p,
        p + n,
    )
}

/// A ball of radius `r` about `c`, squashed to `sy` along y.
pub fn ball(c: V3, r: f32, sy: f32, seg: usize) -> Mesh {
    let rings = (seg / 2).max(2);
    let prof: Vec<[f32; 2]> = (0..=rings)
        .map(|i| {
            let a = -TAU * 0.25 + TAU * 0.5 * i as f32 / rings as f32;
            [
                if i == 0 || i == rings {
                    0.0
                } else {
                    r * cos(a)
                },
                r * sy * sin(a),
            ]
        })
        .collect();
    let mut m = Mesh::default();
    add(
        &mut m,
        &lathe_arc(&prof, seg, false, 0.0, 0.0, TAU),
        &M4::translate(c),
        false,
    );
    m
}

pub fn p(x: f32, y: f32, z: f32) -> V3 {
    v3(x, y, z)
}
/// A bevelled box by centre and half sizes.
pub fn bx(c: [f32; 3], h: [f32; 3], b: f32) -> Mesh {
    bbox(v3(c[0], c[1], c[2]), v3(h[0], h[1], h[2]), b)
}
/// A rod along x.
pub fn cx(x0: f32, x1: f32, y: f32, z: f32, r: f32, seg: usize) -> Mesh {
    cyl(v3(x0, y, z), v3(x1, y, z), r, seg, r * 0.12)
}
/// A rod along y.
pub fn cyv(x: f32, y0: f32, y1: f32, z: f32, r: f32, seg: usize) -> Mesh {
    cyl(v3(x, y0, z), v3(x, y1, z), r, seg, r * 0.12)
}
/// A rod along z.
pub fn cz(x: f32, y: f32, z0: f32, z1: f32, r: f32, seg: usize) -> Mesh {
    cyl(v3(x, y, z0), v3(x, y, z1), r, seg, r * 0.12)
}
/// A pipe along z.
pub fn tz(x: f32, y: f32, z0: f32, z1: f32, ro: f32, ri: f32, seg: usize) -> Mesh {
    tube(v3(x, y, z0), v3(x, y, z1), ro, ri, seg)
}
/// Several meshes as one.
pub fn join(list: &[Mesh]) -> Mesh {
    let mut m = Mesh::default();
    for s in list {
        add(&mut m, s, &M4::IDENT, false);
    }
    m
}
/// A row of slats: `n` thin boxes from `a` to `b` (centres), each of half size `h`.
pub fn slats(a: [f32; 3], b: [f32; 3], n: usize, h: [f32; 3]) -> Mesh {
    let mut m = Mesh::default();
    for i in 0..n {
        let t = if n > 1 {
            i as f32 / (n - 1) as f32
        } else {
            0.0
        };
        let c = [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ];
        add(
            &mut m,
            &bx(c, h, h[0].min(h[1]).min(h[2]) * 0.4),
            &M4::IDENT,
            false,
        );
    }
    m
}
/// A mesh moved by `v`.
pub fn at(m: &Mesh, v: V3) -> Mesh {
    let mut o = Mesh::default();
    add(&mut o, m, &M4::translate(v), false);
    o
}
/// A mesh turned about `a` so that its y axis runs toward `b`.
pub fn along(m: &Mesh, a: V3, b: V3) -> Mesh {
    let mut o = Mesh::default();
    add(&mut o, m, &frame_y(a, b), false);
    o
}
/// A mesh grown by `s` about `c`, then moved by `d`.
pub fn scaled(m: &Mesh, c: V3, s: f32, d: V3) -> Mesh {
    let mut o = m.clone();
    for q in o.pos.iter_mut() {
        *q = c + (*q - c) * s + d;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Every edge of every triangle is met by the same edge running the
    /// other way, by position: the solid has no cracks and no faces turned
    /// inside out.
    fn closed(m: &Mesh) -> bool {
        let key = |p: V3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
        let mut count: HashMap<_, i32> = HashMap::new();
        for t in m.idx.as_chunks::<3>().0 {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *count
                    .entry((key(m.pos[a as usize]), key(m.pos[b as usize])))
                    .or_default() += 1;
            }
        }
        count
            .iter()
            .all(|(&(a, b), &n)| count.get(&(b, a)).copied().unwrap_or(0) == n)
    }

    fn volume(m: &Mesh) -> f32 {
        m.idx
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let (a, b, c) = (
                    m.pos[t[0] as usize],
                    m.pos[t[1] as usize],
                    m.pos[t[2] as usize],
                );
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// Each triangle faces the way its corners' normals say. A sliver one
    /// rounding step thick has no direction of its own, so it is not asked.
    fn facing(m: &Mesh) -> bool {
        m.idx.as_chunks::<3>().0.iter().all(|t| {
            let (a, b, c) = (
                m.pos[t[0] as usize],
                m.pos[t[1] as usize],
                m.pos[t[2] as usize],
            );
            let g = (b - a).cross(c - a);
            g.len() < 1e-6 || t.iter().all(|&i| g.dot(m.nrm[i as usize]) > 0.0)
        })
    }

    fn square(x: f32, y: f32, s: f32, ccw: bool) -> Vec<[f32; 2]> {
        let mut o = vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]];
        if !ccw {
            o.reverse();
        }
        o
    }

    fn ring(r: f32, n: usize, ccw: bool) -> Vec<[f32; 2]> {
        let mut o: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let a = TAU * i as f32 / n as f32;
                [r * cos(a), r * sin(a)]
            })
            .collect();
        if !ccw {
            o.reverse();
        }
        o
    }

    fn solid(contours: &[Vec<[f32; 2]>], even_odd: bool, area: f32) {
        for rim in [0.0, 0.05] {
            let m = extrude_outlines(contours, -0.25, 0.25, even_odd, rim);
            assert!(closed(&m), "cracks or faces turned inside out (rim {rim})");
            assert!(facing(&m), "a face turned against its normals (rim {rim})");
            let v = volume(&m);
            assert!(
                (v - area * 0.5).abs() < 1e-3 * area.max(1.0),
                "volume {v}, expected {} (rim {rim})",
                area * 0.5
            );
        }
    }

    #[test]
    fn a_hole_is_a_hole_whichever_way_it_winds() {
        solid(
            &[square(0.0, 0.0, 4.0, true), square(1.0, 1.0, 2.0, false)],
            false,
            12.0,
        );
        solid(
            &[square(0.0, 0.0, 4.0, false), square(1.0, 1.0, 2.0, true)],
            false,
            12.0,
        );
        solid(
            &[square(0.0, 0.0, 4.0, true), square(1.0, 1.0, 2.0, true)],
            true,
            12.0,
        );
        // Wound the same way under the non-zero rule, the inner square is solid.
        solid(
            &[square(0.0, 0.0, 4.0, true), square(1.0, 1.0, 2.0, true)],
            false,
            16.0,
        );
    }

    #[test]
    fn outlines_that_overlap_join_or_cancel_by_the_rule() {
        let two = [square(0.0, 0.0, 2.0, true), square(1.0, 1.0, 2.0, true)];
        solid(&two, false, 7.0);
        solid(&two, true, 6.0);
    }

    #[test]
    fn an_outline_that_crosses_itself_is_filled_both_sides() {
        solid(
            &[vec![[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]],
            false,
            2.0,
        );
    }

    #[test]
    fn an_e_with_its_level_bars_is_whole() {
        let e = vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [3.0, 4.0],
            [3.0, 5.0],
            [0.0, 5.0],
        ];
        solid(&[e.clone()], false, 10.0);
        let mut back = e;
        back.reverse();
        solid(&[back], false, 10.0);
    }

    #[test]
    fn an_o_faces_into_its_hole() {
        let (outer, inner) = (ring(1.0, 48, true), ring(0.6, 48, false));
        let area = |o: &[[f32; 2]]| area2(o).abs() * 0.5;
        solid(
            &[outer.clone(), inner.clone()],
            false,
            area(&outer) - area(&inner),
        );
        // The wall nearest the middle faces the middle.
        let m = extrude_outlines(&[outer, inner], -0.25, 0.25, false, 0.0);
        let near = (0..m.pos.len())
            .filter(|&i| m.nrm[i].z == 0.0)
            .min_by(|&i, &j| {
                let (a, b) = (m.pos[i], m.pos[j]);
                (a.x * a.x + a.y * a.y).total_cmp(&(b.x * b.x + b.y * b.y))
            })
            .unwrap();
        let (p, n) = (m.pos[near], m.nrm[near]);
        assert!(
            p.x * n.x + p.y * n.y < 0.0,
            "the inside of the O faces outwards"
        );
    }

    #[test]
    fn letters_apart_are_cut_apart() {
        let m = extrude_outlines(
            &[square(0.0, 0.0, 1.0, true), square(5.0, 0.3, 1.0, true)],
            0.0,
            0.2,
            false,
            0.0,
        );
        assert!(closed(&m));
        // Two cubes need no more than two caps of two triangles and four walls each.
        assert!(
            m.idx.len() / 3 <= 2 * (2 * 4 + 4 * 2),
            "{} triangles",
            m.idx.len() / 3
        );
    }
}
