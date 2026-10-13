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

/// Extrudes a multi-contour 2D polygon (supporting holes, such as letters O/A/B
/// or brand logos) along z from `z0` to `z1`.
pub fn extrude_polys(contours: &[Vec<[f32; 2]>], z0: f32, z1: f32) -> Mesh {
    use crate::geometry::tessellate::fill_polys;
    use crate::math::Vec2;
    let (z0, z1) = if z0 < z1 { (z0, z1) } else { (z1, z0) };
    let polys_v2: Vec<Vec<Vec2>> = contours
        .iter()
        .filter(|c| c.len() >= 3)
        .map(|c| c.iter().map(|p| Vec2::new(p[0], p[1])).collect())
        .collect();
    let tris = fill_polys(polys_v2);
    let mut m = Mesh::default();
    for (z, nz) in [(z1, 1.0f32), (z0, -1.0)] {
        let nn = v3(0.0, 0.0, nz);
        for t in tris.indices.as_chunks::<3>().0 {
            let p0 = tris.vertices[t[0] as usize];
            let p1 = tris.vertices[t[1] as usize];
            let p2 = tris.vertices[t[2] as usize];
            poly(
                &mut m,
                &[
                    (v3(p0.x, p0.y, z), nn),
                    (v3(p1.x, p1.y, z), nn),
                    (v3(p2.x, p2.y, z), nn),
                ],
                nn,
            );
        }
    }
    for c in contours {
        if c.len() < 3 {
            continue;
        }
        let n = c.len();
        let sign = if area2(c) >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..n {
            let j = (i + 1) % n;
            let (dx, dy) = (c[j][0] - c[i][0], c[j][1] - c[i][1]);
            let l = (dx * dx + dy * dy).sqrt().max(1e-9);
            let ne = v3(dy / l * sign, -dx / l * sign, 0.0);
            poly(
                &mut m,
                &[
                    (v3(c[i][0], c[i][1], z0), ne),
                    (v3(c[j][0], c[j][1], z0), ne),
                    (v3(c[j][0], c[j][1], z1), ne),
                    (v3(c[i][0], c[i][1], z1), ne),
                ],
                ne,
            );
        }
    }
    m
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
