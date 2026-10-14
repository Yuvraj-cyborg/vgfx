//! Forms the shape builders do not make from a profile or an outline: a
//! solid swept through sections, for the bodies of vehicles and products,
//! whose width, height and roundness all change from front to back. Only +,
//! -, *, /, sqrt and the polynomial sin and cos are used, so the meshes are
//! the same bits on every machine.

use super::math::{PI, cos, sin, v3};
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
        for t in m.idx.chunks_exact(3) {
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
}
