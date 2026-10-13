//! Path tessellation: bezier flattening, fill triangulation (even-odd rule,
//! holes via edge bridging, ear clipping) and stroke outline generation
//! (miter/bevel joins).
//!
//! Everything here is pure CPU math; the GPU backend just uploads the
//! resulting triangles.

use crate::geometry::affine::Affine;
use crate::geometry::path::{Path, PathCmd};
use crate::math::Vec2;

/// Default flattening tolerance (user units, e.g. pixels).
pub const DEFAULT_TOLERANCE: f32 = 0.1;

/// Triangulated 2D geometry ready for upload.
#[derive(Clone, Debug, Default)]
pub struct Triangles {
    pub vertices: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl Triangles {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Signed area covered by the triangulation.
    #[must_use]
    pub fn area(&self) -> f32 {
        let mut a = 0.0f32;
        for t in self.indices.as_chunks::<3>().0 {
            let p0 = self.vertices[t[0] as usize];
            let p1 = self.vertices[t[1] as usize];
            let p2 = self.vertices[t[2] as usize];
            a += (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
        }
        a * 0.5
    }
}

/// A flattened (polyline) subpath.
#[derive(Clone, Debug, Default)]
pub struct Subpath {
    pub points: Vec<Vec2>,
    pub closed: bool,
}

fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

#[must_use]
fn signed_area(poly: &[Vec2]) -> f32 {
    let mut a = 0.0f32;
    let n = poly.len();
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}

/// Fixed subdivision: identical work per cubic regardless of curvature/scale.
/// Tolerance is retained for source compatibility; quality is 32 segments/cubic.
fn flatten_cubic(
    p0: Vec2,
    c1: Vec2,
    c2: Vec2,
    p1: Vec2,
    _tol: f32,
    _depth: u32,
    out: &mut Vec<Vec2>,
) {
    const SEGMENTS: usize = 32;
    out.reserve(SEGMENTS);
    for i in 1..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        let u = 1.0 - t;
        out.push(
            p0 * (u * u * u) + c1 * (3.0 * u * u * t) + c2 * (3.0 * u * t * t) + p1 * (t * t * t),
        );
    }
}

/// Finish one subpath: drop degenerate points, then store it if non-trivial.
///
/// Two cleanups happen here, both load-bearing for stroking:
/// - consecutive duplicates would become zero-length segments, and
///   `offset_loops` rejects the whole subpath on those (a stroked
///   rounded rect would silently vanish);
/// - a closed path whose last point re-visits the start (exact for
///   `rounded_rect`, float-error sliver ~1e-5 for `circle`) would add a
///   zero/sliver wrap segment, self-intersecting the offsets so holes
///   fill in (a stroked circle rendered as a disc).
fn finish_subpath(current: &mut Vec<Vec2>, closed: bool, subs: &mut Vec<Subpath>) {
    current.dedup();
    if closed {
        while current.len() > 1 && current[current.len() - 1].distance_squared(current[0]) < 1e-8 {
            current.pop();
        }
    }
    if current.len() > 1 {
        subs.push(Subpath {
            points: std::mem::take(current),
            closed,
        });
    } else {
        current.clear();
    }
}

/// Flatten a path into polyline subpaths.
#[must_use]
pub fn flatten(path: &Path, tolerance: f32) -> Vec<Subpath> {
    let mut subs: Vec<Subpath> = Vec::new();
    let mut current: Vec<Vec2> = Vec::new();
    let mut closed = false;
    let mut cur = Vec2::ZERO;
    for cmd in path.cmds() {
        match cmd {
            PathCmd::MoveTo(p) => {
                finish_subpath(&mut current, closed, &mut subs);
                closed = false;
                cur = *p;
                current.push(*p);
            }
            PathCmd::LineTo(p) => {
                cur = *p;
                current.push(*p);
            }
            PathCmd::CubicTo { c1, c2, end } => {
                flatten_cubic(cur, *c1, *c2, *end, tolerance, 0, &mut current);
                cur = *end;
            }
            PathCmd::Close => closed = true,
        }
    }
    finish_subpath(&mut current, closed, &mut subs);
    subs
}

/// Even-odd ray cast point-in-polygon test.
#[must_use]
fn point_in_poly(p: Vec2, poly: &[Vec2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
    }
    inside
}

#[must_use]
fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = cross2(p - a, b - a);
    let d2 = cross2(p - b, c - b);
    let d3 = cross2(p - c, a - c);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn segments_properly_intersect(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> bool {
    let d = cross2(p2 - p1, p4 - p3);
    if d.abs() < 1e-12 {
        return false;
    }
    let t = cross2(p3 - p1, p4 - p3) / d;
    let u = cross2(p3 - p1, p2 - p1) / d;
    (1e-9..1.0 - 1e-9).contains(&t) && (1e-9..1.0 - 1e-9).contains(&u)
}

/// Find a valid bridge (o, h) connecting outer polygon `outer` to hole
/// `hole` such that the segment crosses no edges of either.
fn find_bridge(outer: &[Vec2], hole: &[Vec2]) -> Option<(usize, usize)> {
    // Heuristic start: rightmost hole vertex / rightmost outer vertex.
    let hi = (0..hole.len())
        .max_by(|&a, &b| hole[a].x.total_cmp(&hole[b].x))
        .unwrap();
    let oi = (0..outer.len())
        .max_by(|&a, &b| outer[a].x.total_cmp(&outer[b].x))
        .unwrap();
    for dh in 0..hole.len() {
        for do_ in 0..outer.len() {
            let h = (hi + dh) % hole.len();
            let o = (oi + do_) % outer.len();
            let hp = hole[h];
            let op = outer[o];
            let mut ok = true;
            for i in 0..hole.len() {
                let a = hole[i];
                let b = hole[(i + 1) % hole.len()];
                if i == h || (i + 1) % hole.len() == h {
                    continue;
                }
                if segments_properly_intersect(hp, op, a, b) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            for i in 0..outer.len() {
                let a = outer[i];
                let b = outer[(i + 1) % outer.len()];
                if i == o || (i + 1) % outer.len() == o {
                    continue;
                }
                if segments_properly_intersect(hp, op, a, b) {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some((o, h));
            }
        }
    }
    None
}

/// Splice `hole` into `poly` (both CCW) using the bridge technique:
/// walk the outer loop from the bridge vertex, cross over to the hole,
/// walk the hole backwards (to keep one consistent winding), cross back.
fn merge_hole(poly: &mut Vec<Vec2>, hole: &[Vec2]) -> bool {
    let Some((oi, hi)) = find_bridge(poly, hole) else {
        return false;
    };
    let n = poly.len();
    let m = hole.len();
    let mut merged: Vec<Vec2> = Vec::with_capacity(n + m + 2);
    // Outer loop starting at the bridge vertex o (visited twice: once
    // before crossing to the hole, once implicitly when closing back).
    merged.extend_from_slice(&poly[oi..]);
    merged.extend_from_slice(&poly[..oi]);
    merged.push(poly[oi]);
    // Hole walked backwards starting and ending at h.
    for k in 0..m {
        merged.push(hole[(hi + m - k) % m]);
    }
    merged.push(hole[hi]);
    *poly = merged;
    true
}

/// Ear-clipping triangulation of a (near-)simple CCW polygon.
#[must_use]
fn triangulate_polygon(pts: &[Vec2]) -> Vec<u32> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let mut idx: Vec<u32> = (0..n as u32).collect();
    // Drop consecutive duplicates.
    let mut deduped: Vec<u32> = Vec::with_capacity(idx.len());
    for &i in &idx {
        let i = i as usize;
        if deduped.is_empty() || pts[i] != pts[deduped[deduped.len() - 1] as usize] {
            deduped.push(i as u32);
        }
    }
    idx = deduped;
    if idx.len() > 3 {
        let m = idx.len();
        let mut out: Vec<u32> = Vec::with_capacity(m);
        for k in 0..m {
            let a = pts[idx[(k + m - 1) % m] as usize];
            let b = pts[idx[k] as usize];
            let c = pts[idx[(k + 1) % m] as usize];
            if cross2(c - a, b - a).abs() > 1e-9 {
                out.push(idx[k]);
            }
        }
        if out.len() >= 3 {
            idx = out;
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    let mut guard = (idx.len() * idx.len() + 64) as i64;
    while idx.len() > 3 {
        guard -= 1;
        let mut ear_found = false;
        let len = idx.len();
        for i in 0..len {
            let a = idx[(i + len - 1) % len] as usize;
            let b = idx[i] as usize;
            let c = idx[(i + 1) % len] as usize;
            let pa = pts[a];
            let pb = pts[b];
            let pc = pts[c];
            let convex = cross2(pb - pa, pc - pa) > 1e-9;
            if convex {
                let mut blocked = false;
                for &v in &idx {
                    // Compare by position, not index: hole-bridging inserts
                    // the slit endpoints twice (same point, two indices), and
                    // a slit twin sitting on the ear's own corner must not
                    // veto it — otherwise no slit ear is ever cut, the
                    // clipper stalls, and the fan fallback covers the hole.
                    let pv = pts[v as usize];
                    if pv == pa || pv == pb || pv == pc {
                        continue;
                    }
                    if point_in_triangle(pv, pa, pb, pc) {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    indices.extend_from_slice(&[
                        idx[(i + len - 1) % len],
                        idx[i],
                        idx[(i + 1) % len],
                    ]);
                    idx.remove(i);
                    ear_found = true;
                    break;
                }
            }
        }
        if !ear_found || guard <= 0 {
            // Best-effort fan (exact for convex, approximate for concave).
            for i in 1..idx.len().saturating_sub(1) {
                indices.extend_from_slice(&[idx[0], idx[i], idx[i + 1]]);
            }
            break;
        }
    }
    if idx.len() == 3 {
        indices.extend_from_slice(&[idx[0], idx[1], idx[2]]);
    }
    indices
}

/// Core fill: triangulate a set of CCW polygons with even-odd hole
/// semantics. Input polygons may be in any winding.
#[must_use]
pub fn fill_polys(polys_in: Vec<Vec<Vec2>>) -> Triangles {
    let mut polys: Vec<Vec<Vec2>> = polys_in
        .into_iter()
        .map(|mut p| {
            // Drop consecutive duplicates.
            let mut q: Vec<Vec2> = Vec::with_capacity(p.len());
            for pt in p.drain(..) {
                if q.last() != Some(&pt) {
                    q.push(pt);
                }
            }
            // Ensure CCW.
            if signed_area(&q) < 0.0 {
                q.reverse();
            }
            q
        })
        .filter(|p| p.len() >= 3 && signed_area(p).abs() > 1e-9)
        .collect();
    if polys.is_empty() {
        return Triangles::default();
    }

    let mut is_hole = vec![false; polys.len()];
    for i in 0..polys.len() {
        let mut depth = 0usize;
        for j in 0..polys.len() {
            if j == i {
                continue;
            }
            if point_in_poly(polys[i][0], &polys[j]) {
                depth += 1;
            }
        }
        is_hole[i] = depth % 2 == 1;
    }

    let mut used = vec![false; polys.len()];
    let mut triangles = Triangles::default();
    let mut voff = 0u32;
    for i in 0..polys.len() {
        if is_hole[i] || used[i] {
            continue;
        }
        used[i] = true;
        let mut poly = std::mem::take(&mut polys[i]);
        for j in 0..polys.len() {
            if j == i || !is_hole[j] || used[j] {
                continue;
            }
            if point_in_poly(polys[j][0], &poly) && merge_hole(&mut poly, &polys[j]) {
                used[j] = true;
            }
        }
        let idx = triangulate_polygon(&poly);
        triangles.vertices.extend_from_slice(&poly);
        for &k in &idx {
            triangles.indices.push(k + voff);
        }
        voff += poly.len() as u32;
    }
    triangles
}

/// Tessellate `path` for filling (even-odd, holes supported).
#[must_use]
pub fn fill(path: &Path, transform: &Affine, tolerance: f32) -> Triangles {
    let subs = flatten(path, tolerance);
    let polys: Vec<Vec<Vec2>> = subs
        .into_iter()
        .map(|s| {
            s.points
                .iter()
                .map(|p| transform.transform_point(*p))
                .collect()
        })
        .collect();
    fill_polys(polys)
}

fn line_intersect(p1: Vec2, d1: Vec2, p2: Vec2, d2: Vec2) -> Option<Vec2> {
    let den = cross2(d1, d2);
    if den.abs() < 1e-9 {
        return None;
    }
    let t = cross2(p2 - p1, d2) / den;
    Some(p1 + d1 * t)
}

struct OffsetLoop {
    left: Vec<Vec2>,
    right: Vec<Vec2>,
}

/// Build left/right offset loops of a polyline with miter/bevel joins.
fn offset_loops(pts: &[Vec2], closed: bool, width: f32, miter_limit: f32) -> Option<OffsetLoop> {
    let n = pts.len();
    if n < 2 {
        return None;
    }
    let hw = width * 0.5;
    let segs = if closed { n } else { n - 1 };

    // Segment directions and left normals.
    let mut dirs: Vec<Vec2> = Vec::with_capacity(segs);
    for i in 0..segs {
        let a = pts[i % n];
        let b = pts[(i + 1) % n];
        let d = b - a;
        if d.length_squared() < 1e-12 {
            return None;
        }
        dirs.push(d.normalize());
    }

    // Miter/bevel points per vertex.
    let mut left: Vec<Vec2> = Vec::with_capacity(n);
    let mut right: Vec<Vec2> = Vec::with_capacity(n);
    for (i, &p) in pts.iter().enumerate() {
        // Incoming/outgoing segment indices for vertex i. Endpoints of an
        // open path reuse their single adjacent segment (butt caps, handled
        // below); interior vertices join segments i-1 and i.
        let in_seg = if closed {
            (i + n - 1) % n
        } else {
            i.saturating_sub(1)
        };
        let out_seg = if closed { i % segs } else { i.min(n - 2) };
        let dp = dirs[in_seg];
        let dn = dirs[out_seg];
        let np = Vec2::new(-dp.y, dp.x);
        let nn = Vec2::new(-dn.y, dn.x);
        let lp = p + np * hw; // left offset point on incoming side
        let ln = p + nn * hw; // left offset point on outgoing side
        let rp = p - np * hw;
        let rn = p - nn * hw;

        if !closed && (i == 0 || i == n - 1) {
            // Butt caps: endpoints.
            if i == 0 {
                left.push(lp);
                right.push(rp);
            } else {
                left.push(ln);
                right.push(rn);
            }
            continue;
        }

        // Miter joins on both offset lines, clamped by miter limit
        // (CSS `stroke-linejoin: miter` semantics).
        if let Some(m) = line_intersect(lp, dp, ln, dn)
            && m.distance(p) <= hw * miter_limit
        {
            left.push(m);
        } else {
            left.push(lp);
            left.push(ln);
        }
        if let Some(m) = line_intersect(rp, dp, rn, dn)
            && m.distance(p) <= hw * miter_limit
        {
            right.push(m);
        } else {
            right.push(rp);
            right.push(rn);
        }
    }
    Some(OffsetLoop { left, right })
}

/// Tessellate `path` for stroking with width `width`.
#[must_use]
pub fn stroke(path: &Path, transform: &Affine, width: f32, tolerance: f32) -> Triangles {
    let w = width * transform.avg_scale();
    if w <= 1e-5 {
        return Triangles::default();
    }
    let subs = flatten(path, tolerance);
    let mut triangles = Triangles::default();
    let mut voff = 0u32;
    for sub in subs {
        let pts: Vec<Vec2> = sub
            .points
            .iter()
            .map(|p| transform.transform_point(*p))
            .collect();
        let Some(ol) = offset_loops(&pts, sub.closed, w, 4.0) else {
            continue;
        };
        let geom: Vec<Vec<Vec2>> = if sub.closed {
            // Two loops: outer + inner (hole).
            vec![ol.left, ol.right]
        } else {
            // Single loop: left forward, right backward (shared join points
            // live on exactly one side per turn, so no duplicates).
            let mut loop_: Vec<Vec2> = ol.left.clone();
            loop_.extend(ol.right.iter().rev());
            vec![loop_]
        };
        let t = fill_polys(geom);
        triangles.vertices.extend_from_slice(&t.vertices);
        for &k in &t.indices {
            triangles.indices.push(k + voff);
        }
        voff += t.vertices.len() as u32;
    }
    triangles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::path::Path;
    use std::f32::consts::PI;

    fn area_close(got: f32, want: f32, eps: f32) {
        assert!((got - want).abs() < eps, "area {got} !~ {want}");
    }

    fn append(dst: &mut Path, src: &Path) {
        for c in src.cmds() {
            match c {
                PathCmd::MoveTo(q) => dst.move_to(*q),
                PathCmd::LineTo(q) => dst.line_to(*q),
                PathCmd::CubicTo { c1, c2, end } => dst.cubic_to(*c1, *c2, *end),
                PathCmd::Close => dst.close(),
            }
        }
    }

    #[test]
    fn fill_square() {
        let p = Path::rect(0.0, 0.0, 10.0, 6.0);
        let t = fill(&p, &Affine::IDENTITY, DEFAULT_TOLERANCE);
        assert_eq!(t.indices.len() % 3, 0);
        assert!(!t.is_empty());
        area_close(t.area().abs(), 60.0, 0.01);
    }

    #[test]
    fn fill_triangle() {
        let p = Path::polygon(&[
            Vec2::new(0.0, 0.0),
            Vec2::new(8.0, 0.0),
            Vec2::new(4.0, 6.0),
        ]);
        let t = fill(&p, &Affine::IDENTITY, DEFAULT_TOLERANCE);
        area_close(t.area().abs(), 24.0, 0.01);
    }

    #[test]
    fn fill_ring_with_hole() {
        // Donut: outer circle + inner circle (hole) via even-odd.
        let mut p = Path::new();
        append(&mut p, &Path::circle(50.0, 50.0, 20.0, 32));
        append(&mut p, &Path::circle(50.0, 50.0, 8.0, 24));
        let t = fill(&p, &Affine::IDENTITY, DEFAULT_TOLERANCE);
        let want = PI * (20.0 * 20.0 - 8.0 * 8.0);
        area_close(t.area().abs(), want, want * 0.02);
        // Area alone can't catch a covered hole (overlaps cancel), so check
        // coverage too.
        assert_eq!(
            coverage(&t, Vec2::new(50.37, 51.13)),
            0,
            "hole must stay open"
        );
    }

    #[test]
    fn fill_rect_hole() {
        let mut p = Path::new();
        append(&mut p, &Path::rect(0.0, 0.0, 20.0, 20.0));
        append(&mut p, &Path::rect(5.0, 5.0, 10.0, 10.0));
        let t = fill(&p, &Affine::IDENTITY, DEFAULT_TOLERANCE);
        area_close(t.area().abs(), 400.0 - 100.0, 0.01);
        assert_eq!(
            coverage(&t, Vec2::new(10.37, 7.13)),
            0,
            "hole must stay open"
        );
        assert_eq!(coverage(&t, Vec2::new(2.37, 3.13)), 1, "frame covered once");
    }

    #[test]
    fn stroke_line_area() {
        let mut p = Path::new();
        p.move_to(Vec2::new(0.0, 0.0));
        p.line_to(Vec2::new(10.0, 0.0));
        let t = stroke(&p, &Affine::IDENTITY, 2.0, DEFAULT_TOLERANCE);
        // 10 long x 2 wide, butt caps.
        area_close(t.area().abs(), 20.0, 0.1);
    }

    #[test]
    fn stroke_closed_rect_area() {
        let p = Path::rect(0.0, 0.0, 10.0, 10.0);
        let t = stroke(&p, &Affine::IDENTITY, 2.0, DEFAULT_TOLERANCE);
        // Perimeter * width = 40 * 2 = 80 (miter corners exact for a rect).
        area_close(t.area().abs(), 80.0, 0.5);
    }

    /// Triangles covering `pt` (winding-independent count).
    fn coverage(t: &Triangles, pt: Vec2) -> usize {
        let mut n = 0;
        for tri in t.indices.as_chunks::<3>().0 {
            let (a, b, c) = (
                t.vertices[tri[0] as usize],
                t.vertices[tri[1] as usize],
                t.vertices[tri[2] as usize],
            );
            if point_in_triangle(pt, a, b, c) {
                n += 1;
            }
        }
        n
    }

    #[test]
    fn stroke_circle_is_a_ring() {
        // Regression (two chained bugs): the flattened circle used to end
        // where it began, and that closing duplicate self-intersected the
        // stroke offsets — and independently, slit-twin vertices vetoed
        // every hole ear, so the fan fallback covered the hole. A stroked
        // circle rendered as a disc (or vanished entirely, before that).
        let p = Path::circle(0.0, 0.0, 90.0, 48);
        let t = stroke(&p, &Affine::IDENTITY, 8.0, DEFAULT_TOLERANCE);
        area_close(t.area().abs(), 2.0 * PI * 90.0 * 8.0, 30.0);
        // Off-edge samples (boundary counts as covered, so avoid edges).
        assert_eq!(
            coverage(&t, Vec2::new(0.37, 1.13)),
            0,
            "hole must stay open"
        );
        assert_eq!(coverage(&t, Vec2::new(90.0, 1.7)), 1, "band covered once");
        assert_eq!(coverage(&t, Vec2::new(200.0, 3.1)), 0);
    }

    #[test]
    fn stroke_rounded_rect_is_visible() {
        // Regression: rounded_rect ends exactly where it began, and that
        // exact duplicate made offset_loops reject the path, so the stroke
        // silently vanished (area 0).
        let p = crate::geometry::rounded_rect(0.0, 0.0, 10.0, 10.0, 2.0);
        let t = stroke(&p, &Affine::IDENTITY, 2.0, DEFAULT_TOLERANCE);
        assert!(!t.is_empty(), "stroke must produce geometry");
        // Straights 4x(10-2*2)=24 plus corner arcs 2*pi*2: perimeter ~36.6.
        area_close(t.area().abs(), 36.57 * 2.0, 2.0);
    }

    #[test]
    fn flatten_cubic_midpoint() {
        let mut p = Path::new();
        p.move_to(Vec2::new(0.0, 0.0));
        p.cubic_to(
            Vec2::new(3.0, 10.0),
            Vec2::new(7.0, 10.0),
            Vec2::new(10.0, 0.0),
        );
        let subs = flatten(&p, 0.01);
        let pts = &subs[0].points;
        // Exact bezier midpoint at t=0.5: (5, 7.5).
        let best = pts.iter().min_by(|a, b| {
            a.distance_squared(Vec2::new(5.0, 7.5))
                .partial_cmp(&b.distance_squared(Vec2::new(5.0, 7.5)))
                .unwrap()
        });
        assert!(best.unwrap().distance(Vec2::new(5.0, 7.5)) < 0.05);
    }

    #[test]
    fn transform_scales_fill() {
        let p = Path::rect(0.0, 0.0, 10.0, 10.0);
        let t = fill(&p, &Affine::scale(2.0, 3.0), DEFAULT_TOLERANCE);
        // 20 x 30.
        area_close(t.area().abs(), 600.0, 0.1);
    }
}
