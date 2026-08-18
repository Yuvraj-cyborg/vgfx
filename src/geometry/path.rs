use glam::Vec2;

/// A single vector path command.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCmd {
    MoveTo(Vec2),
    LineTo(Vec2),
    CubicTo { c1: Vec2, c2: Vec2, end: Vec2 },
    Close,
}

/// A 2D vector path (sequence of commands), the core primitive of the
/// 2D API. Paths are tessellated on the GPU backend side into triangles.
#[derive(Clone, Debug, Default)]
pub struct Path {
    cmds: Vec<PathCmd>,
    version: u64,
}

impl Path {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Monotonic version, bumped on every mutation (used to invalidate
    /// tessellation caches).
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    #[must_use]
    pub fn cmds(&self) -> &[PathCmd] {
        &self.cmds
    }

    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    pub fn clear(&mut self) {
        self.cmds.clear();
        self.version += 1;
    }

    pub fn move_to(&mut self, p: Vec2) {
        self.cmds.push(PathCmd::MoveTo(p));
        self.version += 1;
    }

    pub fn line_to(&mut self, p: Vec2) {
        self.cmds.push(PathCmd::LineTo(p));
        self.version += 1;
    }

    pub fn cubic_to(&mut self, c1: Vec2, c2: Vec2, end: Vec2) {
        self.cmds.push(PathCmd::CubicTo { c1, c2, end });
        self.version += 1;
    }

    /// Quadratic bezier, converted to a cubic.
    pub fn quad_to(&mut self, ctrl: Vec2, end: Vec2) {
        let start = self.last_point().unwrap_or(Vec2::ZERO);
        let c1 = start + (ctrl - start) * (2.0 / 3.0);
        let c2 = end + (ctrl - end) * (2.0 / 3.0);
        self.cubic_to(c1, c2, end);
    }

    pub fn close(&mut self) {
        self.cmds.push(PathCmd::Close);
        self.version += 1;
    }

    /// Last explicit point (move/line/cubic endpoint), if any.
    #[must_use]
    pub fn last_point(&self) -> Option<Vec2> {
        self.cmds.iter().rev().find_map(|c| match c {
            PathCmd::MoveTo(p) | PathCmd::LineTo(p) => Some(*p),
            PathCmd::CubicTo { end, .. } => Some(*end),
            PathCmd::Close => None,
        })
    }

    /// Rectangle path (axis aligned).
    #[must_use]
    pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        let mut p = Self::new();
        p.move_to(Vec2::new(x, y));
        p.line_to(Vec2::new(x + w, y));
        p.line_to(Vec2::new(x + w, y + h));
        p.line_to(Vec2::new(x, y + h));
        p.close();
        p
    }

    /// Circle path approximated with `segments` cubic beziers.
    #[must_use]
    pub fn circle(cx: f32, cy: f32, r: f32, segments: u32) -> Self {
        let n = segments.max(4);
        let step = std::f32::consts::TAU / n as f32;
        let offset = r * (4.0 / 3.0) * (step / 4.0).tan();
        let mut p = Self::new();
        p.move_to(Vec2::new(cx + r, cy));
        for i in 1..=n {
            let a0 = -(i as f32) * step;
            let a1 = a0 + step;
            let p0 = Vec2::new(cx + r * a0.cos(), cy + r * a0.sin());
            let p1 = Vec2::new(cx + r * a1.cos(), cy + r * a1.sin());
            let t0 = Vec2::new(-a0.sin(), a0.cos());
            let t1 = Vec2::new(-a1.sin(), a1.cos());
            p.cubic_to(p0 + t0 * offset, p1 - t1 * offset, p1);
        }
        p.close();
        p
    }

    /// Regular polygon path (`sides` >= 3), circumradius `r`.
    #[must_use]
    pub fn regular_polygon(cx: f32, cy: f32, r: f32, sides: u32) -> Self {
        let n = sides.max(3);
        let mut p = Self::new();
        for i in 0..n {
            let a = -std::f32::consts::PI / 2.0 + i as f32 * std::f32::consts::TAU / n as f32;
            let pt = Vec2::new(cx + r * a.cos(), cy + r * a.sin());
            if i == 0 {
                p.move_to(pt);
            } else {
                p.line_to(pt);
            }
        }
        p.close();
        p
    }

    /// Translate the path.
    #[must_use]
    pub fn translate(self, dx: f32, dy: f32) -> Self {
        let mut out = Self::new();
        for c in self.cmds() {
            match c {
                PathCmd::MoveTo(p) => out.move_to(*p + Vec2::new(dx, dy)),
                PathCmd::LineTo(p) => out.line_to(*p + Vec2::new(dx, dy)),
                PathCmd::CubicTo { c1, c2, end } => out.cubic_to(*c1 + Vec2::new(dx, dy), *c2 + Vec2::new(dx, dy), *end + Vec2::new(dx, dy)),
                PathCmd::Close => out.close(),
            }
        }
        out
    }

    /// Scale the path about a center point (used by shape builders).
    #[must_use]
    pub fn scale_about(self, cx: f32, cy: f32, sx: f32, sy: f32) -> Self {
        let s = |p: Vec2| Vec2::new(cx + (p.x - cx) * sx, cy + (p.y - cy) * sy);
        let mut out = Self::new();
        for c in self.cmds() {
            match c {
                PathCmd::MoveTo(p) => out.move_to(s(*p)),
                PathCmd::LineTo(p) => out.line_to(s(*p)),
                PathCmd::CubicTo { c1, c2, end } => out.cubic_to(s(*c1), s(*c2), s(*end)),
                PathCmd::Close => out.close(),
            }
        }
        out
    }

    /// Polygon path through `points` (closed).
    #[must_use]
    pub fn polygon(points: &[Vec2]) -> Self {
        let mut p = Self::new();
        if let Some(&first) = points.first() {
            p.move_to(first);
            for &pt in &points[1..] {
                p.line_to(pt);
            }
            p.close();
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_bumps_on_mutation() {
        let mut p = Path::new();
        let v0 = p.version();
        p.move_to(Vec2::new(0.0, 0.0));
        p.line_to(Vec2::new(1.0, 1.0));
        assert!(p.version() > v0);
    }

    #[test]
    fn circle_has_expected_commands() {
        let p = Path::circle(0.0, 0.0, 10.0, 8);
        assert_eq!(p.cmds().len(), 1 + 8 + 1); // move + 8 cubics + close
    }

    #[test]
    fn quad_to_converts_to_cubic() {
        let mut p = Path::new();
        p.move_to(Vec2::new(0.0, 0.0));
        p.quad_to(Vec2::new(5.0, 5.0), Vec2::new(10.0, 0.0));
        assert!(matches!(p.cmds()[1], PathCmd::CubicTo { .. }));
    }
}
