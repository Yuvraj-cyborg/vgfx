use crate::math::Vec2;

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
///
/// Paths are plain value types: clone them freely, transform them with the
/// builder methods below (each returns a new `Path`).
#[derive(Clone, Debug, Default)]
pub struct Path {
    cmds: Vec<PathCmd>,
    last: Option<Vec2>,
}

impl Path {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(commands: usize) -> Self {
        Self {
            cmds: Vec::with_capacity(commands),
            last: None,
        }
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
        self.last = None;
    }

    pub fn move_to(&mut self, p: Vec2) {
        self.cmds.push(PathCmd::MoveTo(p));
        self.last = Some(p);
    }

    pub fn line_to(&mut self, p: Vec2) {
        self.cmds.push(PathCmd::LineTo(p));
        self.last = Some(p);
    }

    pub fn cubic_to(&mut self, c1: Vec2, c2: Vec2, end: Vec2) {
        self.cmds.push(PathCmd::CubicTo { c1, c2, end });
        self.last = Some(end);
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
    }

    /// Last explicit point (move/line/cubic endpoint), if any.
    #[must_use]
    pub fn last_point(&self) -> Option<Vec2> {
        self.last
    }

    /// Rectangle path (axis aligned).
    #[must_use]
    pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        let mut p = Self::with_capacity(5);
        p.move_to(Vec2::new(x, y));
        p.line_to(Vec2::new(x + w, y));
        p.line_to(Vec2::new(x + w, y + h));
        p.line_to(Vec2::new(x, y + h));
        p.close();
        p
    }

    /// Circle path approximated with `segments` cubic beziers
    /// (counter-clockwise, starting at angle 0).
    #[must_use]
    pub fn circle(cx: f32, cy: f32, r: f32, segments: u32) -> Self {
        let n = segments.max(4);
        let step = std::f32::consts::TAU / n as f32;
        // Cubic control offset for a near-circular arc of `step` radians.
        let offset = r * (4.0 / 3.0) * (step / 4.0).tan();
        let at = |a: f32| Vec2::new(cx + r * a.cos(), cy + r * a.sin());
        let tangent = |a: f32| Vec2::new(-a.sin(), a.cos());
        let mut p = Self::with_capacity(n as usize + 2);
        p.move_to(at(0.0));
        for i in 0..n {
            let a0 = i as f32 * step;
            let a1 = (i + 1) as f32 * step;
            p.cubic_to(
                at(a0) + tangent(a0) * offset,
                at(a1) - tangent(a1) * offset,
                at(a1),
            );
        }
        p.close();
        p
    }

    /// Regular polygon path (`sides` >= 3), circumradius `r`.
    #[must_use]
    pub fn regular_polygon(cx: f32, cy: f32, r: f32, sides: u32) -> Self {
        let n = sides.max(3);
        let mut p = Self::with_capacity(n as usize + 1);
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
    pub fn translate(mut self, dx: f32, dy: f32) -> Self {
        self.map_points(|p| p + Vec2::new(dx, dy));
        self
    }

    /// Scale owned storage in place; no second command allocation.
    #[must_use]
    pub fn scale_about(mut self, cx: f32, cy: f32, sx: f32, sy: f32) -> Self {
        self.map_points(|p| Vec2::new(cx + (p.x - cx) * sx, cy + (p.y - cy) * sy));
        self
    }

    fn map_points(&mut self, map: impl Fn(Vec2) -> Vec2) {
        for cmd in &mut self.cmds {
            match cmd {
                PathCmd::MoveTo(p) | PathCmd::LineTo(p) => *p = map(*p),
                PathCmd::CubicTo { c1, c2, end } => {
                    *c1 = map(*c1);
                    *c2 = map(*c2);
                    *end = map(*end);
                }
                PathCmd::Close => {}
            }
        }
        self.last = self.last.map(map);
    }

    /// Polygon path through `points` (closed).
    #[must_use]
    pub fn polygon(points: &[Vec2]) -> Self {
        let mut p = Self::with_capacity(points.len() + 1);
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
    fn path_records_commands() {
        let mut p = Path::new();
        assert!(p.is_empty());
        p.move_to(Vec2::new(0.0, 0.0));
        p.line_to(Vec2::new(1.0, 1.0));
        assert_eq!(p.cmds().len(), 2);
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
