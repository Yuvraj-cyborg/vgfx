//! Convenience shape builders. Each returns a [`Path`] which can be filled
//! or stroked by the renderer.

use crate::geometry::path::Path;
use glam::Vec2;

/// Axis-aligned rectangle.
#[must_use]
pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Path {
    Path::rect(x, y, w, h)
}

/// Circle (cubic bezier approximation).
#[must_use]
pub fn circle(cx: f32, cy: f32, radius: f32, segments: u32) -> Path {
    Path::circle(cx, cy, radius, segments)
}

/// Ellipse (cubic bezier approximation).
#[must_use]
pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32, segments: u32) -> Path {
    Path::circle(0.0, 0.0, 1.0, segments.max(4)).scale_about(0.0, 0.0, rx, ry).translate(cx, cy)
}

/// Rounded rectangle.
#[must_use]
pub fn rounded_rect(x: f32, y: f32, w: f32, h: f32, corner: f32) -> Path {
    const K: f32 = 0.5522847498;
    let c = corner.min(w * 0.5).min(h * 0.5).max(0.0);
    let mut p = Path::new();
    p.move_to(Vec2::new(x + c, y));
    p.line_to(Vec2::new(x + w - c, y));
    p.cubic_to(
        Vec2::new(x + w - c + c * K, y),
        Vec2::new(x + w, y + c - c * K),
        Vec2::new(x + w, y + c),
    );
    p.line_to(Vec2::new(x + w, y + h - c));
    p.cubic_to(
        Vec2::new(x + w, y + h - c + c * K),
        Vec2::new(x + w - c + c * K, y + h),
        Vec2::new(x + w - c, y + h),
    );
    p.line_to(Vec2::new(x + c, y + h));
    p.cubic_to(
        Vec2::new(x + c - c * K, y + h),
        Vec2::new(x, y + h - c + c * K),
        Vec2::new(x, y + h - c),
    );
    p.line_to(Vec2::new(x, y + c));
    p.cubic_to(
        Vec2::new(x, y + c - c * K),
        Vec2::new(x + c - c * K, y),
        Vec2::new(x + c, y),
    );
    p.close();
    p
}

/// Regular polygon (triangle, square, pentagon, ...).
#[must_use]
pub fn regular_polygon(cx: f32, cy: f32, radius: f32, sides: u32) -> Path {
    Path::regular_polygon(cx, cy, radius, sides)
}

/// Arbitrary polygon through `points` (closed).
#[must_use]
pub fn polygon(points: &[Vec2]) -> Path {
    Path::polygon(points)
}
