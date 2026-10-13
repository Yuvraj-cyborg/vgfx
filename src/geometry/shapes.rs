//! Shape builders that have no direct `Path` constructor. For rectangles, circles and polygons use the `Path::rect`,
//! `Path::circle`, `Path::regular_polygon` and `Path::polygon` methods
//! directly — this module only holds shapes that need extra math.

use crate::geometry::path::Path;
use crate::math::Vec2;

/// Ellipse (unit circle scaled by `(rx, ry)`, then translated).
#[must_use]
pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32, segments: u32) -> Path {
    Path::circle(0.0, 0.0, 1.0, segments.max(4))
        .scale_about(0.0, 0.0, rx, ry)
        .translate(cx, cy)
}

/// Rounded rectangle with corner radius clamped to half the smaller side.
#[must_use]
pub fn rounded_rect(x: f32, y: f32, w: f32, h: f32, corner: f32) -> Path {
    const K: f32 = 0.552_284_8; // cubic control offset for a quarter circle
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
