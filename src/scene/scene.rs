use std::sync::Arc;

use glam::Mat4;

use crate::geometry::{Affine, Path};

use super::camera::Camera;
use super::mesh::Mesh;

/// RGBA color, components in [0, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Default for Color {
    fn default() -> Self {
        Self::BLACK
    }
}

impl Color {
    #[must_use]
    pub const fn srgb(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    #[must_use]
    pub const fn opaque(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const BLACK: Color = Color::srgb(0.0, 0.0, 0.0, 1.0);
    pub const WHITE: Color = Color::srgb(1.0, 1.0, 1.0, 1.0);
    pub const RED: Color = Color::srgb(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: Color = Color::srgb(0.0, 1.0, 0.0, 1.0);
    pub const BLUE: Color = Color::srgb(0.0, 0.0, 1.0, 1.0);
    pub const TRANSPARENT: Color = Color::srgb(0.0, 0.0, 0.0, 0.0);

    #[must_use]
    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// How a 2D path is rendered.
#[derive(Clone, Copy, Debug)]
pub enum TwoDStyle {
    Fill(Color),
    Stroke { color: Color, width: f32 },
}

/// A 2D vector item.
#[derive(Clone, Debug)]
pub struct TwoDItem {
    pub path: Path,
    pub style: TwoDStyle,
    pub transform: Affine,
}

/// Per-3D-item lighting/ tinting.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub color: Color,
    /// Ambient light level (0..1).
    pub ambient: f32,
    /// Specular strength (0 = matte).
    pub specular: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self { color: Color::WHITE, ambient: 0.25, specular: 0.35 }
    }
}

/// A 3D mesh item.
#[derive(Clone, Debug)]
pub struct ThreeDItem {
    pub mesh: Arc<Mesh>,
    pub model: Mat4,
    pub material: Material,
}

/// One drawable in a scene.
#[derive(Clone, Debug)]
pub enum Item {
    TwoD(TwoDItem),
    ThreeD(ThreeDItem),
}

/// A frame's contents: background color, camera and ordered items.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub background: Color,
    pub camera: Camera,
    pub items: Vec<Item>,
}

impl Scene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            background: Color::BLACK,
            camera: Camera::default(),
            items: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn set_background(&mut self, color: Color) {
        self.background = color;
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    pub fn fill_2d(&mut self, path: Path, color: Color) {
        self.items.push(Item::TwoD(TwoDItem { path, style: TwoDStyle::Fill(color), transform: Affine::IDENTITY }));
    }

    pub fn stroke_2d(&mut self, path: Path, color: Color, width: f32) {
        self.items.push(Item::TwoD(TwoDItem { path, style: TwoDStyle::Stroke { color, width }, transform: Affine::IDENTITY }));
    }

    pub fn add_2d(&mut self, item: TwoDItem) {
        self.items.push(Item::TwoD(item));
    }

    pub fn draw_3d(&mut self, mesh: Arc<Mesh>, model: Mat4, material: Material) {
        self.items.push(Item::ThreeD(ThreeDItem { mesh, model, material }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_builds_items() {
        let mut s = Scene::new();
        s.fill_2d(Path::rect(0.0, 0.0, 10.0, 10.0), Color::RED);
        s.draw_3d(Arc::new(Mesh::cube(1.0)), Mat4::IDENTITY, Material::default());
        assert_eq!(s.items.len(), 2);
        s.clear();
        assert!(s.items.is_empty());
    }

    #[test]
    fn color_roundtrip() {
        let c = Color::srgb(0.5, 0.25, 0.75, 1.0);
        assert_eq!(c.to_array(), [0.5, 0.25, 0.75, 1.0]);
    }
}
