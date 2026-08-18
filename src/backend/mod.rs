//! GPU backends behind a common trait, plus the high-level [`Renderer`]
//! façade with a tessellation cache.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::error::Result;
use crate::geometry::{Affine, Path, DEFAULT_TOLERANCE};
use crate::scene::{Camera, Color, Material, Mesh, Scene, TwoDItem, TwoDStyle};

#[cfg(target_vendor = "apple")]
pub mod metal;
#[cfg(not(target_vendor = "apple"))]
pub mod vulkan;

/// 2D vertex: screen-space position + RGBA color.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct V2D {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// 2D pipeline uniforms: viewport size in pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct U2D {
    pub vp: [f32; 4],
}

/// Uniforms for one 3D draw call (column-major matrices).
/// Layout matches MSL default and GLSL std430 (vec3 fields 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct U3D {
    pub mvp: [f32; 16],
    pub model: [f32; 16],
    pub cam_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub tint: [f32; 4],
    pub ambient: f32,
    pub specular: f32,
    pub _pad: f32,
}

/// Build 3D uniforms from camera + model + material.
#[must_use]
pub fn make_u3d(camera: &Camera, model: &Mat4, material: &Material) -> U3D {
    U3D {
        mvp: (camera.view_projection() * model).to_cols_array(),
        model: model.to_cols_array(),
        cam_pos: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
        light_dir: normalize3([0.4, 0.7, 0.6]),
        tint: material.color.to_array(),
        ambient: material.ambient,
        specular: material.specular,
        _pad: 0.0,
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 4] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-9 {
        [0.0, 0.0, 1.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l, 0.0]
    }
}

/// One draw operation in a [`Frame`].
pub enum Draw {
    /// 2D triangle batch (screen-space, alpha blended, no depth).
    TwoD {
        verts: Vec<V2D>,
        indices: Vec<u32>,
    },
    /// 3D indexed mesh (depth tested, lit).
    ThreeD {
        mesh: Arc<Mesh>,
        model: Mat4,
        material: Material,
    },
}

/// A complete frame: clear color + draw operations (3D first, then 2D).
pub struct Frame {
    pub clear: Color,
    pub draws: Vec<Draw>,
}

/// Internal GPU backend interface implemented by Metal and Vulkan.
/// Backends render a whole [`Frame`] in one call.
pub trait Gpu: Send {
    fn name(&self) -> &'static str;
    fn size(&self) -> (u32, u32);
    fn resize(&mut self, w: u32, h: u32);
    /// Render the frame (clear + all draws), commit, and present if windowed.
    fn render(&mut self, frame: &Frame, camera: &Camera) -> Result<()>;
    /// Read back the last offscreen frame as RGBA8 (None for windows).
    fn readback(&mut self) -> Option<Vec<u8>>;
}

/// The high-level, cross-platform renderer. Accumulates a [`Scene`] per
/// frame and uploads tessellated geometry to the active GPU backend.
pub struct Renderer {
    backend: Box<dyn Gpu>,
    scene: Scene,
    cache: HashMap<u64, (Vec<V2D>, Vec<u32>)>,
}

impl Renderer {
    /// Create an offscreen renderer on the platform's native GPU backend.
    #[must_use]
    pub fn offscreen(width: u32, height: u32) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            let b = metal::MetalBackend::offscreen(width, height)?;
            return Ok(Self::new_boxed(Box::new(b)));
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let b = vulkan::VulkanBackend::offscreen(width, height)?;
            Ok(Self::new_boxed(Box::new(b)))
        }
    }

    /// Renderer for a windowed (on-screen) Metal target.
    #[cfg(target_vendor = "apple")]
    pub fn window(width: u32, height: u32, layer: metal::MetalLayer) -> Result<Self> {
        let b = metal::MetalBackend::window(width, height, layer)?;
        Ok(Self::new_boxed(Box::new(b)))
    }

    fn new_boxed(backend: Box<dyn Gpu>) -> Self {
        Self { backend, scene: Scene::new(), cache: HashMap::new() }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn size(&self) -> (u32, u32) {
        self.backend.size()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.backend.resize(width, height);
        self.scene.camera.set_aspect(width as f32 / height.max(1) as f32);
    }

    /// Begin accumulating a frame (does not touch the GPU until `present`).
    pub fn begin_frame(&mut self) {
        self.scene.clear();
    }

    pub fn set_background(&mut self, color: Color) {
        self.scene.set_background(color);
    }

    pub fn set_camera_perspective(
        &mut self,
        fov_y_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
    ) {
        self.scene.set_camera(Camera::look_at(
            Vec3::from(eye),
            Vec3::from(target),
            Vec3::from(up),
            fov_y_deg,
            aspect,
            near,
            far,
        ));
    }

    // ---- 2D convenience builders ---------------------------------------

    pub fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.fill_2d(Path::rect(x, y, w, h), TwoDStyle::Fill(color));
    }

    pub fn stroke_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        width: f32,
        color: Color,
    ) {
        self.stroke_2d(Path::rect(x, y, w, h), width, color);
    }

    pub fn fill_circle(&mut self, cx: f32, cy: f32, r: f32, color: Color) {
        self.fill_2d(Path::circle(cx, cy, r, 48), TwoDStyle::Fill(color));
    }

    pub fn stroke_circle(
        &mut self,
        cx: f32,
        cy: f32,
        r: f32,
        width: f32,
        color: Color,
    ) {
        self.stroke_2d(Path::circle(cx, cy, r, 48), width, color);
    }

    pub fn fill_rounded_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        corner: f32,
        color: Color,
    ) {
        use crate::geometry::rounded_rect;
        self.fill_2d(rounded_rect(x, y, w, h, corner), TwoDStyle::Fill(color));
    }

    /// Fill an arbitrary [`Path`].
    pub fn fill_2d(&mut self, path: Path, style: impl Into<TwoDStyle>) {
        self.scene.add_2d(TwoDItem { path, style: style.into(), transform: Affine::IDENTITY });
    }

    /// Stroke an arbitrary [`Path`] with a width.
    pub fn stroke_2d(&mut self, path: Path, width: f32, color: Color) {
        self.scene
            .add_2d(TwoDItem { path, style: TwoDStyle::Stroke { color, width }, transform: Affine::IDENTITY });
    }

    /// Draw a 3D [`Mesh`] with a model transform and per-vertex color tint.
    pub fn draw_mesh(&mut self, mesh: &Mesh, model: Mat4, color: Color) {
        let material = Material { color, ..Material::default() };
        self.scene.draw_3d(Arc::new(mesh.clone()), model, material);
    }

    /// Draw a 3D [`Mesh`] with a full [`Material`].
    pub fn draw_mesh_material(
        &mut self,
        mesh: Arc<Mesh>,
        model: Mat4,
        material: Material,
    ) {
        self.scene.draw_3d(mesh, model, material);
    }

    /// Tessellate the scene and upload to the GPU, then present.
    /// 3D items draw first (depth tested), 2D overlays on top.
    pub fn present(&mut self) -> Result<()> {
        let cam = self.scene.camera;
        let twod: Vec<crate::scene::TwoDItem> = self
            .scene
            .items
            .iter()
            .filter_map(|i| match i {
                crate::scene::Item::TwoD(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        let mut draws: Vec<Draw> = Vec::with_capacity(self.scene.items.len());
        for item in &self.scene.items {
            if let crate::scene::Item::ThreeD(t) = item {
                draws.push(Draw::ThreeD { mesh: t.mesh.clone(), model: t.model, material: t.material });
            }
        }
        for t in &twod {
            let (verts, indices) = self.cached_2d(t);
            draws.push(Draw::TwoD { verts, indices });
        }
        let frame = Frame { clear: self.scene.background, draws };
        self.backend.render(&frame, &cam)
    }

    fn cached_2d(&mut self, t: &TwoDItem) -> (Vec<V2D>, Vec<u32>) {
        let key = hash_2d(t);
        match self.cache.get(&key) {
            Some(v) => (v.0.clone(), v.1.clone()),
            None => {
                let (verts, indices) = tessellate_2d(t);
                self.cache.insert(key, (verts.clone(), indices.clone()));
                (verts, indices)
            }
        }
    }

    /// Read back the last offscreen frame as RGBA8.
    pub fn read_pixels(&mut self) -> Option<Vec<u8>> {
        self.backend.readback()
    }

    /// Save the current (already presented) offscreen frame to a PNG.
    pub fn save_png(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let (w, h) = self.backend.size();
        let rgba = self
            .backend
            .readback()
            .ok_or_else(|| crate::error::Error::Backend("readback unavailable".into()))?;
        use image::{ImageBuffer, RgbaImage};
        let img: RgbaImage =
            ImageBuffer::from_raw(w, h, rgba).ok_or_else(|| crate::error::Error::Backend("bad pixel data".into()))?;
        img.save(path)?;
        Ok(())
    }
}

/// Tessellate a 2D item into V2D vertices + indices (screen space).
fn tessellate_2d(t: &TwoDItem) -> (Vec<V2D>, Vec<u32>) {
    let tol = DEFAULT_TOLERANCE;
    let (tri, color) = match t.style {
        TwoDStyle::Fill(c) => (crate::geometry::fill(&t.path, &t.transform, tol), c),
        TwoDStyle::Stroke { color, width } => (crate::geometry::stroke(&t.path, &t.transform, width, tol), color),
    };
    if tri.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let verts = tri
        .vertices
        .iter()
        .map(|p| V2D { x: p.x, y: p.y, r: color.r, g: color.g, b: color.b, a: color.a })
        .collect();
    (verts, tri.indices)
}

fn fnv(h: &mut u64, b: u8) {
    *h ^= b as u64;
    *h = h.wrapping_mul(0x100000001b3);
}

fn fnv_f32(h: &mut u64, v: f32) {
    for b in v.to_bits().to_le_bytes() {
        fnv(h, b);
    }
}

/// FNV-1a hash of a 2D item's geometry + style (tessellation cache key).
fn hash_2d(t: &TwoDItem) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    fnv(&mut h, t.path.version() as u8);
    for f in t.path.cmds() {
        let pts: Vec<glam::Vec2> = match f {
            crate::geometry::PathCmd::MoveTo(p) | crate::geometry::PathCmd::LineTo(p) => vec![*p],
            crate::geometry::PathCmd::CubicTo { c1, c2, end } => vec![*c1, *c2, *end],
            crate::geometry::PathCmd::Close => vec![],
        };
        for p in pts {
            fnv_f32(&mut h, p.x);
            fnv_f32(&mut h, p.y);
        }
        fnv(&mut h, match f {
            crate::geometry::PathCmd::MoveTo(_) => 1,
            crate::geometry::PathCmd::LineTo(_) => 2,
            crate::geometry::PathCmd::CubicTo { .. } => 3,
            crate::geometry::PathCmd::Close => 4,
        });
    }
    for row in t.transform.m {
        for v in row {
            fnv_f32(&mut h, v);
        }
    }
    match t.style {
        TwoDStyle::Fill(c) => {
            fnv(&mut h, 0);
            for v in c.to_array() {
                fnv_f32(&mut h, v);
            }
        }
        TwoDStyle::Stroke { color, width } => {
            fnv(&mut h, 1);
            for v in color.to_array() {
                fnv_f32(&mut h, v);
            }
            fnv_f32(&mut h, width);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_differs_for_different_paths() {
        let a = TwoDItem { path: Path::rect(0.0, 0.0, 10.0, 10.0), style: TwoDStyle::Fill(Color::RED), transform: Affine::IDENTITY };
        let b = TwoDItem { path: Path::rect(0.0, 0.0, 12.0, 10.0), style: TwoDStyle::Fill(Color::RED), transform: Affine::IDENTITY };
        assert_ne!(hash_2d(&a), hash_2d(&b));
    }

    #[test]
    fn hash_same_for_identical() {
        let a = TwoDItem { path: Path::rect(0.0, 0.0, 10.0, 10.0), style: TwoDStyle::Fill(Color::RED), transform: Affine::IDENTITY };
        let b = TwoDItem { path: Path::rect(0.0, 0.0, 10.0, 10.0), style: TwoDStyle::Fill(Color::RED), transform: Affine::IDENTITY };
        assert_eq!(hash_2d(&a), hash_2d(&b));
    }

    #[test]
    fn tessellates_rect() {
        let t = TwoDItem { path: Path::rect(0.0, 0.0, 10.0, 10.0), style: TwoDStyle::Fill(Color::WHITE), transform: Affine::IDENTITY };
        let (verts, idx) = tessellate_2d(&t);
        assert!(!verts.is_empty());
        assert_eq!(idx.len() % 3, 0);
    }
}
