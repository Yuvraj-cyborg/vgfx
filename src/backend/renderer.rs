//! The high-level, cross-platform [`Renderer`] façade.
//!
//! Each frame the app calls `begin_frame`, records 2D/3D items, then calls
//! `present`, which submits the persistent [`Frame`] to the active [`Gpu`]. 3D items draw first (depth tested), 2D
//! overlays draw on top (alpha blended, no depth).
//!
use std::sync::Arc;

use crate::math::{Mat4, Vec3};

use crate::backend::common::{Draw, Frame, Gpu, V2D};
use crate::error::Result;
use crate::geometry::{Affine, DEFAULT_TOLERANCE, Path};
use crate::scene::{Camera, Color, Material, Mesh, Scene, TwoDItem, TwoDStyle};

#[cfg(target_vendor = "apple")]
use crate::backend::metal;
#[cfg(not(target_vendor = "apple"))]
use crate::backend::vulkan;

// Precomputed unit circle, rounded by rustc to f32. No frame-time trig.
#[allow(clippy::excessive_precision)]
const CIRCLE: [crate::math::Vec2; 48] = [
    crate::math::Vec2::new(1.000000000, 0.000000000),
    crate::math::Vec2::new(0.991444861, 0.130526192),
    crate::math::Vec2::new(0.965925826, 0.258819045),
    crate::math::Vec2::new(0.923879533, 0.382683432),
    crate::math::Vec2::new(0.866025404, 0.500000000),
    crate::math::Vec2::new(0.793353340, 0.608761429),
    crate::math::Vec2::new(
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ),
    crate::math::Vec2::new(0.608761429, 0.793353340),
    crate::math::Vec2::new(0.500000000, 0.866025404),
    crate::math::Vec2::new(0.382683432, 0.923879533),
    crate::math::Vec2::new(0.258819045, 0.965925826),
    crate::math::Vec2::new(0.130526192, 0.991444861),
    crate::math::Vec2::new(0.000000000, 1.000000000),
    crate::math::Vec2::new(-0.130526192, 0.991444861),
    crate::math::Vec2::new(-0.258819045, 0.965925826),
    crate::math::Vec2::new(-0.382683432, 0.923879533),
    crate::math::Vec2::new(-0.500000000, 0.866025404),
    crate::math::Vec2::new(-0.608761429, 0.793353340),
    crate::math::Vec2::new(
        -std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ),
    crate::math::Vec2::new(-0.793353340, 0.608761429),
    crate::math::Vec2::new(-0.866025404, 0.500000000),
    crate::math::Vec2::new(-0.923879533, 0.382683432),
    crate::math::Vec2::new(-0.965925826, 0.258819045),
    crate::math::Vec2::new(-0.991444861, 0.130526192),
    crate::math::Vec2::new(-1.000000000, 0.000000000),
    crate::math::Vec2::new(-0.991444861, -0.130526192),
    crate::math::Vec2::new(-0.965925826, -0.258819045),
    crate::math::Vec2::new(-0.923879533, -0.382683432),
    crate::math::Vec2::new(-0.866025404, -0.500000000),
    crate::math::Vec2::new(-0.793353340, -0.608761429),
    crate::math::Vec2::new(
        -std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ),
    crate::math::Vec2::new(-0.608761429, -0.793353340),
    crate::math::Vec2::new(-0.500000000, -0.866025404),
    crate::math::Vec2::new(-0.382683432, -0.923879533),
    crate::math::Vec2::new(-0.258819045, -0.965925826),
    crate::math::Vec2::new(-0.130526192, -0.991444861),
    crate::math::Vec2::new(-0.000000000, -1.000000000),
    crate::math::Vec2::new(0.130526192, -0.991444861),
    crate::math::Vec2::new(0.258819045, -0.965925826),
    crate::math::Vec2::new(0.382683432, -0.923879533),
    crate::math::Vec2::new(0.500000000, -0.866025404),
    crate::math::Vec2::new(0.608761429, -0.793353340),
    crate::math::Vec2::new(
        std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ),
    crate::math::Vec2::new(0.793353340, -0.608761429),
    crate::math::Vec2::new(0.866025404, -0.500000000),
    crate::math::Vec2::new(0.923879533, -0.382683432),
    crate::math::Vec2::new(0.965925826, -0.258819045),
    crate::math::Vec2::new(0.991444861, -0.130526192),
];
static NEXT_RENDERER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// Immutable, renderer-specific resident mesh. Drop renderer to free its GPU slots.
pub struct PreparedMesh {
    mesh: Arc<Mesh>,
    slot: usize,
    renderer: u64,
}
/// Immutable tessellated 2D geometry, allocated during preparation.
pub struct Prepared2D {
    vertices: Vec<V2D>,
    indices: Vec<u32>,
}

/// The high-level renderer. Keeps reserved frame storage across frames and
/// submits prepared geometry to the active GPU backend.
pub struct Renderer {
    backend: Box<dyn Gpu>,
    scene: Scene,
    frame: Frame,
    pending_error: Option<crate::Error>,
    id: u64,
}

impl Renderer {
    /// Create an offscreen renderer on the platform's native GPU backend.
    pub fn offscreen(width: u32, height: u32) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            let b = metal::MetalBackend::offscreen(width, height)?;
            Ok(Self::new_boxed(Box::new(b)))
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
        Self {
            scene: {
                let mut s = Scene::new();
                let (w, h) = backend.size();
                s.camera.set_aspect(w as f32 / h.max(1) as f32);
                s
            },
            backend,
            frame: Frame {
                clear: Color::srgb(0., 0., 0., 1.),
                draws: Vec::with_capacity(4096),
                vertices: Vec::with_capacity(65536),
                indices: Vec::with_capacity(262144),
            },
            pending_error: None,
            id: NEXT_RENDERER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn size(&self) -> (u32, u32) {
        self.backend.size()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.backend.resize(width, height);
        let (width, height) = self.backend.size();
        self.scene
            .camera
            .set_aspect(width as f32 / height.max(1) as f32);
    }

    /// Begin accumulating a frame (does not touch the GPU until `present`).
    pub fn begin_frame(&mut self) {
        self.frame.draws.clear();
        self.frame.vertices.clear();
        self.frame.indices.clear();
        self.pending_error = None;
    }

    pub fn set_background(&mut self, color: Color) {
        self.scene.set_background(color);
    }

    /// Camera parameters mirror [`Camera::look_at`]; kept as one flat call
    /// so the C ABI (`vgfx_set_camera_perspective`) maps 1:1.
    #[allow(clippy::too_many_arguments)]
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
        let points = [
            crate::math::Vec2::new(x, y),
            crate::math::Vec2::new(x + w, y),
            crate::math::Vec2::new(x + w, y + h),
            crate::math::Vec2::new(x, y + h),
        ];
        self.record_2d(&points, &[0, 1, 2, 0, 2, 3], color, &Affine::IDENTITY);
    }

    pub fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32, width: f32, color: Color) {
        if width <= 0.0 {
            return;
        }
        let hw = width * 0.5;
        let outer = [
            crate::math::Vec2::new(x - hw, y - hw),
            crate::math::Vec2::new(x + w + hw, y - hw),
            crate::math::Vec2::new(x + w + hw, y + h + hw),
            crate::math::Vec2::new(x - hw, y + h + hw),
        ];
        let inner = [
            crate::math::Vec2::new(x + hw, y + hw),
            crate::math::Vec2::new(x + w - hw, y + hw),
            crate::math::Vec2::new(x + w - hw, y + h - hw),
            crate::math::Vec2::new(x + hw, y + h - hw),
        ];
        if width >= w.min(h) {
            self.record_2d(&outer, &[0, 1, 2, 0, 2, 3], color, &Affine::IDENTITY);
        } else {
            self.record_ring(&outer, &inner, color);
        }
    }

    pub fn fill_circle(&mut self, cx: f32, cy: f32, r: f32, color: Color) {
        let mut points = [crate::math::Vec2::ZERO; 48];
        for (p, unit) in points.iter_mut().zip(CIRCLE) {
            *p = crate::math::Vec2::new(cx, cy) + unit * r;
        }
        self.record_fan(&points, color);
    }

    pub fn stroke_circle(&mut self, cx: f32, cy: f32, r: f32, width: f32, color: Color) {
        if width <= 0.0 {
            return;
        }
        let mut outer = [crate::math::Vec2::ZERO; 48];
        let mut inner = outer;
        let center = crate::math::Vec2::new(cx, cy);
        for i in 0..48 {
            outer[i] = center + CIRCLE[i] * (r + width * 0.5);
            inner[i] = center + CIRCLE[i] * (r - width * 0.5).max(0.0);
        }
        self.record_ring(&outer, &inner, color);
    }

    pub fn fill_rounded_rect(&mut self, x: f32, y: f32, w: f32, h: f32, corner: f32, color: Color) {
        let corner = corner.max(0.0).min(w * 0.5).min(h * 0.5);
        let centers = [
            crate::math::Vec2::new(x + w - corner, y + h - corner),
            crate::math::Vec2::new(x + corner, y + h - corner),
            crate::math::Vec2::new(x + corner, y + corner),
            crate::math::Vec2::new(x + w - corner, y + corner),
        ];
        let mut points = [crate::math::Vec2::ZERO; 52];
        for quadrant in 0..4 {
            for step in 0..=12 {
                points[quadrant * 13 + step] =
                    centers[quadrant] + CIRCLE[(quadrant * 12 + step) % 48] * corner;
            }
        }
        self.record_fan(&points, color);
    }

    fn record_fan(&mut self, points: &[crate::math::Vec2], color: Color) {
        // Built-in convex shapes have a fixed upper bound of 52 points.
        let mut indices = [0u32; 150];
        let mut count = 0;
        for i in 1..points.len() - 1 {
            indices[count..count + 3].copy_from_slice(&[0, i as u32, i as u32 + 1]);
            count += 3;
        }
        self.record_2d(points, &indices[..count], color, &Affine::IDENTITY);
    }

    fn record_ring(
        &mut self,
        outer: &[crate::math::Vec2],
        inner: &[crate::math::Vec2],
        color: Color,
    ) {
        let n = outer.len();
        let mut points = [crate::math::Vec2::ZERO; 96];
        let mut indices = [0u32; 288];
        points[..n].copy_from_slice(outer);
        points[n..n * 2].copy_from_slice(inner);
        for i in 0..n {
            let j = (i + 1) % n;
            indices[i * 6..i * 6 + 6].copy_from_slice(&[
                i as u32,
                j as u32,
                (n + i) as u32,
                j as u32,
                (n + j) as u32,
                (n + i) as u32,
            ]);
        }
        self.record_2d(
            &points[..n * 2],
            &indices[..n * 6],
            color,
            &Affine::IDENTITY,
        );
    }

    /// Fill an arbitrary [`Path`].
    pub fn fill_2d(&mut self, path: Path, style: impl Into<TwoDStyle>) {
        let item = TwoDItem {
            path,
            style: style.into(),
            transform: Affine::IDENTITY,
        };
        let (vertices, indices) = tessellate_2d(&item);
        self.record_vertices(&vertices, &indices);
    }

    pub fn stroke_2d(&mut self, path: Path, width: f32, color: Color) {
        self.fill_2d(path, TwoDStyle::Stroke { color, width });
    }

    /// Prepare arbitrary path geometry outside the frame loop. Reuse this
    /// immutable object through draw_prepared_2d without tessellation/allocation.
    pub fn prepare_2d(path: &Path, style: TwoDStyle) -> Prepared2D {
        let item = TwoDItem {
            path: path.clone(),
            style,
            transform: Affine::IDENTITY,
        };
        let (vertices, indices) = tessellate_2d(&item);
        Prepared2D { vertices, indices }
    }

    pub fn draw_prepared_2d(&mut self, geometry: &Prepared2D) {
        self.record_vertices(&geometry.vertices, &geometry.indices);
    }

    /// Explicit setup-time allocation. Call between frames.
    pub fn reserve_geometry(
        &mut self,
        draws: usize,
        vertices: usize,
        indices: usize,
    ) -> Result<()> {
        if vertices > u32::MAX as usize {
            return Err(crate::Error::Backend(
                "vertex capacity exceeds u32 index range".into(),
            ));
        }
        self.backend.reserve_geometry(vertices, indices)?;
        if draws > self.frame.draws.capacity() {
            self.frame
                .draws
                .reserve_exact(draws - self.frame.draws.len());
        }
        if vertices > self.frame.vertices.capacity() {
            self.frame
                .vertices
                .reserve_exact(vertices - self.frame.vertices.len());
        }
        if indices > self.frame.indices.capacity() {
            self.frame
                .indices
                .reserve_exact(indices - self.frame.indices.len());
        }
        Ok(())
    }

    fn fits(&mut self, vertices: usize, indices: usize) -> bool {
        if self.pending_error.is_some() {
            return false;
        }
        if self.frame.draws.len() == self.frame.draws.capacity()
            || vertices > self.frame.vertices.capacity() - self.frame.vertices.len()
            || indices > self.frame.indices.capacity() - self.frame.indices.len()
        {
            self.pending_error = Some(crate::Error::Backend(
                "frame capacity exceeded; reserve_geometry between frames".into(),
            ));
            return false;
        }
        true
    }

    fn record_batch(&mut self, start: usize, count: usize) {
        if let Some(Draw::TwoD { index_count, .. }) = self.frame.draws.last_mut() {
            *index_count += count;
        } else {
            self.frame.draws.push(Draw::TwoD {
                index_start: start,
                index_count: count,
            });
        }
    }

    fn record_vertices(&mut self, vertices: &[V2D], indices: &[u32]) {
        if !self.fits(vertices.len(), indices.len()) {
            return;
        }
        let base = self.frame.vertices.len() as u32;
        let start = self.frame.indices.len();
        self.frame.vertices.extend_from_slice(vertices);
        self.frame.indices.extend(indices.iter().map(|i| base + i));
        self.record_batch(start, indices.len());
    }

    fn record_2d(
        &mut self,
        points: &[crate::math::Vec2],
        indices: &[u32],
        color: Color,
        transform: &Affine,
    ) {
        if !self.fits(points.len(), indices.len()) {
            return;
        }
        let base = self.frame.vertices.len() as u32;
        let start = self.frame.indices.len();
        self.frame.vertices.extend(points.iter().map(|p| {
            let p = transform.transform_point(*p);
            V2D {
                x: p.x,
                y: p.y,
                r: color.r,
                g: color.g,
                b: color.b,
                a: color.a,
            }
        }));
        self.frame.indices.extend(indices.iter().map(|i| base + i));
        self.record_batch(start, indices.len());
    }

    /// Upload once. The returned renderer-specific slot makes draw submission
    /// independent of vertex count and avoids cache lookup in the frame loop.
    pub fn prepare_mesh(&mut self, mesh: Arc<Mesh>) -> Result<PreparedMesh> {
        let slot = self.backend.prepare_mesh(&mesh)?;
        Ok(PreparedMesh {
            mesh,
            slot,
            renderer: self.id,
        })
    }

    pub fn draw_prepared_mesh(&mut self, mesh: &PreparedMesh, model: Mat4, material: Material) {
        if mesh.renderer != self.id {
            self.pending_error = Some(crate::Error::Backend(
                "mesh belongs to another renderer".into(),
            ));
            return;
        }
        if !self.fits(0, 0) {
            return;
        }
        self.frame.draws.push(Draw::ThreeD {
            mesh: Arc::clone(&mesh.mesh),
            model,
            material,
            gpu_slot: Some(mesh.slot),
        });
    }

    /// Draw a 3D [`Mesh`] with a model transform and color tint.
    ///
    /// Takes ownership so the mesh is wrapped in an `Arc` exactly once
    /// (no per-frame clone). Pass `Mesh::cube(..)` / `Mesh::sphere(..)`
    /// directly, or use [`draw_mesh_shared`](Self::draw_mesh_shared) when
    /// several items share one allocation.
    pub fn draw_mesh(&mut self, mesh: Mesh, model: Mat4, color: Color) {
        let material = Material {
            color,
            ..Material::default()
        };
        self.draw_mesh_material(Arc::new(mesh), model, material);
    }

    /// Draw a shared 3D [`Mesh`] with a full [`Material`].
    pub fn draw_mesh_shared(&mut self, mesh: &Arc<Mesh>, model: Mat4, material: Material) {
        self.draw_mesh_material(Arc::clone(mesh), model, material);
    }

    /// Draw a 3D [`Mesh`] with a full [`Material`].
    pub fn draw_mesh_material(&mut self, mesh: Arc<Mesh>, model: Mat4, material: Material) {
        match self.prepare_mesh(mesh) {
            Ok(mesh) => self.draw_prepared_mesh(&mesh, model, material),
            Err(error) => self.pending_error = Some(error),
        }
    }

    /// Submit recorded geometry using reserved storage, then present.
    /// 3D items draw first (depth tested), 2D overlays on top.
    pub fn present(&mut self) -> Result<()> {
        if let Some(error) = &self.pending_error {
            return Err(error.clone());
        }
        self.frame.clear = self.scene.background;
        self.backend.render(&self.frame, &self.scene.camera)
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
        crate::png::save(path.as_ref(), w, h, &rgba)?;
        Ok(())
    }
}

/// Tessellate a 2D item into V2D vertices + indices (screen space).
fn tessellate_2d(t: &TwoDItem) -> (Vec<V2D>, Vec<u32>) {
    let tol = DEFAULT_TOLERANCE;
    let (tri, color) = match t.style {
        TwoDStyle::Fill(c) => (crate::geometry::fill(&t.path, &t.transform, tol), c),
        TwoDStyle::Stroke { color, width } => (
            crate::geometry::stroke(&t.path, &t.transform, width, tol),
            color,
        ),
    };
    if tri.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let verts = tri
        .vertices
        .iter()
        .map(|p| V2D {
            x: p.x,
            y: p.y,
            r: color.r,
            g: color.g,
            b: color.b,
            a: color.a,
        })
        .collect();
    (verts, tri.indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tessellates_rect() {
        let t = TwoDItem {
            path: Path::rect(0.0, 0.0, 10.0, 10.0),
            style: TwoDStyle::Fill(Color::WHITE),
            transform: Affine::IDENTITY,
        };
        let (verts, idx) = tessellate_2d(&t);
        assert!(!verts.is_empty());
        assert_eq!(idx.len() % 3, 0);
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };
    thread_local! {
        static TRACK:Cell<bool>=const{Cell::new(false)};
        static COUNT:Cell<usize>=const{Cell::new(0)};
    }
    struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let _ = TRACK.try_with(|t| {
                if t.get() {
                    COUNT.with(|c| c.set(c.get() + 1));
                }
            });
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            let _ = TRACK.try_with(|t| {
                if t.get() {
                    COUNT.with(|c| c.set(c.get() + 1));
                }
            });
            unsafe { System.realloc(p, l, n) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: Counting = Counting;
    struct Mock;
    impl Gpu for Mock {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn size(&self) -> (u32, u32) {
            (640, 480)
        }
        fn resize(&mut self, _: u32, _: u32) {}
        fn prepare_mesh(&mut self, _: &Arc<Mesh>) -> Result<usize> {
            Ok(0)
        }
        fn render(&mut self, f: &Frame, _: &Camera) -> Result<()> {
            assert!(f.indices.iter().all(|&i| (i as usize) < f.vertices.len()));
            Ok(())
        }
        fn readback(&mut self) -> Option<Vec<u8>> {
            None
        }
    }
    #[test]
    fn prepared_frames_do_not_allocate_for_small_or_large_meshes() {
        let mut r = Renderer::new_boxed(Box::new(Mock));
        let small = r.prepare_mesh(Arc::new(Mesh::cube(1.))).unwrap();
        let large = r
            .prepare_mesh(Arc::new(Mesh::sphere(1., 256, 128)))
            .unwrap();
        let path =
            Renderer::prepare_2d(&Path::rect(0., 0., 10., 10.), TwoDStyle::Fill(Color::WHITE));
        COUNT.with(|c| c.set(0));
        TRACK.with(|t| t.set(true));
        for i in 0..100 {
            r.begin_frame();
            r.fill_rect(1., 2., 3., 4., Color::WHITE);
            r.stroke_rect(1., 2., 30., 40., 3., Color::WHITE);
            r.fill_circle(50., 50., 20., Color::WHITE);
            r.stroke_circle(50., 50., 20., 3., Color::WHITE);
            r.fill_rounded_rect(0., 0., 100., 50., 10., Color::WHITE);
            r.draw_prepared_2d(&path);
            crate::text::draw_text(&mut r, 0., 0., 2., Color::WHITE, "FPS 60");
            r.draw_prepared_mesh(
                if i % 2 == 0 { &small } else { &large },
                Mat4::IDENTITY,
                Material::default(),
            );
            r.present().unwrap();
        }
        TRACK.with(|t| t.set(false));
        assert_eq!(
            COUNT.with(Cell::get),
            0,
            "frame loop allocated/reallocated Rust storage"
        );
    }
    #[test]
    fn capacity_error_is_atomic_and_persists_until_begin_frame() {
        let mut r = Renderer::new_boxed(Box::new(Mock));
        r.frame.vertices = Vec::with_capacity(4);
        r.frame.indices = Vec::with_capacity(6);
        r.fill_rect(0., 0., 1., 1., Color::WHITE);
        r.fill_rect(0., 0., 1., 1., Color::WHITE);
        assert_eq!(r.frame.vertices.len(), 4);
        assert_eq!(r.frame.indices.len(), 6);
        assert!(r.present().is_err());
        assert!(r.present().is_err());
        r.begin_frame();
        r.fill_rect(0., 0., 1., 1., Color::WHITE);
        assert!(r.present().is_ok());
    }
    #[test]
    fn prepared_mesh_cannot_cross_renderers() {
        let mut a = Renderer::new_boxed(Box::new(Mock));
        let mut b = Renderer::new_boxed(Box::new(Mock));
        let mesh = a.prepare_mesh(Arc::new(Mesh::cube(1.))).unwrap();
        b.draw_prepared_mesh(&mesh, Mat4::IDENTITY, Material::default());
        assert!(b.present().is_err());
    }
    #[test]
    fn circle_stroke_has_no_center_coverage() {
        let mut r = Renderer::new_boxed(Box::new(Mock));
        r.stroke_circle(0., 0., 10., 2., Color::WHITE);
        for tri in r.frame.indices.as_chunks::<3>().0 {
            let p: Vec<_> = tri
                .iter()
                .map(|&i| {
                    let v = r.frame.vertices[i as usize];
                    crate::math::Vec2::new(v.x, v.y)
                })
                .collect();
            let cross = |a: crate::math::Vec2, b: crate::math::Vec2| a.x * b.y - a.y * b.x;
            let d = [cross(p[0], p[1]), cross(p[1], p[2]), cross(p[2], p[0])];
            assert!(d.iter().any(|&v| v < 0.) && d.iter().any(|&v| v > 0.));
        }
    }
}
