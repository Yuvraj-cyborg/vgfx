/// # vgfx
///
/// A cross-language, cross-platform API for rendering 2D and 3D vector
/// graphics by talking directly to the GPU.
///
/// ## Backends
///
/// | Platform        | GPU API | Crate   |
/// |-----------------|---------|---------|
/// | macOS / iOS     | Metal   | `metal` |
/// | Linux / Windows | Vulkan  | `ash`   |
///
/// On Apple hardware the GPU (AMD/Intel/Apple-silicon) is used via Metal.
/// Note: the Apple NPU (ANE) is a machine-learning accelerator and cannot
/// render graphics; all rendering here runs on the GPU.
///
/// ## Cross-language
///
/// The whole API is exposed through a plain C ABI (`vgfx::ffi`, header in
/// `include/vgfx.h`), so it can be called from C, C++, Python (ctypes),
/// Go (cgo), Zig, Swift, etc.
///
/// ## Quick start (Rust)
///
/// ```no_run
/// use vgfx::prelude::*;
///
/// // Offscreen 1280x720 target on the platform's native GPU backend.
/// let mut renderer = Renderer::offscreen(1280, 720).unwrap();
///
/// renderer.begin_frame();
/// renderer.set_background(Color::srgb(0.08, 0.09, 0.12, 1.0));
///
/// // 2D vector graphics
/// renderer.fill_rect(40.0, 40.0, 200.0, 120.0, Color::srgb(0.2, 0.8, 0.6, 1.0));
/// renderer.fill_circle(400.0, 200.0, 80.0, Color::srgb(0.9, 0.4, 0.2, 1.0));
///
/// // 3D vector/mesh graphics
/// renderer.set_camera_perspective(60.0, 16.0 / 9.0, 0.1, 100.0,
///     [0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
/// let cube = Mesh::cube(1.0);
/// let model = glam::Mat4::from_rotation_z(0.6);
/// renderer.draw_mesh(&cube, model, Color::srgb(0.3, 0.5, 0.9, 1.0));
///
/// renderer.present().unwrap();
/// renderer.save_png("out.png").unwrap();
/// # drop(renderer);
/// ```
pub mod backend;
pub mod error;
pub mod ffi;
pub mod geometry;
pub mod scene;

pub use backend::Renderer;
pub use error::{Error, Result};
pub use scene::{Camera, Color, Material, Mesh, Scene};

/// Commonly used types.
pub mod prelude {
    pub use crate::geometry::{Affine, Path};
    pub use crate::{
        backend::Renderer,
        error::{Error, Result},
        scene::{Camera, Color, Item, Material, Mesh, Scene, TwoDItem, TwoDStyle},
    };
}
