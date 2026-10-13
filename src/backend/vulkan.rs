//! Vulkan backend for Linux / Windows (native implementation pending).
//!
//! Not implemented yet: this module only provides a compiling stub so the
//! crate builds on non-Apple targets. `VulkanBackend::offscreen` and every
//! `Gpu` method return an explicit "not implemented" error until the real
//! backend lands (see `src/backend/metal.rs` for the reference design:
//! frame submission, 2D/3D pipelines, readback for PNG export).

use crate::backend::common::{Frame, Gpu};
use crate::error::{Error, Result};
use crate::scene::Camera;

/// Placeholder Vulkan backend. Never constructed successfully yet.
pub struct VulkanBackend {
    width: u32,
    height: u32,
}

impl VulkanBackend {
    /// Always fails until the backend is implemented.
    pub fn offscreen(_width: u32, _height: u32) -> Result<Self> {
        Err(Error::Backend(
            "vulkan backend is not implemented yet; use macOS/iOS (Metal) for now".into(),
        ))
    }
}

impl Gpu for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan (unimplemented)"
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }

    fn render(&mut self, _frame: &Frame, _camera: &Camera) -> Result<()> {
        Err(Error::Backend(
            "vulkan backend is not implemented yet".into(),
        ))
    }

    fn readback(&mut self) -> Option<Vec<u8>> {
        None
    }
}
