//! Shared GPU types: vertex/uniform layouts, frames, and the [`Gpu`] trait.
//!
//! Layout contract: [`V2D`] / [`Vertex3D`](crate::scene::Vertex3D) and the
//! uniform structs must match the Metal (MSL) struct layouts byte for byte.
//! MSL pads `float3` to 16 bytes, so the Rust side stores positions, camera
//! positions and light directions as 4-float vectors. See `metal.rs` and the
//! `uniform_layout_*` tests there.

use std::sync::Arc;

use crate::error::Result;
use crate::scene::{Camera, Color, Material, Mesh};

/// 2D vertex: screen-space position (y-down pixels) + RGBA color.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct V2D {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// 2D pipeline uniforms: viewport size in pixels (`vp.x` = width,
/// `vp.y` = height; `vp.zw` unused padding).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct U2D {
    pub vp: [f32; 4],
}

/// Uniforms for one 3D draw call (column-major matrices).
///
/// `cam_pos` / `light_dir` are 4 floats because MSL aligns `float3` to
/// 16 bytes; the `w` lane is unused. Two trailing floats explicitly initialize
/// the MSL struct's 192-byte extent, including its tail padding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct U3D {
    pub mvp: [f32; 16],
    pub model: [f32; 16],
    pub cam_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub tint: [f32; 4],
    pub ambient: f32,
    pub specular: f32,
    pub _pad: [f32; 2],
}

/// Build 3D uniforms from camera + model + material.
#[must_use]
pub fn make_u3d(camera: &Camera, model: &crate::math::Mat4, material: &Material) -> U3D {
    make_u3d_with_projection(camera, &camera.view_projection(), model, material)
}

pub(crate) fn make_u3d_with_projection(
    camera: &Camera,
    projection: &crate::math::Mat4,
    model: &crate::math::Mat4,
    material: &Material,
) -> U3D {
    U3D {
        mvp: (*projection * model).to_cols_array(),
        model: model.to_cols_array(),
        cam_pos: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
        light_dir: [0.39801487, 0.696526, 0.59702235, 0.0],
        tint: material.color.to_array(),
        ambient: material.ambient,
        specular: material.specular,
        _pad: [0.0; 2],
    }
}

/// One draw operation in a [`Frame`].
pub enum Draw {
    /// 2D triangle batch (screen-space, alpha blended, no depth).
    TwoD {
        index_start: usize,
        index_count: usize,
    },
    /// 3D indexed mesh (depth tested, lit).
    ThreeD {
        mesh: Arc<Mesh>,
        model: crate::math::Mat4,
        material: Material,
        gpu_slot: Option<usize>,
    },
}

/// A complete frame: clear color + draw operations (3D first, then 2D).
pub struct Frame {
    pub clear: Color,
    pub draws: Vec<Draw>,
    pub vertices: Vec<V2D>,
    pub indices: Vec<u32>,
}

/// Internal GPU backend interface implemented by Metal and Vulkan.
/// Backends render a whole [`Frame`] in one call.
pub trait Gpu: Send {
    fn prepare_mesh(&mut self, _mesh: &Arc<Mesh>) -> Result<usize> {
        Err(crate::Error::Backend("mesh preparation unavailable".into()))
    }
    fn reserve_geometry(&mut self, _vertices: usize, _indices: usize) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &'static str;
    fn size(&self) -> (u32, u32);
    fn resize(&mut self, w: u32, h: u32);
    /// Render the frame (clear + all draws), commit, and present if windowed.
    fn render(&mut self, frame: &Frame, camera: &Camera) -> Result<()>;
    /// Read back the last offscreen frame as RGBA8 (None for windows).
    fn readback(&mut self) -> Option<Vec<u8>>;
}
