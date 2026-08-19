//! Plain C ABI over the Rust API. Pair with `include/vgfx.h`.
//!
//! Conventions:
//! - Opaque handles (`vgfx_renderer`, `vgfx_mesh`).
//! - Functions returning a handle give `NULL` on error; functions returning
//!   `i32` give `-1` on error. Call [`vgfx_last_error`] for details.
//! - Matrices are 16 column-major `f32`; colors are 4 `f32` RGBA in [0,1].
//! - Panics never cross the ABI (they are converted to errors).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::vec::Vec;

use crate::error::Result;
use crate::scene::{Color, Material, Mesh};
use crate::Renderer;

type Opaque<T> = *mut T;

struct RendererHandle(Renderer);

// Thread-local error buffer. The pointer returned by `vgfx_last_error` is
// valid until the next vgfx call on the same thread.
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    // NUL-terminated copy of the backend name; valid until the next vgfx call
    // on this thread.
    static BACKEND_NAME: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn set_err(e: &dyn std::fmt::Display) {
    LAST_ERROR.with(|b| *b.borrow_mut() = format!("{e}\0").into_bytes());
}

fn clear_err() {
    LAST_ERROR.with(|b| *b.borrow_mut() = b"ok\0".to_vec());
}

fn guard<T>(f: impl FnOnce() -> Result<T>) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => {
            clear_err();
            Some(v)
        }
        Ok(Err(e)) => {
            set_err(&e);
            None
        }
        Err(_) => {
            set_err(&"panic in vgfx");
            None
        }
    }
}

/// The offscreen renderer handle.
#[repr(C)]
pub struct vgfx_renderer {
    _private: [u8; 0],
}

/// A mesh handle.
#[repr(C)]
pub struct vgfx_mesh {
    _private: [u8; 0],
}

impl vgfx_renderer {
    fn inner<'a>(&'a self) -> &'a RendererHandle {
        unsafe { &*(self as *const vgfx_renderer as *const RendererHandle) }
    }
    fn inner_mut<'a>(&'a mut self) -> &'a mut RendererHandle {
        unsafe { &mut *(self as *mut vgfx_renderer as *mut RendererHandle) }
    }
}

impl vgfx_mesh {
    fn inner<'a>(&'a self) -> &'a Arc<Mesh> {
        unsafe { &*(self as *const vgfx_mesh as *const Arc<Mesh>) }
    }
}

fn color(c: *const f32) -> Color {
    unsafe { Color::srgb(*c, *c.add(1), *c.add(2), *c.add(3)) }
}

/// Create an offscreen renderer on the platform's native GPU backend.
/// Returns NULL on failure (see `vgfx_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_offscreen(width: u32, height: u32) -> Opaque<vgfx_renderer> {
    guard(|| Ok(Renderer::offscreen(width, height)?))
        .map(|r| Box::into_raw(Box::new(RendererHandle(r))) as Opaque<vgfx_renderer>)
        .unwrap_or_else(|| std::ptr::null_mut::<vgfx_renderer>())
}

/// Destroy a renderer created by `vgfx_offscreen`.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_renderer_destroy(r: Opaque<vgfx_renderer>) {
    if !r.is_null() {
        drop(unsafe { Box::from_raw(r as *mut RendererHandle) });
    }
}

/// Last error message (thread-local, NUL-terminated). Valid until the next
/// vgfx call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_last_error() -> *const std::os::raw::c_char {
    static OK: &[u8] = b"ok\0";
    LAST_ERROR.with(|b| {
        let buf = b.borrow();
        if buf.is_empty() {
            OK.as_ptr() as *const std::os::raw::c_char
        } else {
            buf.as_ptr() as *const std::os::raw::c_char
        }
    })
}

/// Begin a new frame (clears accumulated draw operations).
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_begin_frame(r: Opaque<vgfx_renderer>) {
    if r.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.begin_frame() };
        Ok(())
    });
}

/// Set the frame's background color (RGBA, [0,1]).
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_set_background(r: Opaque<vgfx_renderer>, rgba: *const f32) {
    if r.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.set_background(color(rgba)) };
        Ok(())
    });
}

/// Set a right-handed perspective camera. Matrices/vectors are row lists of
/// f32: eye/target/up are 3 components each.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_set_camera_perspective(
    r: Opaque<vgfx_renderer>,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
    eye: *const f32,
    target: *const f32,
    up: *const f32,
) {
    if r.is_null() || eye.is_null() || target.is_null() || up.is_null() {
        return;
    }
    guard(|| {
        unsafe {
            let e = [eye.read(), eye.add(1).read(), eye.add(2).read()];
            let t = [target.read(), target.add(1).read(), target.add(2).read()];
            let u = [up.read(), up.add(1).read(), up.add(2).read()];
            (&mut *r).inner_mut().0.set_camera_perspective(fov_y_deg, aspect, near, far, e, t, u);
        }
        Ok(())
    });
}

// ---- 2D ------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_fill_rect(r: Opaque<vgfx_renderer>, x: f32, y: f32, w: f32, h: f32, rgba: *const f32) {
    if r.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.fill_rect(x, y, w, h, color(rgba)) };
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_fill_circle(r: Opaque<vgfx_renderer>, cx: f32, cy: f32, radius: f32, rgba: *const f32) {
    if r.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.fill_circle(cx, cy, radius, color(rgba)) };
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_fill_rounded_rect(
    r: Opaque<vgfx_renderer>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    corner: f32,
    rgba: *const f32,
) {
    if r.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.fill_rounded_rect(x, y, w, h, corner, color(rgba)) };
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_stroke_rect(
    r: Opaque<vgfx_renderer>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    width: f32,
    rgba: *const f32,
) {
    if r.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.stroke_rect(x, y, w, h, width, color(rgba)) };
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_stroke_circle(
    r: Opaque<vgfx_renderer>,
    cx: f32,
    cy: f32,
    radius: f32,
    width: f32,
    rgba: *const f32,
) {
    if r.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.stroke_circle(cx, cy, radius, width, color(rgba)) };
        Ok(())
    });
}

// ---- 3D ------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_draw_cube(r: Opaque<vgfx_renderer>, model: *const f32, rgba: *const f32) {
    if r.is_null() || model.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe {
            let m = glam::Mat4::from_cols_slice(&std::slice::from_raw_parts(model, 16));
            (&mut *r).inner_mut().0.draw_mesh(&Mesh::cube(1.0), m, color(rgba));
        }
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_draw_sphere(
    r: Opaque<vgfx_renderer>,
    radius: f32,
    width_segments: u32,
    height_segments: u32,
    model: *const f32,
    rgba: *const f32,
) {
    if r.is_null() || model.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe {
            let m = glam::Mat4::from_cols_slice(&std::slice::from_raw_parts(model, 16));
            (&mut *r).inner_mut().0.draw_mesh(&Mesh::sphere(radius, width_segments, height_segments), m, color(rgba));
        }
        Ok(())
    });
}

/// Build a mesh from flat arrays (xyz positions, xyz normals, rgba colors,
/// u32 indices). Returns NULL on invalid input.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_mesh_from_data(
    positions: *const f32,
    n_positions: u32,
    normals: *const f32,
    n_normals: u32,
    colors: *const f32,
    n_colors: u32,
    indices: *const u32,
    n_indices: u32,
) -> Opaque<vgfx_mesh> {
    guard(|| {
        if positions.is_null() || normals.is_null() || colors.is_null() || indices.is_null() {
            return Err(crate::Error::Backend("null mesh pointer".into()));
        }
        unsafe {
            let p = std::slice::from_raw_parts(positions, n_positions as usize);
            let n = std::slice::from_raw_parts(normals, n_normals as usize);
            let c = std::slice::from_raw_parts(colors, n_colors as usize);
            let i = std::slice::from_raw_parts(indices, n_indices as usize);
            Mesh::from_data(p, n, c, i)
                .ok_or_else(|| crate::Error::Backend("invalid mesh data".into()))
        }
    })
    .map(|m| Box::into_raw(Box::new(Arc::new(m))) as Opaque<vgfx_mesh>)
    .unwrap_or_else(|| std::ptr::null_mut::<vgfx_mesh>())
}

/// Destroy a mesh created by `vgfx_mesh_from_data`.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_mesh_destroy(m: Opaque<vgfx_mesh>) {
    if !m.is_null() {
        drop(unsafe { Box::from_raw(m as *mut Arc<Mesh>) });
    }
}

/// Draw a mesh with a model transform and color.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_draw_mesh(r: Opaque<vgfx_renderer>, m: Opaque<vgfx_mesh>, model: *const f32, rgba: *const f32) {
    if r.is_null() || m.is_null() || model.is_null() || rgba.is_null() {
        return;
    }
    guard(|| {
        unsafe {
            let mat = glam::Mat4::from_cols_slice(&std::slice::from_raw_parts(model, 16));
            let material = Material { color: color(rgba), ..Material::default() };
            (&mut *r).inner_mut().0.draw_mesh_material(Arc::clone(&(&*m).inner()), mat, material);
        }
        Ok(())
    });
}

// ---- output ----------------------------------------------------------------

/// Render the accumulated frame. Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_present(r: Opaque<vgfx_renderer>) -> i32 {
    if r.is_null() {
        set_err(&"null renderer");
        return -1;
    }
    match guard(|| unsafe { (&mut *r).inner_mut().0.present() }) {
        Some(()) => 0,
        _ => -1,
    }
}

/// Save the last offscreen frame as a PNG. Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_save_png(r: Opaque<vgfx_renderer>, path: *const std::os::raw::c_char) -> i32 {
    if r.is_null() || path.is_null() {
        set_err(&"null renderer or path");
        return -1;
    }
    let path = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_err(&"path is not valid UTF-8");
            return -1;
        }
    };
    match guard(|| unsafe { (&mut *r).inner_mut().0.save_png(path) }) {
        Some(()) => 0,
        _ => -1,
    }
}

/// Backend name, e.g. "metal". NUL-terminated; valid until the next vgfx call
/// on this thread. Do not free.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_backend_name(r: Opaque<vgfx_renderer>) -> *const std::os::raw::c_char {
    if r.is_null() {
        return std::ptr::null();
    }
    match guard(|| unsafe { Ok((&*r).inner().0.backend_name()) }) {
        Some(name) => BACKEND_NAME.with(|b| {
            let mut buf = b.borrow_mut();
            buf.clear();
            buf.extend_from_slice(name.as_bytes());
            buf.push(0);
            buf.as_ptr() as *const std::os::raw::c_char
        }),
        _ => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_width(r: Opaque<vgfx_renderer>) -> u32 {
    if r.is_null() {
        return 0;
    }
    unsafe { (&*r).inner().0.size().0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn vgfx_height(r: Opaque<vgfx_renderer>) -> u32 {
    if r.is_null() {
        return 0;
    }
    unsafe { (&*r).inner().0.size().1 }
}

/// Resize the render target.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_resize(r: Opaque<vgfx_renderer>, width: u32, height: u32) {
    if r.is_null() {
        return;
    }
    guard(|| {
        unsafe { (&mut *r).inner_mut().0.resize(width, height) };
        Ok(())
    });
}
