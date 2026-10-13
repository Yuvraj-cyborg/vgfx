//! Plain C ABI over the Rust API. Pair with `include/vgfx.h`.
//!
//! Conventions:
//! - Opaque handles (`vgfx_renderer`, `vgfx_mesh`).
//! - Functions returning a handle give `NULL` on error; functions returning
//!   `i32` give `-1` on error. Call [`vgfx_last_error`] for details.
//! - Matrices are 16 column-major `f32`; colors are 4 `f32` RGBA in \[0,1\].
//! - Panics never cross the ABI (they are converted to errors).
//!
//! Safety: every function that takes a pointer is `unsafe extern "C"` —
//! callers must pass valid, non-null pointers (null handles are tolerated
//! and treated as no-ops / errors, as documented per function). This matches
//! edition-2024 idioms and silences `clippy::not_unsafe_ptr_arg_deref`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::Renderer;
use crate::error::Result;
use crate::scene::{Color, Material, Mesh};

type Opaque<T> = *mut T;

struct RendererHandle(Renderer);

// Thread-local error buffer. The pointer returned by `vgfx_last_error` is
// valid until the next vgfx call on the same thread.
struct ErrorBuffer {
    bytes: [u8; 1024],
    len: usize,
}
impl std::fmt::Write for ErrorBuffer {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let count = text.len().min(self.bytes.len() - 1 - self.len);
        for &byte in &text.as_bytes()[..count] {
            self.bytes[self.len] = if byte == 0 { b'?' } else { byte };
            self.len += 1;
        }
        self.bytes[self.len] = 0;
        Ok(())
    }
}
thread_local! {
    static LAST_ERROR: std::cell::RefCell<ErrorBuffer> = const {std::cell::RefCell::new(ErrorBuffer{bytes:[0;1024],len:0})};
}
fn set_err(e: &dyn std::fmt::Display) {
    use std::fmt::Write;
    LAST_ERROR.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.len = 0;
        buffer.bytes[0] = 0;
        let _ = write!(buffer, "{e}");
    });
}
fn clear_err() {
    LAST_ERROR.with(|b| {
        let mut b = b.borrow_mut();
        b.len = 0;
        b.bytes[0] = 0;
    });
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

// ---- unsafe pointer readers (null → None) ---------------------------------

/// # Safety
/// `r` must be null or a live renderer handle.
unsafe fn renderer_mut(r: Opaque<vgfx_renderer>) -> Option<&'static mut Renderer> {
    unsafe { (r as *mut RendererHandle).as_mut().map(|h| &mut h.0) }
}

/// # Safety
/// `r` must be null or a live renderer handle.
unsafe fn renderer_ref(r: Opaque<vgfx_renderer>) -> Option<&'static Renderer> {
    unsafe { (r as *const RendererHandle).as_ref().map(|h| &h.0) }
}

/// # Safety
/// `c` must be null or point to 4 readable `f32`.
unsafe fn read_rgba(c: *const f32) -> Option<Color> {
    unsafe {
        c.as_ref()
            .map(|_| Color::srgb(*c, *c.add(1), *c.add(2), *c.add(3)))
    }
}

/// # Safety
/// `p` must be null or point to 3 readable `f32`.
unsafe fn read_f32_3(p: *const f32) -> Option<[f32; 3]> {
    unsafe { p.as_ref().map(|_| [*p, *p.add(1), *p.add(2)]) }
}

/// # Safety
/// `m` must be null or point to 16 readable `f32` (column-major).
unsafe fn read_mat4(m: *const f32) -> Option<crate::math::Mat4> {
    unsafe {
        m.as_ref()
            .map(|_| crate::math::Mat4::from_cols_slice(std::slice::from_raw_parts(m, 16)))
    }
}

/// # Safety
/// `m` must be null or a live mesh handle from [`vgfx_mesh_from_data`].
unsafe fn mesh_ref(m: Opaque<vgfx_mesh>) -> Option<&'static Arc<Mesh>> {
    unsafe { (m as *const Arc<Mesh>).as_ref() }
}

// ---- lifecycle --------------------------------------------------------------

/// Create an offscreen renderer on the platform's native GPU backend.
/// Returns NULL on failure (see `vgfx_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_offscreen(width: u32, height: u32) -> Opaque<vgfx_renderer> {
    guard(|| Renderer::offscreen(width, height))
        .map(|r| Box::into_raw(Box::new(RendererHandle(r))) as Opaque<vgfx_renderer>)
        .unwrap_or_else(std::ptr::null_mut)
}

/// Destroy a renderer created by `vgfx_offscreen`. NULL is accepted.
///
/// # Safety
/// `r` must be null or a live handle from [`vgfx_offscreen`]; do not use it
/// afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_renderer_destroy(r: Opaque<vgfx_renderer>) {
    if !r.is_null() {
        unsafe { drop(Box::from_raw(r as *mut RendererHandle)) };
    }
}

/// Last error message (thread-local, NUL-terminated). Valid until the next
/// vgfx call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn vgfx_last_error() -> *const std::os::raw::c_char {
    static OK: &[u8] = b"ok\0";
    LAST_ERROR.with(|b| {
        let buf = b.borrow();
        if buf.len == 0 {
            OK.as_ptr() as *const std::os::raw::c_char
        } else {
            buf.bytes.as_ptr() as *const std::os::raw::c_char
        }
    })
}

/// Begin a new frame (clears accumulated draw operations). NULL is a no-op.
///
/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_begin_frame(r: Opaque<vgfx_renderer>) {
    unsafe {
        if let Some(r) = renderer_mut(r) {
            guard(|| {
                r.begin_frame();
                Ok(())
            });
        }
    }
}

/// Set the frame's background color (RGBA, \[0,1\]). NULL is a no-op.
///
/// # Safety
/// `r` must be null or a live handle; `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_set_background(r: Opaque<vgfx_renderer>, rgba: *const f32) {
    unsafe {
        if let (Some(r), Some(c)) = (renderer_mut(r), read_rgba(rgba)) {
            guard(|| {
                r.set_background(c);
                Ok(())
            });
        }
    }
}

/// Set a right-handed perspective camera. Matrices/vectors are row lists of
/// f32: eye/target/up are 3 components each. NULL is a no-op.
///
/// # Safety
/// `r` must be null or a live handle; `eye`/`target`/`up` null or 3
/// readable `f32` each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_set_camera_perspective(
    r: Opaque<vgfx_renderer>,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
    eye: *const f32,
    target: *const f32,
    up: *const f32,
) {
    unsafe {
        if let (Some(r), Some(e), Some(t), Some(u)) = (
            renderer_mut(r),
            read_f32_3(eye),
            read_f32_3(target),
            read_f32_3(up),
        ) {
            guard(|| {
                r.set_camera_perspective(fov_y_deg, aspect, near, far, e, t, u);
                Ok(())
            });
        }
    }
}

// ---- 2D ------------------------------------------------------------------

/// # Safety
/// `r` must be null or a live handle; `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_fill_rect(
    r: Opaque<vgfx_renderer>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(c)) = (renderer_mut(r), read_rgba(rgba)) {
            guard(|| {
                r.fill_rect(x, y, w, h, c);
                Ok(())
            });
        }
    }
}

/// # Safety
/// `r` must be null or a live handle; `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_fill_circle(
    r: Opaque<vgfx_renderer>,
    cx: f32,
    cy: f32,
    radius: f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(c)) = (renderer_mut(r), read_rgba(rgba)) {
            guard(|| {
                r.fill_circle(cx, cy, radius, c);
                Ok(())
            });
        }
    }
}

/// # Safety
/// `r` must be null or a live handle; `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_fill_rounded_rect(
    r: Opaque<vgfx_renderer>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    corner: f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(c)) = (renderer_mut(r), read_rgba(rgba)) {
            guard(|| {
                r.fill_rounded_rect(x, y, w, h, corner, c);
                Ok(())
            });
        }
    }
}

/// # Safety
/// `r` must be null or a live handle; `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_stroke_rect(
    r: Opaque<vgfx_renderer>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    width: f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(c)) = (renderer_mut(r), read_rgba(rgba)) {
            guard(|| {
                r.stroke_rect(x, y, w, h, width, c);
                Ok(())
            });
        }
    }
}

/// # Safety
/// `r` must be null or a live handle; `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_stroke_circle(
    r: Opaque<vgfx_renderer>,
    cx: f32,
    cy: f32,
    radius: f32,
    width: f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(c)) = (renderer_mut(r), read_rgba(rgba)) {
            guard(|| {
                r.stroke_circle(cx, cy, radius, width, c);
                Ok(())
            });
        }
    }
}

// ---- 3D ------------------------------------------------------------------

/// Draw a unit cube (size 1) with a model matrix (16 column-major floats).
/// NULL is a no-op.
///
/// # Safety
/// `r` must be null or a live handle; `model` null or 16 readable `f32`;
/// `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_draw_cube(
    r: Opaque<vgfx_renderer>,
    model: *const f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(m), Some(c)) = (renderer_mut(r), read_mat4(model), read_rgba(rgba)) {
            guard(|| {
                r.draw_mesh(Mesh::cube(1.0), m, c);
                Ok(())
            });
        }
    }
}

/// Draw a UV sphere. NULL is a no-op.
///
/// # Safety
/// Like [`vgfx_draw_cube`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_draw_sphere(
    r: Opaque<vgfx_renderer>,
    radius: f32,
    width_segments: u32,
    height_segments: u32,
    model: *const f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(m), Some(c)) = (renderer_mut(r), read_mat4(model), read_rgba(rgba)) {
            guard(|| {
                r.draw_mesh(Mesh::sphere(radius, width_segments, height_segments), m, c);
                Ok(())
            });
        }
    }
}

/// Build a mesh from flat arrays (xyz positions, xyz normals, rgba colors,
/// u32 indices). Returns NULL on invalid input.
///
/// # Safety
/// Each pointer must be null or point to `n_*` readable elements of its type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_mesh_from_data(
    positions: *const f32,
    n_positions: u32,
    normals: *const f32,
    n_normals: u32,
    colors: *const f32,
    n_colors: u32,
    indices: *const u32,
    n_indices: u32,
) -> Opaque<vgfx_mesh> {
    unsafe {
        guard(|| {
            if positions.is_null() || normals.is_null() || colors.is_null() || indices.is_null() {
                return Err(crate::Error::Backend("null mesh pointer".into()));
            }
            let p = std::slice::from_raw_parts(positions, n_positions as usize);
            let n = std::slice::from_raw_parts(normals, n_normals as usize);
            let c = std::slice::from_raw_parts(colors, n_colors as usize);
            let i = std::slice::from_raw_parts(indices, n_indices as usize);
            Mesh::from_data(p, n, c, i)
                .ok_or_else(|| crate::Error::Backend("invalid mesh data".into()))
        })
        .map(|m| Box::into_raw(Box::new(Arc::new(m))) as Opaque<vgfx_mesh>)
        .unwrap_or_else(std::ptr::null_mut)
    }
}

/// Destroy a mesh created by `vgfx_mesh_from_data`. NULL is accepted.
///
/// # Safety
/// `m` must be null or a live handle from [`vgfx_mesh_from_data`]; do not
/// use it afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_mesh_destroy(m: Opaque<vgfx_mesh>) {
    if !m.is_null() {
        unsafe { drop(Box::from_raw(m as *mut Arc<Mesh>)) };
    }
}

/// Draw a mesh with a model transform and color. NULL is a no-op.
///
/// # Safety
/// `r`/`m` must be null or live handles; `model` null or 16 readable `f32`;
/// `rgba` null or 4 readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_draw_mesh(
    r: Opaque<vgfx_renderer>,
    m: Opaque<vgfx_mesh>,
    model: *const f32,
    rgba: *const f32,
) {
    unsafe {
        if let (Some(r), Some(mesh), Some(mat), Some(c)) = (
            renderer_mut(r),
            mesh_ref(m),
            read_mat4(model),
            read_rgba(rgba),
        ) {
            let mesh = Arc::clone(mesh);
            let material = Material {
                color: c,
                ..Material::default()
            };
            guard(|| {
                r.draw_mesh_material(mesh, mat, material);
                Ok(())
            });
        }
    }
}

// ---- output ----------------------------------------------------------------

/// Render the accumulated frame. Returns 0 on success, -1 on error.
///
/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_present(r: Opaque<vgfx_renderer>) -> i32 {
    unsafe {
        match renderer_mut(r) {
            None => {
                set_err(&"null renderer");
                -1
            }
            Some(r) => match guard(|| r.present()) {
                Some(()) => 0,
                _ => -1,
            },
        }
    }
}

/// Save the last offscreen frame as a PNG. Returns 0 on success, -1 on error.
///
/// # Safety
/// `r` must be null or a live handle; `path` null or a valid NUL-terminated
/// UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_save_png(
    r: Opaque<vgfx_renderer>,
    path: *const std::os::raw::c_char,
) -> i32 {
    unsafe {
        if r.is_null() || path.is_null() {
            set_err(&"null renderer or path");
            return -1;
        }
        let path = match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_err(&"path is not valid UTF-8");
                return -1;
            }
        };
        match renderer_mut(r) {
            None => {
                set_err(&"null renderer");
                -1
            }
            Some(r) => match guard(|| r.save_png(path)) {
                Some(()) => 0,
                _ => -1,
            },
        }
    }
}

/// Backend name, e.g. "metal". NUL-terminated; valid until the next vgfx call
/// on this thread. Do not free. Returns NULL for a null handle.
///
/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_backend_name(
    r: Opaque<vgfx_renderer>,
) -> *const std::os::raw::c_char {
    unsafe {
        match renderer_ref(r) {
            None => std::ptr::null(),
            Some(r) => match guard(|| Ok(r.backend_name())) {
                Some("metal") => c"metal".as_ptr(),
                Some(_) => c"vulkan (unimplemented)".as_ptr(),
                _ => std::ptr::null(),
            },
        }
    }
}

/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_width(r: Opaque<vgfx_renderer>) -> u32 {
    unsafe { renderer_ref(r).map(|r| r.size().0).unwrap_or(0) }
}

/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_height(r: Opaque<vgfx_renderer>) -> u32 {
    unsafe { renderer_ref(r).map(|r| r.size().1).unwrap_or(0) }
}

/// Resize the render target. NULL is a no-op.
///
/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_resize(r: Opaque<vgfx_renderer>, width: u32, height: u32) {
    unsafe {
        if let Some(r) = renderer_mut(r) {
            guard(|| {
                r.resize(width, height);
                Ok(())
            });
        }
    }
}

/// Reserve Rust and GPU frame storage between frames. Returns 0/-1.
/// # Safety
/// `r` must be null or a live renderer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_reserve_geometry(
    r: Opaque<vgfx_renderer>,
    draws: u32,
    vertices: u32,
    indices: u32,
) -> i32 {
    let Some(r) = (unsafe { renderer_mut(r) }) else {
        set_err(&"null renderer");
        return -1;
    };
    if guard(|| r.reserve_geometry(draws as usize, vertices as usize, indices as usize)).is_some() {
        0
    } else {
        -1
    }
}

#[repr(C)]
pub struct vgfx_prepared_mesh {
    _private: [u8; 0],
}
/// Upload a mesh once; prepared handles are specific to their renderer.
/// # Safety
/// Both handles must be null or live handles from vgfx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_prepare_mesh(
    r: Opaque<vgfx_renderer>,
    m: Opaque<vgfx_mesh>,
) -> Opaque<vgfx_prepared_mesh> {
    let (Some(r), Some(m)) = (unsafe { renderer_mut(r) }, unsafe { mesh_ref(m) }) else {
        set_err(&"null renderer or mesh");
        return std::ptr::null_mut();
    };
    guard(|| r.prepare_mesh(Arc::clone(m)))
        .map(|m| Box::into_raw(Box::new(m)).cast())
        .unwrap_or(std::ptr::null_mut())
}
/// # Safety
/// `m` must be null or a live prepared mesh; do not use it after destruction.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_prepared_mesh_destroy(m: Opaque<vgfx_prepared_mesh>) {
    if !m.is_null() {
        unsafe {
            drop(Box::from_raw(
                m.cast::<crate::backend::renderer::PreparedMesh>(),
            ))
        };
    }
}
/// Record a resident mesh without allocation, lookup or vertex upload.
/// # Safety
/// Handles must be null or live; arrays null or readable for 16/4 floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgfx_draw_prepared_mesh(
    r: Opaque<vgfx_renderer>,
    m: Opaque<vgfx_prepared_mesh>,
    model: *const f32,
    rgba: *const f32,
) {
    if let (Some(r), Some(mesh), Some(model), Some(color)) = (
        unsafe { renderer_mut(r) },
        unsafe { m.cast::<crate::backend::renderer::PreparedMesh>().as_ref() },
        unsafe { read_mat4(model) },
        unsafe { read_rgba(rgba) },
    ) {
        guard(|| {
            r.draw_prepared_mesh(
                mesh,
                model,
                Material {
                    color,
                    ..Material::default()
                },
            );
            Ok(())
        });
    }
}
