/* vgfx — C ABI for a cross-platform GPU 2D/3D vector-graphics renderer.
 *
 * Conventions:
 *  - Opaque handles: vgfx_renderer, vgfx_mesh.
 *  - Handle-returning functions return NULL on error; i32-returning
 *    functions return -1 on error. Call vgfx_last_error() for details
 *    (thread-local, valid until the next vgfx call on that thread).
 *  - Matrices are 16 column-major floats; colors are 4 floats RGBA in
 *    [0, 1].
 *
 * Linking:
 *  - Linux:    -lvgfx  (plus your Vulkan/loader deps when using Vulkan)
 *  - macOS:    -lvgfx -lobjc -framework Metal -framework Foundation
 *              -framework QuartzCore -framework AppKit
 *  - Windows:  vgfx.lib (Vulkan backend)
 */
#ifndef VGFX_H
#define VGFX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vgfx_renderer vgfx_renderer;
typedef struct vgfx_mesh vgfx_mesh;
typedef struct vgfx_prepared_mesh vgfx_prepared_mesh;

/* ---- lifecycle ---------------------------------------------------------- */

/* Create an offscreen renderer on the platform's native GPU backend.
 * Returns NULL on failure (see vgfx_last_error). */
vgfx_renderer *vgfx_offscreen(uint32_t width, uint32_t height);

/* Destroy a renderer created by vgfx_offscreen. NULL is accepted. */
void vgfx_renderer_destroy(vgfx_renderer *r);

/* Last error message (thread-local, NUL-terminated; "ok" when no error). */
const char *vgfx_last_error(void);

/* Backend name, e.g. "metal" (static string, do not free). NULL-safe. */
const char *vgfx_backend_name(vgfx_renderer *r);

uint32_t vgfx_width(vgfx_renderer *r);
uint32_t vgfx_height(vgfx_renderer *r);
void vgfx_resize(vgfx_renderer *r, uint32_t width, uint32_t height);

/* ---- frame -------------------------------------------------------------- */

/* Begin a new frame (clears accumulated draw operations). */
void vgfx_begin_frame(vgfx_renderer *r);

/* Set the frame's background color (RGBA in [0,1]). */
void vgfx_set_background(vgfx_renderer *r, const float rgba[4]);

/* Set a right-handed perspective camera (eye/target/up are 3 floats each). */
void vgfx_set_camera_perspective(vgfx_renderer *r,
                                 float fov_y_deg, float aspect,
                                 float near, float far,
                                 const float eye[3],
                                 const float target[3],
                                 const float up[3]);

/* ---- 2D ------------------------------------------------------------------ */

void vgfx_fill_rect(vgfx_renderer *r, float x, float y, float w, float h, const float rgba[4]);
void vgfx_fill_circle(vgfx_renderer *r, float cx, float cy, float radius, const float rgba[4]);
void vgfx_fill_rounded_rect(vgfx_renderer *r, float x, float y, float w, float h,
                            float corner, const float rgba[4]);
void vgfx_stroke_rect(vgfx_renderer *r, float x, float y, float w, float h,
                      float width, const float rgba[4]);
void vgfx_stroke_circle(vgfx_renderer *r, float cx, float cy, float radius,
                        float width, const float rgba[4]);

/* ---- 3D ------------------------------------------------------------------ */

/* Draw a unit cube (size 1) with a model matrix (16 column-major floats). */
void vgfx_draw_cube(vgfx_renderer *r, const float model[16], const float rgba[4]);

/* Draw a UV sphere. */
void vgfx_draw_sphere(vgfx_renderer *r, float radius,
                      uint32_t width_segments, uint32_t height_segments,
                      const float model[16], const float rgba[4]);

/* Build a mesh from flat arrays: xyz positions, xyz normals, rgba colors,
 * u32 indices. Returns NULL on invalid input. */
vgfx_mesh *vgfx_mesh_from_data(const float *positions, uint32_t n_positions,
                               const float *normals, uint32_t n_normals,
                               const float *colors, uint32_t n_colors,
                               const uint32_t *indices, uint32_t n_indices);

/* Destroy a mesh created by vgfx_mesh_from_data. NULL is accepted. */
void vgfx_mesh_destroy(vgfx_mesh *m);

/* Draw a mesh with a model matrix and color. */
void vgfx_draw_mesh(vgfx_renderer *r, vgfx_mesh *m, const float model[16], const float rgba[4]);

/* Setup-time storage and resident meshes. All allocations belong to Rust.
 * Capacity overflow is reported by vgfx_present; begin_frame clears the error.
 * A prepared mesh may only be drawn by the renderer that prepared it.
 * Renderer destruction releases GPU storage. Prepared-handle destruction
 * releases its CPU reference; destroy it before destroying its renderer. */
int vgfx_reserve_geometry(vgfx_renderer *r, uint32_t draws, uint32_t vertices, uint32_t indices);
vgfx_prepared_mesh *vgfx_prepare_mesh(vgfx_renderer *r, vgfx_mesh *m);
void vgfx_prepared_mesh_destroy(vgfx_prepared_mesh *m);
void vgfx_draw_prepared_mesh(vgfx_renderer *r, vgfx_prepared_mesh *m, const float model[16], const float rgba[4]);

/* ---- output -------------------------------------------------------------- */

/* Render the accumulated frame. Returns 0 on success, -1 on error. */
int vgfx_present(vgfx_renderer *r);

/* Save the last offscreen frame as a PNG. Returns 0 on success, -1 on error. */
int vgfx_save_png(vgfx_renderer *r, const char *path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VGFX_H */
