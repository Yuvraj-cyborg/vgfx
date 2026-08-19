/* vgfx C example: render a 2D + 3D scene offscreen and save a PNG.
 *
 * Build & run from this directory: make run
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "vgfx.h"

static void identity(float m[16]) {
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* Column-major composition: out = a * b. */
static void mat4_mul(const float a[16], const float b[16], float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
            out[col * 4 + row] = s;
        }
}

static void mat4_translate(float out[16], float x, float y, float z) {
    identity(out);
    out[12] = x;
    out[13] = y;
    out[14] = z;
}

static void mat4_rot_y(float out[16], float rad) {
    identity(out);
    out[0] = cosf(rad);
    out[2] = -sinf(rad);
    out[8] = sinf(rad);
    out[10] = cosf(rad);
}

static void check(const char *what) {
    const char *err = vgfx_last_error();
    if (err && strcmp(err, "ok") != 0) {
        fprintf(stderr, "vgfx error during %s: %s\n", what, err);
        exit(1);
    }
}

int main(void) {
    vgfx_renderer *r = vgfx_offscreen(960, 540);
    if (!r) {
        fprintf(stderr, "vgfx_offscreen failed: %s\n", vgfx_last_error());
        return 1;
    }
    printf("backend: %s (%ux%u)\n", vgfx_backend_name(r), vgfx_width(r), vgfx_height(r));

    vgfx_begin_frame(r);
    const float bg[4] = {0.07f, 0.08f, 0.11f, 1.0f};
    vgfx_set_background(r, bg);

    /* 2D */
    const float green[4] = {0.20f, 0.75f, 0.60f, 1.0f};
    const float red[4] = {0.85f, 0.30f, 0.35f, 1.0f};
    const float blue[4] = {0.30f, 0.55f, 0.95f, 1.0f};
    vgfx_fill_rect(r, 40.0f, 40.0f, 220.0f, 140.0f, green);
    vgfx_fill_circle(r, 560.0f, 120.0f, 90.0f, red);
    vgfx_stroke_rect(r, 40.0f, 240.0f, 220.0f, 140.0f, 6.0f, blue);

    /* 3D */
    vgfx_set_camera_perspective(r, 50.0f, 960.0f / 540.0f, 0.1f, 100.0f,
                                (const float[]){0.0f, 0.6f, 4.5f},
                                (const float[]){0.0f, 0.0f, 0.0f},
                                (const float[]){0.0f, 1.0f, 0.0f});
    const float cube_color[4] = {0.35f, 0.55f, 0.95f, 1.0f};
    float rot[16], trans[16], model[16];
    mat4_rot_y(rot, 0.9f);
    mat4_translate(trans, -1.6f, 0.0f, 0.0f);
    mat4_mul(trans, rot, model);
    vgfx_draw_cube(r, model, cube_color);
    mat4_translate(model, 1.5f, 0.0f, 0.0f);
    const float sphere_color[4] = {0.95f, 0.45f, 0.25f, 1.0f};
    vgfx_draw_sphere(r, 1.1f, 32, 24, model, sphere_color);

    if (vgfx_present(r) != 0) check("present");
    if (vgfx_save_png(r, "c_example.png") != 0) check("save_png");
    printf("wrote c_example.png\n");

    vgfx_renderer_destroy(r);
    return 0;
}
