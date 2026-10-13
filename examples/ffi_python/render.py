#!/usr/bin/env python3
"""vgfx Python (ctypes) example: render a 2D + 3D scene offscreen, save a PNG.

Mirrors examples/c_example/main.c, using the C ABI from include/vgfx.h.

Run from the repo root after `cargo build`:
    python3 examples/ffi_python/render.py          # writes ffi_python.png
or via the Makefile:
    make ffi-py
"""

import ctypes
import math
import os
from ctypes import (
    POINTER,
    c_char_p,
    c_float,
    c_uint32,
    c_void_p,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _lib_path():
    for dylib in (
        "libvgfx.dylib",
        "libvgfx.so",
        "vgfx.dll",
    ):
        cand = os.path.join(ROOT, "target", "debug", dylib)
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        "libvgfx not found; run `cargo build` from the repo root first"
    )


class Vgfx:
    def __init__(self):
        self.lib = ctypes.CDLL(_lib_path())
        # lifecycle
        self.lib.vgfx_offscreen.restype = c_void_p
        self.lib.vgfx_offscreen.argtypes = [c_uint32, c_uint32]
        self.lib.vgfx_renderer_destroy.argtypes = [c_void_p]
        self.lib.vgfx_last_error.restype = c_char_p
        self.lib.vgfx_backend_name.restype = c_char_p
        self.lib.vgfx_backend_name.argtypes = [c_void_p]
        self.lib.vgfx_width.restype = c_uint32
        self.lib.vgfx_width.argtypes = [c_void_p]
        self.lib.vgfx_height.restype = c_uint32
        self.lib.vgfx_height.argtypes = [c_void_p]
        self.lib.vgfx_resize.argtypes = [c_void_p, c_uint32, c_uint32]
        # frame
        self.lib.vgfx_begin_frame.argtypes = [c_void_p]
        self.lib.vgfx_set_background.argtypes = [c_void_p, POINTER(c_float)]
        self.lib.vgfx_set_camera_perspective.argtypes = [
            c_void_p, c_float, c_float, c_float, c_float,
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
        ]
        # 2D
        self.lib.vgfx_fill_rect.argtypes = [c_void_p, c_float, c_float, c_float, c_float, POINTER(c_float)]
        self.lib.vgfx_fill_circle.argtypes = [c_void_p, c_float, c_float, c_float, POINTER(c_float)]
        self.lib.vgfx_fill_rounded_rect.argtypes = [c_void_p, c_float, c_float, c_float, c_float, c_float, POINTER(c_float)]
        self.lib.vgfx_stroke_rect.argtypes = [c_void_p, c_float, c_float, c_float, c_float, c_float, POINTER(c_float)]
        self.lib.vgfx_stroke_circle.argtypes = [c_void_p, c_float, c_float, c_float, c_float, POINTER(c_float)]
        # 3D
        self.lib.vgfx_draw_cube.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
        self.lib.vgfx_draw_sphere.argtypes = [c_void_p, c_float, c_uint32, c_uint32, POINTER(c_float), POINTER(c_float)]
        # output
        self.lib.vgfx_present.restype = ctypes.c_int
        self.lib.vgfx_present.argtypes = [c_void_p]
        self.lib.vgfx_save_png.restype = ctypes.c_int
        self.lib.vgfx_save_png.argtypes = [c_void_p, c_char_p]

    def check(self, what):
        err = self.lib.vgfx_last_error().decode()
        if err != "ok":
            raise SystemExit(f"vgfx error during {what}: {err}")


def identity():
    m = [0.0] * 16
    m[0] = m[5] = m[10] = m[15] = 1.0
    return m


def mat4_mul(a, b):
    # column-major: out = a * b
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return out


def mat4_translate(x, y, z):
    m = identity()
    m[12], m[13], m[14] = x, y, z
    return m


def mat4_rot_y(rad):
    m = identity()
    m[0] = math.cos(rad)
    m[2] = -math.sin(rad)
    m[8] = math.sin(rad)
    m[10] = math.cos(rad)
    return m


def main():
    v = Vgfx()
    r = v.lib.vgfx_offscreen(960, 540)
    if not r:
        raise SystemExit(f"vgfx_offscreen failed: {v.lib.vgfx_last_error().decode()}")

    name = v.lib.vgfx_backend_name(r).decode()
    w, h = v.lib.vgfx_width(r), v.lib.vgfx_height(r)
    print(f"backend: {name} ({w}x{h})")

    f4 = lambda vals: (c_float * len(vals))(*vals)
    bg = f4([0.07, 0.08, 0.11, 1.0])
    green = f4([0.20, 0.75, 0.60, 1.0])
    red = f4([0.85, 0.30, 0.35, 1.0])
    blue = f4([0.30, 0.55, 0.95, 1.0])

    v.lib.vgfx_begin_frame(r)
    v.lib.vgfx_set_background(r, bg)

    v.lib.vgfx_fill_rect(r, 40.0, 40.0, 220.0, 140.0, green)
    v.lib.vgfx_fill_circle(r, 560.0, 120.0, 90.0, red)
    v.lib.vgfx_stroke_rect(r, 40.0, 240.0, 220.0, 140.0, 6.0, blue)

    eye = f4([0.0, 0.6, 4.5])
    target = f4([0.0, 0.0, 0.0])
    up = f4([0.0, 1.0, 0.0])
    v.lib.vgfx_set_camera_perspective(r, 50.0, 960.0 / 540.0, 0.1, 100.0, eye, target, up)

    cube_color = f4([0.35, 0.55, 0.95, 1.0])
    rot = f4(mat4_rot_y(0.9))
    trans = f4(mat4_translate(-1.6, 0.0, 0.0))
    model = f4(mat4_mul(list(trans), list(rot)))
    v.lib.vgfx_draw_cube(r, model, cube_color)

    sphere_color = f4([0.95, 0.45, 0.25, 1.0])
    model = f4(mat4_translate(1.5, 0.0, 0.0))
    v.lib.vgfx_draw_sphere(r, 1.1, 32, 24, model, sphere_color)

    v.lib.vgfx_present(r)
    v.check("present")
    v.lib.vgfx_save_png(r, b"ffi_python.png")
    v.check("save_png")
    print("wrote ffi_python.png")

    v.lib.vgfx_renderer_destroy(r)


if __name__ == "__main__":
    main()
