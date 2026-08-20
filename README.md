# vgfx

Cross-language, cross-platform 2D/3D vector-graphics engine in Rust that talks
**directly to the GPU**:

- **Apple (macOS/iOS)** → Metal — implemented
- **Linux / Windows** → Vulkan (`ash`) — stub, in progress

> All rendering runs on the GPU. Apple's NPU is an ML accelerator, not a
> graphics device, and is not used.

## Features

- 2D: rects, rounded rects, circles, cubic-bezier paths — fill (even-odd,
  holes) and stroke
- 3D: indexed meshes, perspective camera, per-vertex colors, Lambert +
  specular lighting, depth
- Immediate-mode `Renderer` API (retained `Scene` under the hood)
- Windowed (winit) and headless/offscreen rendering with PNG export
- Plain C ABI (`include/vgfx.h`) — usable from C/C++, Python, Go, Zig, Swift…

## Quick start

```rust
use vgfx::prelude::*;

let mut r = Renderer::offscreen(1280, 720).unwrap();
r.begin_frame();
r.set_background(Color::srgb(0.08, 0.09, 0.12, 1.0));
r.fill_rect(40.0, 40.0, 200.0, 120.0, Color::srgb(0.2, 0.8, 0.6, 1.0));
r.fill_circle(400.0, 200.0, 80.0, Color::srgb(0.9, 0.4, 0.2, 1.0));
r.present().unwrap();
r.save_png("out.png").unwrap();
```

Bundled examples:

```sh
make run-2d        # offscreen 2D scene → out.png
make run-3d        # offscreen 3D scene → out.png
make run-windowed  # interactive Metal window: spinning cube, bouncing sphere
```

## C ABI

```sh
make ffi-c     # examples/c_example/main.c, compiled against the cdylib
make ffi-py    # examples/ffi_python/render.py, via ctypes
```

Link `libvgfx.{a,so,dylib}`, include `include/vgfx.h`. The full API surface is
in `src/ffi.rs`.

## Cross-compilation

```sh
make check-all             # cargo check: native + x86_64-linux
make cross-linux-aarch64   # needs `cargo install cross` (Cross.toml included)
make cross-linux-x86_64
make cross-windows
```

CI (`.github/workflows/ci.yml`) builds and tests macOS/Linux/Windows on every
push.

## Layout

- `src/geometry` — paths, bezier flattening, fill/stroke tessellation
- `src/scene` — camera, meshes, materials
- `src/backend` — `Gpu` trait, Metal + Vulkan implementations, `Renderer`
- `src/ffi.rs` — C ABI; `include/vgfx.h`; `examples/` — one folder per demo

## License

MIT
