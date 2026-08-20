# vgfx

Cross-language, cross-platform 2D/3D vector graphics API in Rust that talks
**directly to the GPU**:

- **Apple (macOS/iOS)** → [Metal](https://developer.apple.com/metal/)
- **Linux / Windows** → [Vulkan](https://www.vulkan.org/) (via `ash`)

> Note on "NPU": Apple's NPU (ANE) is a machine-learning accelerator and
> cannot render graphics. vgfx uses the **GPU** (Apple silicon / AMD / Intel)
> for all 3D rendering, which is the correct acceleration path on Mac.

## Features

- 2D vector graphics: rects, rounded rects, circles, polygons, cubic-bezier
  paths with **fill** (even-odd, holes supported) and **stroke** (miter joins)
- 3D vector/mesh graphics: indexed meshes, perspective camera, per-vertex
  colors, Lambert + specular lighting, depth testing
- Immediate-mode `Renderer` API plus a retained `Scene` for advanced users
- Windowed (winit) and **headless/offscreen** rendering (PNG export, CI)
- **C ABI** (`include/vgfx.h`) → usable from C/C++, Python, Go, Zig, Swift…
- Cross-compilation setup for macOS (aarch64/x86_64), Linux (gnu/musl),
  Windows (gnu) — see [Cross-compilation](#cross-compilation)

## Layout

```
src/
  geometry/   paths, bezier flattening, tessellation (fill + stroke)
  scene/      camera, meshes, materials, scene graph
  backend/    Backend trait, Metal + Vulkan implementations, Renderer
  ffi.rs      C ABI (cross-language entry point)
include/      vgfx.h C header
examples/     one sub-folder per feature
```

## Quick start (Rust)

```rust
use vgfx::prelude::*;

let mut r = vgfx::Renderer::offscreen(1280, 720).unwrap();
r.begin_frame();
r.set_background(Color::srgb(0.08, 0.09, 0.12, 1.0));
r.fill_rect(40.0, 40.0, 200.0, 120.0, Color::srgb(0.2, 0.8, 0.6, 1.0));
r.fill_circle(400.0, 200.0, 80.0, Color::srgb(0.9, 0.4, 0.2, 1.0));
r.present();
r.save_png("out.png").unwrap();
```

Run the bundled examples (each feature has its own sub-folder):

```sh
make run-2d        # cargo run --example offscreen_2d   → out.png (headless)
make run-3d        # cargo run --example offscreen_3d   → out.png (headless)
make run-windowed  # cargo run --example windowed       → interactive Metal window
```

`offscreen_2d`/`offscreen_3d` render to a PNG in the current directory
(nothing GPU-interactive, so they also work in CI/headless setups).

## Cross-language (C ABI)

```sh
make ffi-c     # builds the cdylib + compiles examples/c_example/main.c → runs it
make ffi-py    # runs examples/ffi_python/render.py via ctypes against the cdylib
```

From any language, link against `libvgfx.{a,so,dylib}` and include
`include/vgfx.h`. The full API surface is in `src/ffi.rs`.

## Cross-compilation

```sh
make check-all        # cargo check for all supported targets
make cross-linux-x86_64   # cargo zigbuild (or: cross build) → target/.../libvgfx.a
make cross-windows
make universal-mac    # lipo aarch64 + x86_64 dylib
```

Prereqs (pick one):
- `cargo install cargo-zigbuild` (recommended, no Docker)
- `cross` (Docker) — `Cross.toml` is included
- plain `rustup target add` + a C cross-toolchain (musl/gnu)

CI (`.github/workflows/ci.yml`) checks every target on every push.

## License

MIT
