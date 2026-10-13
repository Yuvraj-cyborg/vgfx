# vgfx

Rust 2D/3D vector graphics with **zero third-party libraries**. Rust owns frame
storage and GPU shared-buffer backing memory. Metal is accessed through local
system bindings; macOS demos use native AppKit. Linux/Windows Vulkan is currently
an explicit unimplemented backend.

The design aims for a flat execution profile: fixed primitive subdivision,
explicit preparation, reserved frame capacity, retained mesh uploads, and no
Rust heap growth during successful prepared frame submission. GPU duration still
depends on geometry and rasterization. See [ARCHITECTURE.md](ARCHITECTURE.md) for
the philosophy, allocation boundaries, limitations, and ownership rules.

```rust
use std::sync::Arc;
use vgfx::prelude::*;

let mut r = Renderer::offscreen(1280, 720)?;
r.reserve_geometry(4096, 65536, 262144)?;
let cube = r.prepare_mesh(Arc::new(Mesh::cube(1.0)))?;
// Repeat this block without Rust frame-loop allocation:
r.begin_frame();
r.set_background(Color::srgb(0.08, 0.09, 0.12, 1.0));
r.draw_prepared_mesh(&cube, Mat4::IDENTITY, Material::default());
r.fill_rect(40.0, 40.0, 200.0, 120.0, Color::srgb(0.2, 0.8, 0.6, 1.0));
r.fill_circle(400.0, 200.0, 80.0, Color::srgb(0.9, 0.4, 0.2, 1.0));
r.present()?;
// Output is outside the frame loop:
r.save_png("out.png")?;
# Ok::<(), vgfx::Error>(())
```

General paths support even-odd fills, holes and strokes. Prepare them once with
`Renderer::prepare_2d`, then record with `draw_prepared_2d`. Cubics use 32 fixed
segments; circle/rounded-rectangle convenience draws use precomputed polygons.
The local OBJ importer supports positions, normals, triangles and planar convex
n-gons; triangulate concave faces before import. PNG export uses uncompressed
DEFLATE for predictable work and produces larger files.

Math types moved from the external math crate to `vgfx::math` (also exported by
the prelude). Existing convenience mesh/path APIs remain for one-shot rendering;
they can allocate during recording. Use prepared APIs in steady frame loops.

```sh
cargo test --all-targets --offline
cargo clippy --all-targets --offline -- -D warnings
cargo run --release --offline --example resident_probe # requires Metal GPU
make run-2d run-3d run-model  # offscreen examples → out.png
make run-windowed run-doom   # macOS interactive demos
make ffi-c ffi-py           # plain C ABI / Python ctypes examples
make check-all              # native + x86_64 Linux checks
```

Link `libvgfx.{a,so,dylib}` and include [include/vgfx.h](include/vgfx.h).
Mesh/renderer/prepared handles allocate and free in Rust. C callers can reserve
frame capacity and prepare resident meshes through the same API; capacity overflow
is reported by `vgfx_present`. Do not free returned handles with C `free`.

Rust standard library and native OS frameworks are required; no Cargo dependencies,
build helpers, image/model libraries, window libraries, or vendored crate code.

MIT
