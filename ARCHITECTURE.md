# vgfx architecture and performance philosophy

Keep the execution profile as flat as practical. Rust owns CPU allocations and
shared buffer storage. The crate and its examples have zero third-party Rust
libraries, including development dependencies. Rust's standard library and
operating-system GPU/UI frameworks are the platform boundary.

## Preparation, recording, submission

Allocate at renderer creation, explicit capacity reservation, asset preparation,
resize, and output. Reuse storage across frames. Do not add adaptive quality,
automatic cache eviction, hashing, file parsing, or buffer growth to submission.

```text
setup: reserve_geometry + prepare_mesh / prepare_2d
frame: begin_frame -> record primitives / prepared geometry -> present
output: read_pixels / save_png
```

`Renderer` keeps its vertex, index, and draw vectors across `begin_frame` calls.
Default capacities are 4,096 draws, 65,536 2D vertices, and 262,144 indices.
`reserve_geometry(draws, vertices, indices)` explicitly increases CPU/GPU storage
between frames. Recording checks capacity before appending a complete shape;
overflow is reported by `present` and stays an error until `begin_frame`.

Built-in rectangles, circles, rounded rectangles and their supported strokes
use stack scratch arrays and precomputed unit-circle points. Circles have 48
segments; rounded rectangles have 12 segments per corner. Shape dimensions do
not select a different subdivision count. Adjacent 2D records coalesce into one
draw, preserving triangle/color order. Bitmap text records 15 cells per character,
with transparent cells for unset bits, so glyph content does not change the work.

`prepare_2d` runs general fill/stroke triangulation once. `draw_prepared_2d` copies
that geometry to the reserved frame buffers. Compatibility `fill_2d`/`stroke_2d`
perform preparation while recording and allocate: keep them out of a strict frame
loop. Cubics always flatten to 32 segments. The tolerance argument is retained
for source compatibility but no longer drives adaptive recursion. Fixed
subdivision does not promise a pixel-error bound at arbitrary scale. General
polygon triangulation still depends on topology and vertex count; it belongs in
preparation. Hole coverage regression tests remain required.

`prepare_mesh(Arc<Mesh>)` validates and uploads immutable geometry once. Its
`PreparedMesh` contains a renderer-specific slot; `draw_prepared_mesh` uses direct
indexing and records only the mesh reference, transform and material. Submission
cost per prepared 3D draw on the CPU does not traverse its vertices or indices.
A prepared mesh from another renderer is rejected. The backend pins resident
meshes until renderer destruction; limits are 4,096 slots and 256 MiB of aligned
shared buffers. There is no eviction or re-upload on a frame boundary.

Compatibility `draw_mesh`, `draw_mesh_shared`, and `draw_mesh_material` perform
preparation/linear identity lookup while recording. They are useful for one-shot
renders, not the strict path. Rebuilding owned meshes each frame consumes the
resident budget. Reuse prepared handles instead. C exposes the same setup through
`vgfx_reserve_geometry`, `vgfx_prepare_mesh`, `vgfx_draw_prepared_mesh`, and
`vgfx_prepared_mesh_destroy`. Foreign callers never free Rust allocations directly.

## Native backend and ownership

Metal calls go through the local typed Objective-C ABI in `backend/native.rs`.
Each call specifies the system SDK signature. Owned objects adopt one reference;
autoreleased results are retained when they must outlive their pool. The macOS
examples use the small AppKit event loop in `window.rs` (main thread only), draining
at most 64 native events before a frame.

Metal buffers use `newBufferWithBytesNoCopy` over Rust page-aligned allocations.
The backend releases each Metal buffer before Rust deallocates its backing bytes.
2D vertex/index buffers, readback staging, the render-pass descriptor, and depth
targets persist. Resize recreates targets/staging outside normal submission.
Uniforms are passed as inline bytes. Command buffers/encoders/drawables remain
system-managed objects; the driver may allocate internally. System texture
allocation remains the GPU driver's responsibility.

Submission draws 3D before 2D, commits once, and waits for completion. That wait
makes reuse and Rust buffer destruction safe but introduces GPU/scheduler timing
variation. Triple buffering needs an explicit in-flight ownership design before
removing the wait. Do not just remove it.

The local column-major math types are in `math.rs`. Perspective depth is [0,1]
for Metal/Vulkan. GPU structs are `repr(C)` with explicit scalar/array fields;
only these padding-free layouts and u32 indices enter shared buffers. Normals
still assume rigid/uniformly scaled models.

Linux/Windows Vulkan remains the existing explicit unimplemented backend; the
unused Vulkan wrapper dependency has been removed. Builds/checks work without
claiming rendering support there.

## Import/export

The local OBJ parser reads positions, normals, positive/negative indices,
triangle faces and planar convex n-gons. It ignores materials/UVs/groups, keeps
explicit corner normals, and computes area-weighted smooth normals when missing.
Concave/nonplanar n-gons must be triangulated before import and return an error.
Parsing and allocation happen at asset preparation, never submission.

PNG output uses filter 0 and stored DEFLATE blocks with CRC-32/Adler-32. It streams
through fixed scratch space; it trades larger files for predictable encoding work
and no compression search. Readback allocates one returned RGBA vector; staging
is reused and BGRA conversion happens in place.

## What the flat-work contract covers

No Rust heap allocation/reallocation on successful built-in/prepared frame paths
within reserved capacities. Prepared 3D record/encode work depends on draw count,
not mesh triangle count. Primitive subdivision depends on primitive type, not
radius, dimensions or curvature. This is not a cryptographic constant-time or
identical frame-duration guarantee: 2D copies scale with geometry, more draws
mean more encoding, and rasterization, overdraw, driver work and GPU waits vary.
No theoretical maximum throughput claim is made without measurement.

`cargo test --all-targets --offline` includes a thread-local allocation counter
for repeated prepared frames with small/large meshes, capacity recovery, foreign
renderer handles, hole coverage, math and OBJ regressions. Run
`cargo run --release --offline --example resident_probe` on a machine with Metal
for real GPU allocation counts, resident uploads, readback and resize checks.
The probe counts Rust allocator calls; it does not count driver allocations.

## Source map

- `backend/renderer.rs`: persistent frame storage, primitive recording, preparation
- `backend/common.rs`: draw ranges, GPU layouts, backend interface
- `backend/metal.rs`: native Metal, resident meshes, Rust-backed buffers
- `backend/native.rs`: typed Objective-C dispatch and object/pool ownership
- `window.rs`: macOS demo window/input loop
- `geometry/`: path storage, affine math, fixed cubic flattening, triangulation
- `scene/`: scene values, camera, meshes and local OBJ import
- `math.rs`, `png.rs`, `text.rs`: local math, PNG export, fixed-cell HUD font
- `ffi.rs` and `include/vgfx.h`: Rust-owned C lifecycle and prepared mesh API
