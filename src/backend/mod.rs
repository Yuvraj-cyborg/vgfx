//! GPU backends behind a common trait, plus the high-level [`Renderer`].
//!
//! Data flow per frame:
//!
//! ```text
//! app -> Renderer (records prepared draws) -> present() -> Frame -> Gpu::render
//! ```
//!
//! * `common` — shared vertex/uniform layouts, `Frame`, `Draw`, `Gpu` trait.
//! * `renderer` — the [`Renderer`] façade (preparation and persistent frame recording).
//! * `metal` (Apple) / `vulkan` (Linux/Windows) — backend implementations.

pub mod common;
#[cfg(target_vendor = "apple")]
pub mod metal;
#[cfg(target_vendor = "apple")]
pub(crate) mod native;
pub mod renderer;
#[cfg(not(target_vendor = "apple"))]
pub mod vulkan;

pub use common::{Draw, Frame, Gpu, U2D, U3D, V2D, make_u3d};
pub use renderer::{Prepared2D, PreparedMesh, Renderer};
