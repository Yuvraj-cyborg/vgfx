//! Deterministic CPU 3D studio engine: exact polynomial maths, parametric solid
//! builders, mechanical rigs, belts, hydraulic rams, and a 4x MSAA scanline
//! rasteriser with soft contact-hardening shadows, mirrored studio floor,
//! ambient occlusion, and bloom.
#![allow(
    clippy::approx_constant,
    clippy::manual_clamp,
    clippy::should_implement_trait,
    clippy::doc_overindented_list_items,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::neg_multiply,
    clippy::too_many_arguments
)]

pub mod forms;
pub mod math;
pub mod render;
pub mod rig;
pub mod scene;
pub mod shapes;
pub mod stage;
pub mod transformer;

pub use math::{M4, PI, Quat, TAU, V3, cos, quat_from_axes, sin, v3};
pub use render::Renderer as StudioRenderer;
pub use rig::{Alt, Belt, Ch, DEG, Ease, Joint, NONE, Parts, Rig, Track, Xf};
pub use scene::{Camera, Film, Floor, Frame, Instance, Light, Material, Mesh};
pub use stage::{CamTracks, CustomStage, LampSpec, Part3D, Place3D, Scratch, Stage};
pub use transformer::Transform;
