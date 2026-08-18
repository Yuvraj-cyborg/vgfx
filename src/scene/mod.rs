//! Immediate-mode-ish scene description: 2D vector items and 3D mesh items,
//! camera, materials.

pub mod camera;
pub mod mesh;
pub mod scene;

pub use camera::Camera;
pub use mesh::{Mesh, Vertex3D};
pub use scene::{Color, Material, Scene, TwoDItem, TwoDStyle, ThreeDItem};
