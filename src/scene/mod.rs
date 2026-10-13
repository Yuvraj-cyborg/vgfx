//! Scene description: colors, materials, camera, meshes and the
//! per-frame item list consumed by a backend.

pub mod camera;
pub mod items;
pub mod mesh;

pub use camera::Camera;
pub use items::{Color, Item, Material, Scene, ThreeDItem, TwoDItem, TwoDStyle};
pub use mesh::{Mesh, Vertex3D};
