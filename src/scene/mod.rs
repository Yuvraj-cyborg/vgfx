//! Scene description: colors, materials, camera, meshes and the
//! per-frame item list consumed by a backend.

pub mod camera;
pub mod mesh;
pub mod scene;

pub use camera::Camera;
pub use mesh::{Mesh, Vertex3D};
pub use scene::{Color, Item, Material, Scene, ThreeDItem, TwoDItem, TwoDStyle};
