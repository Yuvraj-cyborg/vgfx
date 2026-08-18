/// Vector path primitives and tessellation (bezier flattening, fill
/// triangulation with hole support, stroke outline generation).
pub mod affine;
pub mod path;
pub mod shapes;
pub mod tessellate;

pub use affine::Affine;
pub use path::{Path, PathCmd};
pub use shapes::{circle, ellipse, polygon, rect, regular_polygon, rounded_rect};
pub use tessellate::{fill, fill_polys, flatten, stroke, Subpath, Triangles, DEFAULT_TOLERANCE};
