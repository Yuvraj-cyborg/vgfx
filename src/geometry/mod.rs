/// Vector path primitives and tessellation (bezier flattening, fill
/// triangulation with hole support, stroke outline generation).
pub mod affine;
pub mod path;
pub mod shapes;
pub mod tessellate;

pub use affine::Affine;
pub use path::{Path, PathCmd};
pub use shapes::{ellipse, rounded_rect};
pub use tessellate::{DEFAULT_TOLERANCE, Subpath, Triangles, fill, fill_polys, flatten, stroke};
