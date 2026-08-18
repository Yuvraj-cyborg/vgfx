/// Vector path primitives and tessellation (bezier flattening, fill
/// triangulation with hole support, stroke outline generation).
pub mod path;
pub mod shapes;
pub mod tessellate;

pub use path::{Path, PathCmd, Vec2};
pub use shapes::{Circle, Polygon, Rect, RoundedRectangle};
pub use tessellate::{StrokeGeometry, Triangles};
