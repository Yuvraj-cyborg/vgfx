use glam::Vec2;

/// 2D affine transform (row-major 2x3): `p' = A * p + t`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Affine {
    pub m: [[f32; 3]; 2],
}

impl Affine {
    pub const IDENTITY: Affine = Affine {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    };

    #[must_use]
    pub const fn new(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> Self {
        Affine { m: [[a, b, tx], [c, d, ty]] }
    }

    #[must_use]
    pub const fn translate(x: f32, y: f32) -> Self {
        Affine::new(1.0, 0.0, 0.0, 1.0, x, y)
    }

    #[must_use]
    pub const fn scale(sx: f32, sy: f32) -> Self {
        Affine::new(sx, 0.0, 0.0, sy, 0.0, 0.0)
    }

    #[must_use]
    pub fn rotate(rad: f32) -> Self {
        let (s, c) = rad.sin_cos();
        Affine::new(c, s, -s, c, 0.0, 0.0)
    }

    /// `self * other` (apply `other` first).
    #[must_use]
    pub fn mul(&self, other: &Affine) -> Affine {
        let a = self.m[0][0] * other.m[0][0] + self.m[0][1] * other.m[1][0];
        let b = self.m[0][0] * other.m[0][1] + self.m[0][1] * other.m[1][1];
        let c = self.m[1][0] * other.m[0][0] + self.m[1][1] * other.m[1][0];
        let d = self.m[1][0] * other.m[0][1] + self.m[1][1] * other.m[1][1];
        let tx = self.m[0][0] * other.m[0][2] + self.m[0][1] * other.m[1][2] + self.m[0][2];
        let ty = self.m[1][0] * other.m[0][2] + self.m[1][1] * other.m[1][2] + self.m[1][2];
        Affine::new(a, b, c, d, tx, ty)
    }

    #[must_use]
    pub fn transform_point(&self, p: Vec2) -> Vec2 {
        Vec2::new(
            self.m[0][0] * p.x + self.m[0][1] * p.y + self.m[0][2],
            self.m[1][0] * p.x + self.m[1][1] * p.y + self.m[1][2],
        )
    }

    /// Average scale factor (used to scale stroke widths under transforms).
    #[must_use]
    pub fn avg_scale(&self) -> f32 {
        let sx = (self.m[0][0] * self.m[0][0] + self.m[1][0] * self.m[1][0]).sqrt();
        let sy = (self.m[0][1] * self.m[0][1] + self.m[1][1] * self.m[1][1]).sqrt();
        (sx + sy) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_point_composes() {
        let t = Affine::translate(10.0, 20.0);
        let s = Affine::scale(2.0, 2.0);
        let ts = t.mul(&s);
        let p = ts.transform_point(Vec2::new(1.0, 1.0));
        assert!((p.x - 12.0).abs() < 1e-5);
        assert!((p.y - 22.0).abs() < 1e-5);
    }

    #[test]
    fn rotation_keeps_length() {
        // y-down (screen) convention: +90deg turns +x (right) into -y (up).
        let r = Affine::rotate(std::f32::consts::PI / 2.0);
        let p = r.transform_point(Vec2::new(1.0, 0.0));
        assert!((p.x - 0.0).abs() < 1e-5);
        assert!((p.y + 1.0).abs() < 1e-5);
    }
}
