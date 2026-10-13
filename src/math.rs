//! Small, allocation-free math types. Matrices are column-major.
use std::ops::{Add, AddAssign, Div, Index, Mul, Neg, Sub, SubAssign};

macro_rules! vector {
    ($name:ident, $n:expr, $($f:ident),+) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name { $(pub $f: f32),+ }
        impl $name {
            pub const ZERO: Self = Self { $($f: 0.0),+ };
            pub const fn new($($f: f32),+) -> Self { Self { $($f),+ } }
            pub const fn splat(v: f32) -> Self { Self { $($f: v),+ } }
            pub fn dot(self, rhs: Self) -> f32 { 0.0 $(+ self.$f * rhs.$f)+ }
            pub fn length_squared(self) -> f32 { self.dot(self) }
            pub fn length(self) -> f32 { self.length_squared().sqrt() }
            pub fn normalize(self) -> Self { self / self.length() }
            pub fn distance_squared(self, rhs: Self) -> f32 { (self-rhs).length_squared() }
            pub fn distance(self, rhs: Self) -> f32 { (self-rhs).length() }
            pub fn to_array(self) -> [f32; $n] { [$(self.$f),+] }
        }
        impl From<[f32; $n]> for $name {
            fn from(v: [f32; $n]) -> Self { let [$($f),+] = v; Self { $($f),+ } }
        }
        impl Add for $name { type Output=Self; fn add(self,r:Self)->Self {Self{$($f:self.$f+r.$f),+}} }
        impl Sub for $name { type Output=Self; fn sub(self,r:Self)->Self {Self{$($f:self.$f-r.$f),+}} }
        impl Neg for $name { type Output=Self; fn neg(self)->Self {Self{$($f:-self.$f),+}} }
        impl Mul<f32> for $name { type Output=Self; fn mul(self,r:f32)->Self {Self{$($f:self.$f*r),+}} }
        impl Div<f32> for $name { type Output=Self; fn div(self,r:f32)->Self {Self{$($f:self.$f/r),+}} }
        impl AddAssign for $name {fn add_assign(&mut self,r:Self){*self=*self+r;}}
        impl SubAssign for $name {fn sub_assign(&mut self,r:Self){*self=*self-r;}}
        impl Index<usize> for $name {type Output=f32; fn index(&self,i:usize)->&f32 {let a=[$(&self.$f),+]; a[i]}}
    }
}
vector!(Vec2, 2, x, y);
vector!(Vec3, 3, x, y, z);
vector!(Vec4, 4, x, y, z, w);
impl Vec3 {
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub fn cross(self, r: Self) -> Self {
        Self::new(
            self.y * r.z - self.z * r.y,
            self.z * r.x - self.x * r.z,
            self.x * r.y - self.y * r.x,
        )
    }
}
impl From<(f32, f32, f32, f32)> for Vec4 {
    fn from(v: (f32, f32, f32, f32)) -> Self {
        Self::new(v.0, v.1, v.2, v.3)
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    cols: [f32; 16],
}
impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ],
    };
    pub const fn from_cols_array(cols: &[f32; 16]) -> Self {
        Self { cols: *cols }
    }
    pub fn from_cols_slice(cols: &[f32]) -> Self {
        let mut a = [0.; 16];
        a.copy_from_slice(&cols[..16]);
        Self { cols: a }
    }
    pub const fn to_cols_array(self) -> [f32; 16] {
        self.cols
    }
    pub fn from_translation(v: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[12] = v.x;
        m.cols[13] = v.y;
        m.cols[14] = v.z;
        m
    }
    pub fn from_scale(v: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[0] = v.x;
        m.cols[5] = v.y;
        m.cols[10] = v.z;
        m
    }
    pub fn from_rotation_y(a: f32) -> Self {
        let (s, c) = a.sin_cos();
        Self {
            cols: [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.],
        }
    }
    pub fn from_rotation_z(a: f32) -> Self {
        let (s, c) = a.sin_cos();
        Self {
            cols: [c, s, 0., 0., -s, c, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
        }
    }
    pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let z = (eye - target).normalize();
        let x = up.cross(z).normalize();
        let y = z.cross(x);
        Self {
            cols: [
                x.x,
                y.x,
                z.x,
                0.,
                x.y,
                y.y,
                z.y,
                0.,
                x.z,
                y.z,
                z.z,
                0.,
                -x.dot(eye),
                -y.dot(eye),
                -z.dot(eye),
                1.,
            ],
        }
    }
    pub fn perspective_rh(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov * 0.5).tan();
        let z = far / (near - far);
        Self {
            cols: [
                f / aspect,
                0.,
                0.,
                0.,
                0.,
                f,
                0.,
                0.,
                0.,
                0.,
                z,
                -1.,
                0.,
                0.,
                z * near,
                0.,
            ],
        }
    }
    pub fn perspective_rh_gl(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov * 0.5).tan();
        let z = 1.0 / (near - far);
        Self {
            cols: [
                f / aspect,
                0.,
                0.,
                0.,
                0.,
                f,
                0.,
                0.,
                0.,
                0.,
                (far + near) * z,
                -1.,
                0.,
                0.,
                2. * far * near * z,
                0.,
            ],
        }
    }
}
impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        let a = self.cols;
        Vec4::new(
            a[0] * v.x + a[4] * v.y + a[8] * v.z + a[12] * v.w,
            a[1] * v.x + a[5] * v.y + a[9] * v.z + a[13] * v.w,
            a[2] * v.x + a[6] * v.y + a[10] * v.z + a[14] * v.w,
            a[3] * v.x + a[7] * v.y + a[11] * v.z + a[15] * v.w,
        )
    }
}
impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        let mut a = [0.; 16];
        for c in 0..4 {
            let v = self
                * Vec4::from([
                    r.cols[c * 4],
                    r.cols[c * 4 + 1],
                    r.cols[c * 4 + 2],
                    r.cols[c * 4 + 3],
                ]);
            a[c * 4..c * 4 + 4].copy_from_slice(&v.to_array());
        }
        Self { cols: a }
    }
}
impl Mul<&Mat4> for Mat4 {
    type Output = Self;
    fn mul(self, r: &Self) -> Self {
        self * *r
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn composition_and_depth() {
        let m =
            Mat4::from_translation(Vec3::new(3., 4., 5.)) * Mat4::from_scale(Vec3::new(2., 3., 4.));
        assert_eq!(m * Vec4::new(1., 1., 1., 1.), Vec4::new(5., 7., 9., 1.));
        let p = Mat4::perspective_rh(1., 1., 0.1, 100.);
        for (z, want) in [(-0.1, 0.), (-100., 1.)] {
            let v = p * Vec4::new(0., 0., z, 1.);
            assert!((v.z / v.w - want).abs() < 1e-5);
        }
    }
}
