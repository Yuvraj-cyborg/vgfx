//! Vector maths under one rule: on the way to a pixel, only +, -, *, /, sqrt,
//! floor and abs on f32. IEEE 754 defines those exactly, so the same input
//! gives the same bits natively and in WebAssembly. The platform's sin, cos,
//! tan, exp, ln and pow are never called; `sin` and `cos` here are polynomials.
//! `mul_add` is not used either, because a fused multiply-add rounds once
//! where a multiply and an add round twice.

use core::ops::{Add, AddAssign, Mul, Neg, Sub};

pub const PI: f32 = 3.141_592_7;
pub const TAU: f32 = 6.283_185_5;

/// Sine of an angle in radians. Error under 1e-6.
pub fn sin(x: f32) -> f32 {
    // Whole turns off, leaving [-0.5, 0.5) of a turn.
    let t = x * (1.0 / TAU);
    let t = t - (t + 0.5).floor();
    // Fold onto [-0.25, 0.25], where the series converges fast.
    let t = if t > 0.25 {
        0.5 - t
    } else if t < -0.25 {
        -0.5 - t
    } else {
        t
    };
    let a = t * TAU;
    let a2 = a * a;
    a * (1.0
        + a2 * (-1.0 / 6.0
            + a2 * (1.0 / 120.0
                + a2 * (-1.0 / 5040.0 + a2 * (1.0 / 362_880.0 + a2 * (-1.0 / 39_916_800.0))))))
}

pub fn cos(x: f32) -> f32 {
    sin(x + PI * 0.5)
}

pub fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 0 below `a`, 1 above `b`, a smooth S between.
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp01((x - a) / (b - a));
    t * t * (3.0 - 2.0 * t)
}

// ------------------------------------------------------------------ vectors

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

impl V3 {
    pub const ZERO: V3 = v3(0.0, 0.0, 0.0);
    pub const ONE: V3 = v3(1.0, 1.0, 1.0);
    pub const X: V3 = v3(1.0, 0.0, 0.0);
    pub const Y: V3 = v3(0.0, 1.0, 0.0);
    pub const Z: V3 = v3(0.0, 0.0, 1.0);

    pub fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: V3) -> V3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    /// Unit length; a zero vector comes back as zero.
    pub fn norm(self) -> V3 {
        let l = self.len();
        if l > 0.0 { self * (1.0 / l) } else { V3::ZERO }
    }
    pub fn lerp(self, o: V3, t: f32) -> V3 {
        self + (o - self) * t
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for V3 {
    fn add_assign(&mut self, o: V3) {
        *self = *self + o;
    }
}
impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<V3> for V3 {
    type Output = V3;
    fn mul(self, o: V3) -> V3 {
        v3(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}
impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        v3(-self.x, -self.y, -self.z)
    }
}

// -------------------------------------------------------------- quaternions

/// A rotation. Always unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENT: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// A turn of `angle` radians about `axis`, right-handed.
    pub fn axis_angle(axis: V3, angle: f32) -> Quat {
        let a = axis.norm();
        let s = sin(angle * 0.5);
        Quat {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: cos(angle * 0.5),
        }
    }
    pub fn rot_x(angle: f32) -> Quat {
        Quat::axis_angle(V3::X, angle)
    }
    pub fn rot_y(angle: f32) -> Quat {
        Quat::axis_angle(V3::Y, angle)
    }
    pub fn rot_z(angle: f32) -> Quat {
        Quat::axis_angle(V3::Z, angle)
    }

    /// `a.mul(b)` turns by `b` first, then by `a`.
    pub fn mul(self, b: Quat) -> Quat {
        let a = self;
        Quat {
            x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        }
    }

    pub fn rotate(self, v: V3) -> V3 {
        let u = v3(self.x, self.y, self.z);
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }

    pub fn normalized(self) -> Quat {
        let l = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if l > 0.0 {
            let k = 1.0 / l;
            Quat {
                x: self.x * k,
                y: self.y * k,
                z: self.z * k,
                w: self.w * k,
            }
        } else {
            Quat::IDENT
        }
    }

    /// The short way from `self` to `o`. A straight blend, renormalised: exact
    /// at both ends, and close enough to constant speed for turns under 90
    /// degrees. Split a larger turn into two keys.
    pub fn nlerp(self, o: Quat, t: f32) -> Quat {
        let d = self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w;
        let s = if d < 0.0 { -1.0 } else { 1.0 };
        Quat {
            x: self.x + (o.x * s - self.x) * t,
            y: self.y + (o.y * s - self.y) * t,
            z: self.z + (o.z * s - self.z) * t,
            w: self.w + (o.w * s - self.w) * t,
        }
        .normalized()
    }
}

/// The turn that carries x, y, z onto the three given unit axes.
pub fn quat_from_axes(x: V3, y: V3, z: V3) -> Quat {
    let tr = x.x + y.y + z.z;
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        Quat {
            w: 0.25 * s,
            x: (y.z - z.y) / s,
            y: (z.x - x.z) / s,
            z: (x.y - y.x) / s,
        }
    } else if x.x > y.y && x.x > z.z {
        let s = (1.0 + x.x - y.y - z.z).sqrt() * 2.0;
        Quat {
            w: (y.z - z.y) / s,
            x: 0.25 * s,
            y: (y.x + x.y) / s,
            z: (z.x + x.z) / s,
        }
    } else if y.y > z.z {
        let s = (1.0 + y.y - x.x - z.z).sqrt() * 2.0;
        Quat {
            w: (z.x - x.z) / s,
            x: (y.x + x.y) / s,
            y: 0.25 * s,
            z: (z.y + y.z) / s,
        }
    } else {
        let s = (1.0 + z.z - x.x - y.y).sqrt() * 2.0;
        Quat {
            w: (x.y - y.x) / s,
            x: (z.x + x.z) / s,
            y: (z.y + y.z) / s,
            z: 0.25 * s,
        }
    };
    q.normalized()
}

// ----------------------------------------------------------------- matrices

/// A 4 by 4 matrix, column-major: `m[col * 4 + row]`. Points are columns, so
/// `a.mul(b)` applies `b` first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct M4 {
    pub m: [f32; 16],
}

impl M4 {
    pub const IDENT: M4 = M4 {
        m: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };

    /// Scale by `s`, then turn by `r`, then move by `t`.
    pub fn trs(t: V3, r: Quat, s: f32) -> M4 {
        let x = r.rotate(V3::X) * s;
        let y = r.rotate(V3::Y) * s;
        let z = r.rotate(V3::Z) * s;
        M4 {
            m: [
                x.x, x.y, x.z, 0.0, y.x, y.y, y.z, 0.0, z.x, z.y, z.z, 0.0, t.x, t.y, t.z, 1.0,
            ],
        }
    }
    pub fn translate(t: V3) -> M4 {
        M4::trs(t, Quat::IDENT, 1.0)
    }
    pub fn rotate(r: Quat) -> M4 {
        M4::trs(V3::ZERO, r, 1.0)
    }
    /// A different scale on each axis. Normals of a mesh drawn through this
    /// are only right if the three are equal.
    pub fn scale(s: V3) -> M4 {
        M4 {
            m: [
                s.x, 0.0, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, 0.0, s.z, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn mul(self, b: M4) -> M4 {
        let a = &self.m;
        let b = &b.m;
        let mut o = [0.0f32; 16];
        for c in 0..4 {
            for r in 0..4 {
                o[c * 4 + r] = a[r] * b[c * 4]
                    + a[4 + r] * b[c * 4 + 1]
                    + a[8 + r] * b[c * 4 + 2]
                    + a[12 + r] * b[c * 4 + 3];
            }
        }
        M4 { m: o }
    }

    /// Where a point goes (w = 1, no perspective divide).
    pub fn point(&self, p: V3) -> V3 {
        let m = &self.m;
        v3(
            m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
            m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
        )
    }
    /// Where a direction goes (w = 0).
    pub fn dir(&self, d: V3) -> V3 {
        let m = &self.m;
        v3(
            m[0] * d.x + m[4] * d.y + m[8] * d.z,
            m[1] * d.x + m[5] * d.y + m[9] * d.z,
            m[2] * d.x + m[6] * d.y + m[10] * d.z,
        )
    }
    /// A point with its w, for projection matrices.
    pub fn point4(&self, p: V3) -> [f32; 4] {
        let m = &self.m;
        [
            m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
            m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
            m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15],
        ]
    }

    /// The inverse of a matrix whose last row is 0 0 0 1 (any turn, scale and
    /// move). A matrix that flattens space comes back as the identity.
    pub fn inverse_affine(&self) -> M4 {
        let m = &self.m;
        let (a, b, c) = (
            v3(m[0], m[1], m[2]),
            v3(m[4], m[5], m[6]),
            v3(m[8], m[9], m[10]),
        );
        let det = a.dot(b.cross(c));
        if det == 0.0 {
            return M4::IDENT;
        }
        let k = 1.0 / det;
        // Rows of the inverse of the 3 by 3 part.
        let r0 = b.cross(c) * k;
        let r1 = c.cross(a) * k;
        let r2 = a.cross(b) * k;
        let t = v3(m[12], m[13], m[14]);
        M4 {
            m: [
                r0.x,
                r1.x,
                r2.x,
                0.0,
                r0.y,
                r1.y,
                r2.y,
                0.0,
                r0.z,
                r1.z,
                r2.z,
                0.0,
                -r0.dot(t),
                -r1.dot(t),
                -r2.dot(t),
                1.0,
            ],
        }
    }

    /// World to view. The view looks down its own -z, with +y up and +x right.
    pub fn look_at(eye: V3, target: V3, up: V3) -> M4 {
        let f = (target - eye).norm();
        let s = f.cross(up).norm();
        let u = s.cross(f);
        M4 {
            m: [
                s.x,
                u.x,
                -f.x,
                0.0,
                s.y,
                u.y,
                -f.y,
                0.0,
                s.z,
                u.z,
                -f.z,
                0.0,
                -s.dot(eye),
                -u.dot(eye),
                f.dot(eye),
                1.0,
            ],
        }
    }

    /// View to clip. After the divide by w: x and y in [-1, 1] (y up), depth
    /// in [0, 1] from `near` to `far`. `tan_half_fov_y` is the tangent of half
    /// the vertical angle of view; `aspect` is width over height.
    pub fn perspective(tan_half_fov_y: f32, aspect: f32, near: f32, far: f32) -> M4 {
        let f = 1.0 / tan_half_fov_y;
        let mut m = [0.0f32; 16];
        m[0] = f / aspect;
        m[5] = f;
        m[10] = far / (near - far);
        m[11] = -1.0;
        m[14] = near * far / (near - far);
        M4 { m }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_is_close() {
        let mut worst = 0.0f32;
        let mut x = -20.0f32;
        while x < 20.0 {
            worst = worst
                .max((sin(x) - x.sin()).abs())
                .max((cos(x) - x.cos()).abs());
            x += 0.001;
        }
        assert!(worst < 3e-6, "{worst}");
    }

    #[test]
    fn inverse_undoes() {
        let m = M4::trs(
            v3(1.0, -2.0, 3.0),
            Quat::axis_angle(v3(1.0, 2.0, 0.5), 1.1),
            1.7,
        );
        let p = v3(0.3, 0.9, -4.0);
        let q = m.inverse_affine().point(m.point(p));
        assert!((p - q).len() < 1e-4);
    }

    #[test]
    fn camera_looks_down_minus_z() {
        let v = M4::look_at(v3(0.0, 0.0, 5.0), V3::ZERO, V3::Y);
        let p = v.point(V3::ZERO);
        assert!((p - v3(0.0, 0.0, -5.0)).len() < 1e-5);
        let c = M4::perspective(0.5, 1.0, 1.0, 100.0).point4(v3(0.0, 0.0, -1.0));
        assert!((c[2] / c[3]).abs() < 1e-5);
    }
}
