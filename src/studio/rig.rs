//! The rig: keyed tracks, a tree of joints, and its evaluation at a time.
//! Nothing here remembers anything between calls.

use super::math::{M4, Quat, V3, cos, sin, v3};
use super::scene::Mesh;
use super::shapes::add;

pub const DEG: f32 = 0.017_453_292;

/// How a value travels from the key before to this key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ease {
    Lin,
    /// Slow out, slow in.
    Smooth,
    /// Slower out and in than `Smooth`: for heavy things.
    Heavy,
    /// Starts slowly, arrives at speed.
    In,
    /// Leaves at speed, arrives slowly.
    Out,
    /// Passes the mark by a tenth and comes back.
    Back,
    /// A quick move, a small overshoot and a ring: machinery locking.
    Lock,
    /// Gathers speed gently, runs a little past the mark and rings: a heavy
    /// part finding its stop.
    Clunk,
    /// Falls, hits, bounces twice.
    Drop,
}

impl Ease {
    pub fn from_name(name: &str) -> Option<Ease> {
        Some(match name {
            "linear" | "lin" => Ease::Lin,
            "smooth" | "move" => Ease::Smooth,
            "heavy" => Ease::Heavy,
            "in" | "exit" => Ease::In,
            "out" | "enter" => Ease::Out,
            "back" | "bounce" => Ease::Back,
            "lock" => Ease::Lock,
            "clunk" => Ease::Clunk,
            "drop" => Ease::Drop,
            _ => return None,
        })
    }
}

pub fn ease(e: Ease, u: f32) -> f32 {
    match e {
        Ease::Lin => u,
        Ease::Smooth => u * u * (3.0 - 2.0 * t_clamp(u)),
        Ease::Heavy => u * u * u * (u * (6.0 * u - 15.0) + 10.0),
        Ease::In => u * u * u,
        Ease::Out => {
            let v = 1.0 - u;
            1.0 - v * v * v
        }
        Ease::Back => {
            let v = u - 1.0;
            1.0 + 2.70158 * v * v * v + 1.70158 * v * v
        }
        Ease::Lock => {
            let s = u * (0.25 + 0.75 * u);
            let v = 1.0 - s;
            1.0 - v * v * v * cos(5.2 * s)
        }
        Ease::Clunk => {
            let s = u * u * (1.6 - 0.6 * u);
            let v = 1.0 - s;
            1.0 - v * v * v * v * cos(5.2 * s)
        }
        Ease::Drop => {
            // Free fall to the mark at u = 0.4, then two shrinking hops.
            if u < 0.4 {
                let a = u / 0.4;
                a * a
            } else if u < 0.75 {
                let a = (u - 0.4) / 0.35;
                1.0 - 0.22 * 4.0 * a * (1.0 - a)
            } else {
                let a = (u - 0.75) / 0.25;
                1.0 - 0.06 * 4.0 * a * (1.0 - a)
            }
        }
    }
}

#[inline(always)]
fn t_clamp(u: f32) -> f32 {
    u
}

/// One number over time.
#[derive(Clone, Debug, Default)]
pub struct Track {
    pub keys: Vec<(f32, f32, Ease)>,
}

impl Track {
    pub fn new(keys: &[(f32, f32, Ease)]) -> Track {
        Track {
            keys: keys.to_vec(),
        }
    }
    pub fn constant(v: f32) -> Track {
        Track {
            keys: vec![(0.0, v, Ease::Lin)],
        }
    }
    pub fn at(&self, t: f32) -> f32 {
        let k = &self.keys;
        if k.is_empty() {
            return 0.0;
        }
        if t <= k[0].0 {
            return k[0].1;
        }
        for i in 1..k.len() {
            if t < k[i].0 {
                let span = k[i].0 - k[i - 1].0;
                if span <= 1e-6 {
                    return k[i].1;
                }
                let u = (t - k[i - 1].0) / span;
                return k[i - 1].1 + (k[i].1 - k[i - 1].1) * ease(k[i].2, u);
            }
        }
        k[k.len() - 1].1
    }
}

/// The six things a joint can do: slide along x, y, z and turn about them
/// (degrees). Slides are measured from the joint's rest place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ch {
    Px = 0,
    Py = 1,
    Pz = 2,
    Rx = 3,
    Ry = 4,
    Rz = 5,
}

impl Ch {
    pub fn from_name(name: &str) -> Option<Ch> {
        Some(match name {
            "px" | "x" => Ch::Px,
            "py" | "y" => Ch::Py,
            "pz" | "z" => Ch::Pz,
            "rx" => Ch::Rx,
            "ry" => Ch::Ry,
            "rz" => Ch::Rz,
            _ => return None,
        })
    }
}

/// A second parent a joint can be handed to. At weight 0 the joint hangs from
/// its own parent, at 1 from this one, at `pos` and `rot` in its frame.
#[derive(Clone, Debug)]
pub struct Alt {
    pub parent: usize,
    pub pos: V3,
    pub rot: Quat,
    pub w: Track,
}

#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    pub parent: usize,
    /// Rest place and turn in the parent's frame.
    pub rest: V3,
    pub pre: Quat,
    pub ch: [Track; 6],
    pub alt: Option<Alt>,
}

pub const NONE: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Xf {
    pub r: Quat,
    pub p: V3,
}

impl Xf {
    pub const IDENT: Xf = Xf {
        r: Quat::IDENT,
        p: V3::ZERO,
    };
    pub fn mul(self, o: Xf) -> Xf {
        Xf {
            r: self.r.mul(o.r).normalized(),
            p: self.p + self.r.rotate(o.p),
        }
    }
    pub fn point(self, p: V3) -> V3 {
        self.p + self.r.rotate(p)
    }
    pub fn inv(self) -> Xf {
        let r = Quat {
            x: -self.r.x,
            y: -self.r.y,
            z: -self.r.z,
            w: self.r.w,
        };
        Xf {
            r,
            p: -r.rotate(self.p),
        }
    }
    pub fn m4(self) -> M4 {
        M4::trs(self.p, self.r, 1.0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Rig {
    pub joints: Vec<Joint>,
    /// Each joint's frame with the robot standing at rest, and its inverse.
    pub bind: Vec<Xf>,
    pub unbind: Vec<Xf>,
}

impl Rig {
    /// Adds a joint whose rest place is `at` in the standing robot's own
    /// space (when `pre` turns along the chain are all identity) or, with
    /// `local` set, in its parent's frame.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        parent: usize,
        at: V3,
        pre: Quat,
        local: bool,
    ) -> usize {
        let pb = if parent == NONE {
            Xf::IDENT
        } else {
            self.bind[parent]
        };
        let rest = if local { at } else { pb.inv().point(at) };
        let b = pb.mul(Xf { r: pre, p: rest });
        self.joints.push(Joint {
            name: name.into(),
            parent,
            rest,
            pre,
            ch: Default::default(),
            alt: None,
        });
        self.bind.push(b);
        self.unbind.push(b.inv());
        self.joints.len() - 1
    }

    pub fn key(&mut self, j: usize, ch: Ch, keys: &[(f32, f32, Ease)]) {
        self.joints[j].ch[ch as usize] = Track::new(keys);
    }

    /// The same keys on a left joint and its right twin, mirrored.
    pub fn key2(&mut self, j: [usize; 2], ch: Ch, keys: &[(f32, f32, Ease)]) {
        self.key(j[0], ch, keys);
        let flip = matches!(ch, Ch::Px | Ch::Ry | Ch::Rz);
        let k: Vec<(f32, f32, Ease)> = keys
            .iter()
            .map(|&(t, v, e)| (t, if flip { -v } else { v }, e))
            .collect();
        self.key(j[1], ch, &k);
    }

    /// Each joint's own move and turn at `t`, before the tree is walked.
    pub fn local(&self, t: f32, out: &mut Vec<Xf>) {
        out.clear();
        for j in &self.joints {
            let c = |i: usize| j.ch[i].at(t);
            let p = j.rest + v3(c(0), c(1), c(2));
            let (rx, ry, rz) = (c(3), c(4), c(5));
            let mut r = j.pre;
            if rx != 0.0 {
                r = r.mul(Quat::rot_x(rx * DEG));
            }
            if rz != 0.0 {
                r = r.mul(Quat::rot_z(rz * DEG));
            }
            if ry != 0.0 {
                r = r.mul(Quat::rot_y(ry * DEG));
            }
            out.push(Xf { r, p });
        }
    }

    /// Walks the tree: `local` in, world frames out. Parents come before
    /// their children in the list, second parents too.
    pub fn world(&self, t: f32, local: &[Xf], out: &mut Vec<Xf>) {
        out.clear();
        for (i, j) in self.joints.iter().enumerate() {
            let mut w = if j.parent == NONE {
                local[i]
            } else {
                out[j.parent].mul(local[i])
            };
            if let Some(a) = &j.alt {
                let k = a.w.at(t);
                if k > 0.0 {
                    let b = out[a.parent].mul(Xf { r: a.rot, p: a.pos });
                    w = if k >= 1.0 {
                        b
                    } else {
                        Xf {
                            r: w.r.nlerp(b.r, k),
                            p: w.p.lerp(b.p, k),
                        }
                    };
                }
            }
            out.push(w);
        }
    }
}

/// Collects shapes into one mesh for each joint and material. A shape drawn
/// for the left side is reflected for the right.
pub struct Bucket {
    pub joint: usize,
    pub mat: u32,
    pub mesh: Mesh,
}

pub struct Parts<'a> {
    pub rig: &'a Rig,
    pub buckets: Vec<Bucket>,
    /// Building the right-hand twin: reflect everything in x = 0.
    pub right: bool,
}

impl<'a> Parts<'a> {
    pub fn new(rig: &'a Rig) -> Parts<'a> {
        Parts {
            rig,
            buckets: Vec::new(),
            right: false,
        }
    }
    fn slot(&mut self, joint: usize, mat: u32) -> &mut Mesh {
        let i = match self
            .buckets
            .iter()
            .position(|b| b.joint == joint && b.mat == mat)
        {
            Some(i) => i,
            None => {
                self.buckets.push(Bucket {
                    joint,
                    mat,
                    mesh: Mesh::default(),
                });
                self.buckets.len() - 1
            }
        };
        &mut self.buckets[i].mesh
    }
    /// Which of a pair is being built.
    pub fn s(&self, pair: [usize; 2]) -> usize {
        pair[self.right as usize]
    }
    /// A shape given in the standing robot's space, carried by joint `j`.
    pub fn w(&mut self, j: usize, mat: u32, m: Mesh) {
        let mut tmp = Mesh::default();
        add(&mut tmp, &m, &M4::IDENT, self.right);
        let xf = self.rig.unbind[j].m4();
        add(self.slot(j, mat), &tmp, &xf, false);
    }
    /// A shape given in joint `j`'s own frame.
    pub fn l(&mut self, j: usize, mat: u32, m: Mesh) {
        let right = self.right;
        add(self.slot(j, mat), &m, &M4::IDENT, right);
    }
}

/// The path a track runs along: a belt drawn tight round the wheels, walked
/// once into a table so that a link's place is a look-up and a blend.
///
/// Track frame: x out from the hull, y up from the ground, z forward.
#[derive(Clone, Debug)]
pub struct Belt {
    table: Vec<[f32; 4]>,
    ds: f32,
    pub len: f32,
}

impl Belt {
    /// Builds a closed belt wrapped around the given wheels `[z, y, radius + half_link_thickness]`.
    pub fn from_wheels(c: &[[f32; 3]]) -> Belt {
        let n = c.len();
        if n < 2 {
            return Belt {
                table: vec![[0.0, 0.0, 1.0, 0.0]; 2],
                ds: 1.0,
                len: 1.0,
            };
        }
        let out: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let (a, b) = (c[i], c[(i + 1) % n]);
                let (du, dv) = (b[0] - a[0], b[1] - a[1]);
                let l = (du * du + dv * dv).sqrt().max(1e-6);
                let (cu, cv) = (du / l, dv / l);
                let k = (-(b[2] - a[2]) / l).clamp(-0.999, 0.999);
                let s = (1.0 - k * k).sqrt();
                [k * cu + s * cv, k * cv - s * cu]
            })
            .collect();
        let mut pts: Vec<[f32; 2]> = Vec::new();
        let (cd, sd) = (cos(0.02), sin(0.02));
        for i in 0..n {
            let (n0, n1) = (out[(i + n - 1) % n], out[i]);
            let w = c[i];
            let mut m = n0;
            pts.push([w[0] + w[2] * m[0], w[1] + w[2] * m[1]]);
            for _ in 0..400 {
                if m[0] * n1[1] - m[1] * n1[0] <= 1e-4 {
                    break;
                }
                let r = [m[0] * cd - m[1] * sd, m[0] * sd + m[1] * cd];
                if r[0] * n1[1] - r[1] * n1[0] < 0.0 {
                    break;
                }
                m = r;
                pts.push([w[0] + w[2] * m[0], w[1] + w[2] * m[1]]);
            }
            pts.push([w[0] + w[2] * n1[0], w[1] + w[2] * n1[1]]);
        }
        let m = pts.len();
        let mut start = 0;
        for i in 0..m {
            let (a, b) = (pts[i], pts[(i + 1) % m]);
            if a[0] <= 0.0 && b[0] > 0.0 && a[1] < 0.2 {
                start = i;
            }
        }
        let (a, b) = (pts[start], pts[(start + 1) % m]);
        let f = if (b[0] - a[0]).abs() > 1e-6 {
            -a[0] / (b[0] - a[0])
        } else {
            0.0
        };
        let mut poly = vec![[0.0, a[1] + (b[1] - a[1]) * f]];
        for k in 1..=m {
            poly.push(pts[(start + k) % m]);
        }
        poly.push(poly[0]);
        let mut acc = vec![0.0f32];
        for i in 1..poly.len() {
            let (du, dv) = (poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]);
            acc.push(acc[i - 1] + (du * du + dv * dv).sqrt());
        }
        let len = acc[acc.len() - 1].max(1e-4);
        let steps = 2600usize;
        let ds = len / steps as f32;
        let mut table = Vec::with_capacity(steps + 1);
        let mut seg = 1;
        for k in 0..=steps {
            let s = (k as f32 * ds).min(len);
            while seg + 1 < poly.len() && (acc[seg] < s || acc[seg] - acc[seg - 1] < 1e-7) {
                seg += 1;
            }
            let l = (acc[seg] - acc[seg - 1]).max(1e-7);
            let f = ((s - acc[seg - 1]) / l).clamp(0.0, 1.0);
            let (a, b) = (poly[seg - 1], poly[seg]);
            table.push([
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                (b[0] - a[0]) / l,
                (b[1] - a[1]) / l,
            ]);
        }
        Belt { table, ds, len }
    }

    /// Where a link sits, `s` along the loop, in the track's frame.
    pub fn at(&self, s: f32) -> M4 {
        let s = s - (s / self.len).floor() * self.len;
        let x = s / self.ds;
        let i = (x.floor() as usize).min(self.table.len() - 2);
        let f = x - i as f32;
        let (a, b) = (self.table[i], self.table[i + 1]);
        let mix = |k: usize| a[k] + (b[k] - a[k]) * f;
        let (z, y, mut tz, mut ty) = (mix(0), mix(1), mix(2), mix(3));
        let l = (tz * tz + ty * ty).sqrt();
        tz /= l;
        ty /= l;
        M4 {
            m: [
                -1.0, 0.0, 0.0, 0.0, 0.0, -tz, ty, 0.0, 0.0, ty, tz, 0.0, 0.0, y, z, 1.0,
            ],
        }
    }

    /// Where along z a link at `s` sits (for sorting links into halves).
    pub fn z_at(&self, s: f32) -> f32 {
        self.at(s).m[14]
    }
}
