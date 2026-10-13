//! What a point on a surface sends to the eye.
//!
//! Metal and paint by the usual microfacet model (GGX), with no `pow`: every
//! power here is a product. Each lamp is a ball of its `size`: its highlight
//! is taken at the point of the ball nearest the mirror direction, and widened
//! and dimmed by the angle the ball fills, so polished metal shows a soft
//! bright disc and brushed metal a broad sheen. Around the lamps is a dark
//! studio: a bank of soft light overhead, dim walls, and the floor below with
//! the key lamp's pool on it.

use super::light::Light;
use super::raster::Tri;
use super::{clamp01, fmax, fmin};
use crate::studio::math::{PI, V3};
use crate::studio::scene::Material;

pub struct Mat {
    /// Colour of the diffuse light it returns.
    pub cd: [f32; 3],
    /// Mirror reflection facing the eye.
    pub f0: [f32; 3],
    pub rough: f32,
    pub emit: [f32; 3],
    pub groove: f32,
    pub grain: f32,
}

impl Mat {
    pub fn of(m: &Material) -> Mat {
        let metal = clamp01(m.metal);
        let mut cd = [0.0; 3];
        let mut f0 = [0.0; 3];
        for c in 0..3 {
            let b = fmax(m.base[c], 0.0);
            cd[c] = b * (1.0 - metal);
            f0[c] = 0.04 + (b - 0.04) * metal;
        }
        Mat {
            cd,
            f0,
            rough: fmin(fmax(m.rough, 0.045), 1.0),
            emit: m.emit,
            groove: m.groove,
            grain: clamp01(m.grain),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Lamp {
    pub pos: V3,
    pub col: [f32; 3],
    pub size: f32,
}

pub struct Inst {
    pub mat: u32,
    pub glow: f32,
}

pub struct Ctx<'a> {
    pub eye: V3,
    /// The world direction through pixel (x, y), scaled so that a point at
    /// depth w is `eye + (ray0 + rdx * x + rdy * y) * w`.
    pub ray0: V3,
    pub rdx: V3,
    pub rdy: V3,
    /// The key lamp first.
    pub lamps: [Lamp; 4],
    pub nl: usize,
    pub amb: [f32; 3],
    pub floor_base: [f32; 3],
    /// How strongly the floor mirrors, and how rough it is.
    pub floor_mirror: f32,
    pub floor_rough: f32,
    pub light: &'a Light,
    pub vn: &'a [V3],
    pub mats: &'a [Mat],
    pub insts: &'a [Inst],
    pub e2: &'a [f32; 65],
}

#[inline(always)]
fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

#[inline(always)]
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

#[inline(always)]
fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp01((x - a) / (b - a));
    t * t * (3.0 - 2.0 * t)
}

/// 2 to the power -9.28 x for x in 0..1, at 65 points: each is the one
/// before times a constant.
pub fn exp2_table() -> [f32; 65] {
    let mut t = [1.0f32; 65];
    for i in 1..65 {
        t[i] = t[i - 1] * 0.904_379_7;
    }
    t
}

pub struct Surf {
    pub p: V3,
    pub n: V3,
    pub w: f32,
}

/// The point of triangle `t` seen at screen position (sx, sy).
#[inline(always)]
pub fn surf(cx: &Ctx, t: &Tri, sx: f32, sy: f32) -> Surf {
    let (dx, dy) = (sx - t.x0, sy - t.y0);
    let w = 1.0 / (t.q[0] + dx * t.q[1] + dy * t.q[2]);
    let n0 = cx.vn[t.vi[0] as usize];
    let n = if t.nvar > 0.0 {
        let b1 = (t.u[0] + dx * t.u[1] + dy * t.u[2]) * w;
        let b2 = (t.v[0] + dx * t.v[1] + dy * t.v[2]) * w;
        let (n1, n2) = (cx.vn[t.vi[1] as usize], cx.vn[t.vi[2] as usize]);
        let n = n0 + (n1 - n0) * b1 + (n2 - n0) * b2;
        n * rsqrt(fmax(n.dot(n), 1e-12))
    } else {
        n0
    };
    Surf {
        p: cx.eye + (cx.ray0 + cx.rdx * sx + cx.rdy * sy) * w,
        n,
        w,
    }
}

/// The studio's walls and the bank of light overhead, as a multiple of the
/// room's tone, for a direction with height `y`, seen in a surface of
/// roughness `rough`.
#[inline(always)]
pub(crate) fn sky(y: f32, rough: f32) -> f32 {
    let t = clamp01(y);
    0.22 + 0.30 * t + 2.2 * smooth(0.50 - 0.35 * rough, 0.92 + 0.08 * rough, t)
}

/// Four tall soft panels stand round the room: two in front, to either side
/// of where a camera would be, and two behind for the rims. (x, z) of the
/// way to each, and how bright it is against the room's tone.
const PANELS: [(f32, f32, f32); 4] = [
    (0.766, 0.643, 18.0),
    (-0.819, 0.574, 13.0),
    (-0.643, -0.766, 22.0),
    (0.940, -0.342, 15.0),
];

/// The whole studio in direction `r`: walls, overhead bank and panels.
#[inline(always)]
fn studio(r: V3, rough: f32) -> f32 {
    let mut s = sky(r.y, rough);
    let wd = 0.26 + 0.55 * rough;
    let iw = 1.0 / wd;
    // Up the wall from just above the floor, and fading out toward the top.
    let ht = smooth(-0.02 - 0.25 * rough, 0.10 + 0.25 * rough, r.y)
        * (1.0 - smooth(0.45 - 0.2 * rough, 0.80 + 0.1 * rough, r.y));
    if ht > 0.0 {
        for &(px, pz, gain) in &PANELS {
            let toward = r.x * px + r.z * pz;
            if toward > 0.0 {
                let across = (r.x * pz - r.z * px) * iw;
                let a = 1.0 - across * across;
                if a > 0.0 {
                    s += gain * 0.26 * iw * a * a * ht * fmin(toward * 4.0, 1.0);
                }
            }
        }
    }
    s
}

/// What the room sends from direction `r` to a point `p`.
#[inline(always)]
fn env(cx: &Ctx, p: V3, r: V3, rough: f32) -> [f32; 3] {
    let wd = 0.03 + 0.35 * rough;
    let up = smooth(-wd, wd, r.y);
    let mut o = [0.0f32; 3];
    if up > 0.0 {
        let s = studio(r, rough) * up;
        for c in 0..3 {
            o[c] = cx.amb[c] * s;
        }
    }
    if up < 1.0 && p.y > 0.0 {
        // The floor, where this direction meets it, under the key lamp.
        let x = p + r * (p.y / fmax(-r.y, 0.05));
        let (u, v, _) = cx.light.project(x);
        let k = Light::pool(u, v) * 0.7 * (1.0 - up);
        let key = &cx.lamps[0].col;
        // A glossy floor also mirrors the studio: a face that looks down
        // at it sees the panels and the bank overhead in it.
        let m = cx.floor_mirror * (0.45 + 0.55 * pow5(1.0 - clamp01(-r.y))) * (1.0 - up);
        let s = if m > 0.0 {
            studio(
                V3 {
                    x: r.x,
                    y: -r.y,
                    z: r.z,
                },
                fmax(rough, cx.floor_rough),
            ) * m
        } else {
            0.0
        };
        for c in 0..3 {
            o[c] += cx.floor_base[c] * (key[c] * k + cx.amb[c] * (1.0 - up)) + cx.amb[c] * s;
        }
    }
    o
}

/// The mirror-like part of one lamp's light, per unit of its colour.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lamp_spec(
    n: V3,
    v: V3,
    r: V3,
    nv: f32,
    lv: V3,
    inv: f32,
    size: f32,
    a: f32,
    a2: f32,
    f0: &[f32; 3],
) -> [f32; 3] {
    // The point of the ball nearest the mirror direction.
    let c2r = r * lv.dot(r) - lv;
    let c2 = c2r.dot(c2r);
    let t = if c2 > size * size {
        size * rsqrt(c2)
    } else {
        1.0
    };
    let lr = lv + c2r * t;
    let l = lr * rsqrt(lr.dot(lr));
    let nl = n.dot(l);
    if nl <= 0.0 {
        return [0.0; 3];
    }
    let hv = l + v;
    let h = hv * rsqrt(fmax(hv.dot(hv), 1e-12));
    let nh = clamp01(n.dot(h));
    let vh = clamp01(v.dot(h));
    // A ball that fills an angle spreads its highlight and dims it.
    let ap = fmin(a + 0.5 * size * inv, 1.0);
    let norm = (a / ap) * (a / ap);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd) * norm;
    let vis = 0.5 / (2.0 * nl * nv + (nl + nv - 2.0 * nl * nv) * a + 1e-5);
    let k = d * vis * nl * PI;
    let fc = pow5(1.0 - vh);
    [
        (f0[0] + (1.0 - f0[0]) * fc) * k,
        (f0[1] + (1.0 - f0[1]) * fc) * k,
        (f0[2] + (1.0 - f0[2]) * fc) * k,
    ]
}

/// The light leaving a surface point toward the eye. `ao` is how open the
/// point is to the room, 0 to 1.
pub fn shade(cx: &Ctx, sf: &Surf, t: &Tri, ao: f32) -> [f32; 3] {
    let inst = &cx.insts[t.inst as usize];
    let m = &cx.mats[inst.mat as usize];
    let p = sf.p;
    let vv = cx.eye - p;
    let v = vv * rsqrt(vv.dot(vv));
    let mut n = sf.n;
    let mut nv = n.dot(v);
    if nv < 0.02 {
        // A shading normal that leans away from the eye: stand it back up.
        n = n + v * (0.02 - nv);
        n = n * rsqrt(n.dot(n));
        nv = fmax(n.dot(v), 0.02);
    }
    let r = n * (2.0 * nv) - v;
    let rough = m.rough;
    let mut a = rough * rough;
    let mut a2 = a * a;
    if t.nvar > 0.0 {
        // A normal that turns within a pixel spreads the highlight; say so,
        // or a thin bevel sparkles.
        a2 = fmin(a2 + fmin(0.5 * t.nvar, 0.18), 1.0);
        a = a2.sqrt();
    }
    let mut out = [0.0f32; 3];
    let fill_open = 0.3 + 0.7 * ao;
    for i in 0..cx.nl {
        let lamp = &cx.lamps[i];
        let lv = lamp.pos - p;
        let inv = rsqrt(lv.dot(lv));
        let nl = n.dot(lv) * inv;
        let mut e = fill_open;
        if i == 0 {
            let c = -t.ng.dot(lv);
            if c >= 0.0 {
                continue;
            }
            let (pu, pv, pq) = cx.light.project(p);
            let pool = Light::pool(pu, pv);
            if pool <= 0.0 {
                continue;
            }
            // Fade in over the first degrees past the edge of the lit side,
            // where a faceted round would show its facets.
            let face = clamp01(-c * inv * 10.0);
            e = pool * face * cx.light.shadow(pu, pv, pq, t.ng, c);
            if e <= 0.0 {
                continue;
            }
        }
        let s = lamp_spec(n, v, r, nv, lv, inv, lamp.size, a, a2, &m.f0);
        let d = fmax(nl, 0.0);
        for c in 0..3 {
            out[c] += lamp.col[c] * e * (m.cd[c] * d + s[c]);
        }
    }
    // The room: its tone on paint, its reflection in everything.
    // The same spreading holds for the room's reflection: where the normal
    // turns within a pixel, the room is seen as in a rougher surface.
    let rough = if t.nvar > 0.0 { a.sqrt() } else { rough };
    let hemi = (0.62 + 0.38 * n.y) * ao;
    let rf = rough * (-1.0) + 1.0;
    let (r0, r1, r2, r3) = (
        rf,
        rough * -0.0275 + 0.0425,
        rough * -0.572 + 1.04,
        rough * 0.022 - 0.04,
    );
    let x = nv * 64.0;
    let xi = (x as usize).min(63);
    let e2 = cx.e2[xi] + (cx.e2[xi + 1] - cx.e2[xi]) * (x - xi as f32);
    let a004 = fmin(r0 * r0, e2) * r0 + r1;
    let (ea, eb) = (-1.04 * a004 + r2, 1.04 * a004 + r3);
    let so = clamp01((nv + ao).sqrt() * (1.0 - rough) + rough - 1.0 + ao);
    let ev = env(cx, p, r, rough);
    for c in 0..3 {
        out[c] +=
            m.cd[c] * cx.amb[c] * hemi + ev[c] * (m.f0[c] * ea + eb) * so + m.emit[c] * inst.glow;
    }
    out
}
