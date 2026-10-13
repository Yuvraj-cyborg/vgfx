//! What a frame is. The film fills one in for a time; the renderer draws it.
//!
//! Space: right-handed, +y up, the floor is the plane y = 0. One unit is about
//! a metre. The film keeps everything within about ten units of the origin.
//!
//! Colour: linear light, red green blue. 1.0 is paper white under the key
//! light; lamps and glowing parts go above it.

use super::math::{M4, V3};

/// Triangles. Every mesh is a closed solid whose triangles wind
/// counter-clockwise seen from outside, so the renderer may skip the ones
/// that face away. `nrm[i]` is the unit normal at `pos[i]`; a hard edge is two
/// vertices in the same place with different normals.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub pos: Vec<V3>,
    pub nrm: Vec<V3>,
    pub idx: Vec<u32>,
}

/// What a surface is made of.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Its colour: for paint and plastic the colour of the diffuse light it
    /// returns, for bare metal the colour of its reflection.
    pub base: [f32; 3],
    /// 0 for paint and plastic, 1 for bare metal.
    pub metal: f32,
    /// 0 a mirror, 1 chalk.
    pub rough: f32,
    /// Light it gives off. May exceed 1; bright values bloom.
    pub emit: [f32; 3],
    /// Spacing, in the mesh's own units, of fine machined lines across the
    /// surface. 0 for none. A renderer may ignore it.
    pub groove: f32,
    /// How much fine variation the finish has, 0 to 1. A renderer may ignore it.
    pub grain: f32,
}

impl Material {
    pub const fn paint(base: [f32; 3], rough: f32) -> Material {
        Material {
            base,
            metal: 0.0,
            rough,
            emit: [0.0; 3],
            groove: 0.0,
            grain: 0.0,
        }
    }
    pub const fn metal(base: [f32; 3], rough: f32) -> Material {
        Material {
            base,
            metal: 1.0,
            rough,
            emit: [0.0; 3],
            groove: 0.0,
            grain: 0.0,
        }
    }
    pub const fn lamp(emit: [f32; 3]) -> Material {
        Material {
            base: [0.0; 3],
            metal: 0.0,
            rough: 1.0,
            emit,
            groove: 0.0,
            grain: 0.0,
        }
    }
}

/// One mesh, placed.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    /// Index into `Film::meshes`.
    pub mesh: u32,
    /// Index into `Film::materials`.
    pub material: u32,
    /// Mesh space to world: a turn, a move, and at most one scale for all
    /// three axes.
    pub xf: M4,
    /// Multiplies the material's `emit`. 1 leaves it alone.
    pub glow: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: V3,
    pub target: V3,
    pub up: V3,
    /// Tangent of half the vertical angle of view. 0.4 is a normal lens,
    /// 0.25 a long one, 0.6 a wide one.
    pub tan_half_fov_y: f32,
    pub near: f32,
    pub far: f32,
}

/// A lamp at `pos`, aimed at `at`.
#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub pos: V3,
    pub at: V3,
    /// Colour and strength together: the light arriving on a surface that
    /// faces the lamp, whatever the distance.
    pub color: [f32; 3],
    /// The lamp's radius. Larger gives softer shadow edges and wider, dimmer
    /// reflections in polished metal.
    pub size: f32,
}

/// The ground: the whole plane y = 0.
#[derive(Clone, Copy, Debug)]
pub struct Floor {
    pub base: [f32; 3],
    pub rough: f32,
    /// How strongly it mirrors what stands on it, 0 to 1.
    pub mirror: f32,
}

/// Everything in one picture.
#[derive(Clone, Debug)]
pub struct Frame {
    pub instances: Vec<Instance>,
    pub camera: Camera,
    /// The one lamp that casts shadows.
    pub key: Light,
    /// Up to three more lamps that do not: fill, rim, kicker.
    pub fills: Vec<Light>,
    /// The room's own tone: what a surface gets from every direction.
    pub ambient: [f32; 3],
    pub floor: Floor,
    /// Multiplies everything before the tone curve.
    pub exposure: f32,
    /// How far bright things bleed, 0 to 1.
    pub bloom: f32,
}

impl Default for Frame {
    fn default() -> Frame {
        Frame {
            instances: Vec::new(),
            camera: Camera {
                eye: V3 {
                    x: 0.0,
                    y: 2.0,
                    z: 9.0,
                },
                target: V3 {
                    x: 0.0,
                    y: 1.5,
                    z: 0.0,
                },
                up: V3::Y,
                tan_half_fov_y: 0.4,
                near: 0.1,
                far: 200.0,
            },
            key: Light {
                pos: V3 {
                    x: 4.0,
                    y: 9.0,
                    z: 6.0,
                },
                at: V3::ZERO,
                color: [1.0, 1.0, 1.0],
                size: 0.8,
            },
            fills: Vec::new(),
            ambient: [0.05, 0.05, 0.06],
            floor: Floor {
                base: [0.03, 0.03, 0.035],
                rough: 0.3,
                mirror: 0.4,
            },
            exposure: 1.0,
            bloom: 0.5,
        }
    }
}

/// A film: fixed meshes and materials, and a frame for any time. `frame` must
/// be a pure function of `t`: the same time gives the same frame, whatever was
/// asked before, so frames can be drawn in any order on any thread.
pub trait Film {
    fn meshes(&self) -> &[Mesh];
    fn materials(&self) -> &[Material];
    /// Length in seconds.
    fn seconds(&self) -> f32;
    /// Fill `out` for time `t` in seconds. `out` arrives holding some earlier
    /// frame; clear what you fill.
    fn frame(&self, t: f32, out: &mut Frame);
}
