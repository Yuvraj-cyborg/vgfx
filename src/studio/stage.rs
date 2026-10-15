//! Declarative 3D stage that can hold any custom rig, shapes, materials,
//! camera orbit, and studio lamps.

use super::math::{M4, Quat, V3, cos, sin, v3};
use super::rig::{Belt, DEG, Ease, Rig, Track, Xf};
use super::scene::{Camera, Floor, Frame, Instance, Light, Material, Mesh};
use super::shapes::{add, ball, cyl, frame_y};

#[derive(Clone, Debug)]
pub enum Place3D {
    /// Rigid on a joint (`xf` is an optional local offset in the joint's frame).
    Joint { joint: usize, xf: M4 },
    /// A link along a belt.
    Link {
        belt: usize,
        rear_joint: usize,
        front_joint: usize,
        offset: f32,
    },
    /// A wheel on a joint that turns as `roll` advances.
    Wheel {
        joint: usize,
        center: V3,
        radius: f32,
        link_t: f32,
    },
    /// One half of a hydraulic ram between two `(joint, local_point)` anchors.
    Ram([(usize, V3); 2]),
}

#[derive(Clone, Debug)]
pub struct Part3D {
    pub mesh: u32,
    pub material: u32,
    pub place: Place3D,
}

#[derive(Clone, Debug)]
pub struct CamTracks {
    pub az: Track,
    pub y: Track,
    pub dist: Track,
    pub aim_x: Track,
    pub aim_y: Track,
    pub aim_z: Track,
    pub fov: f32,
    pub smooth: bool,
}

impl Default for CamTracks {
    fn default() -> Self {
        Self {
            az: Track::constant(-24.0),
            y: Track::constant(1.8),
            dist: Track::constant(6.8),
            aim_x: Track::constant(0.0),
            aim_y: Track::constant(1.2),
            aim_z: Track::constant(0.0),
            fov: 0.40,
            smooth: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LampSpec {
    pub pos: [Track; 3],
    pub at: [Track; 3],
    pub color: [f32; 3],
    pub size: f32,
    pub gain: Track,
}

impl LampSpec {
    pub fn fixed(pos: V3, at: V3, color: [f32; 3], size: f32) -> Self {
        Self {
            pos: [
                Track::constant(pos.x),
                Track::constant(pos.y),
                Track::constant(pos.z),
            ],
            at: [
                Track::constant(at.x),
                Track::constant(at.y),
                Track::constant(at.z),
            ],
            color,
            size,
            gain: Track::constant(1.0),
        }
    }

    pub fn eval(&self, t: f32) -> Light {
        let g = self.gain.at(t);
        Light {
            pos: v3(self.pos[0].at(t), self.pos[1].at(t), self.pos[2].at(t)),
            at: v3(self.at[0].at(t), self.at[1].at(t), self.at[2].at(t)),
            color: [self.color[0] * g, self.color[1] * g, self.color[2] * g],
            size: self.size,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CustomStage {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub mat_glow: Vec<Track>,
    pub rig: Rig,
    pub belts: Vec<Belt>,
    pub parts: Vec<Part3D>,
    pub roll: Track,
    pub cam: CamTracks,
    pub key: LampSpec,
    pub fills: Vec<LampSpec>,
    pub ambient: [f32; 3],
    pub floor: Floor,
    pub exposure: f32,
    pub bloom: Track,
    /// Optional joint index whose Y is lifted so the lowest vertex of the
    /// scene rests at `y = 0`.
    pub keep_on_floor: Option<usize>,
}

impl Default for CustomStage {
    fn default() -> Self {
        let mut rig = Rig::default();
        rig.add("root", super::rig::NONE, V3::ZERO, Quat::IDENT, false);
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            mat_glow: Vec::new(),
            rig,
            belts: Vec::new(),
            parts: Vec::new(),
            roll: Track::constant(0.0),
            cam: CamTracks::default(),
            key: LampSpec::fixed(
                v3(-3.2, 10.0, 8.0),
                v3(0.0, 1.0, 0.0),
                [1.20, 1.15, 1.06],
                0.9,
            ),
            fills: vec![
                LampSpec::fixed(
                    v3(-10.0, 2.5, 1.5),
                    v3(0.0, 1.2, 0.0),
                    [0.10, 0.15, 0.25],
                    4.0,
                ),
                LampSpec::fixed(
                    v3(5.5, 6.0, -8.0),
                    v3(0.0, 1.2, 0.0),
                    [0.42, 0.16, 0.80],
                    1.6,
                ),
            ],
            ambient: [0.050, 0.052, 0.062],
            floor: Floor {
                base: [0.022, 0.022, 0.027],
                rough: 0.20,
                mirror: 0.5,
            },
            exposure: 1.1,
            bloom: Track::constant(0.55),
            keep_on_floor: None,
        }
    }
}

impl CustomStage {
    pub fn add_material(&mut self, mat: Material, glow: Track) -> u32 {
        self.materials.push(mat);
        self.mat_glow.push(glow);
        self.materials.len() as u32 - 1
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> u32 {
        self.meshes.push(mesh);
        self.meshes.len() as u32 - 1
    }

    /// Adds a two-part hydraulic ram between `(j0, p0)` and `(j1, p1)` (points
    /// given in standing/world rest space if `world_space` is true, else in each
    /// joint's local frame).
    pub fn add_ram(
        &mut self,
        a: (usize, V3),
        b: (usize, V3),
        world_space: bool,
        radius: f32,
        mat_sleeve: u32,
        mat_rod: u32,
        duration: f32,
    ) {
        let end_a = if world_space {
            (a.0, self.rig.unbind[a.0].point(a.1))
        } else {
            a
        };
        let end_b = if world_space {
            (b.0, self.rig.unbind[b.0].point(b.1))
        } else {
            b
        };
        let ends = [end_a, end_b];
        let (mut loc, mut w) = (Vec::new(), Vec::new());
        let mut hi = 0.0f32;
        let steps = 100usize;
        let dur = duration.max(1.0);
        for k in 0..=steps {
            let t = k as f32 * dur / steps as f32;
            self.rig.local(t, &mut loc);
            self.rig.world(t, &loc, &mut w);
            let d = (w[ends[1].0].point(ends[1].1) - w[ends[0].0].point(ends[0].1)).len();
            hi = hi.max(d);
        }
        let len = (hi * 0.56).max(0.08);
        let rod = |rr: f32, l: f32| {
            let mut m = cyl(V3::ZERO, v3(0.0, l, 0.0), rr, 12, rr * 0.2);
            add(
                &mut m,
                &ball(V3::ZERO, rr * 1.5, 1.0, 10),
                &M4::IDENT,
                false,
            );
            m
        };
        let m_sleeve = self.add_mesh(rod(radius, len));
        self.parts.push(Part3D {
            mesh: m_sleeve,
            material: mat_sleeve,
            place: Place3D::Ram(ends),
        });
        let m_rod = self.add_mesh(rod(radius * 0.6, len));
        self.parts.push(Part3D {
            mesh: m_rod,
            material: mat_rod,
            place: Place3D::Ram([ends[1], ends[0]]),
        });
    }

    pub fn eval_into(&self, t: f32, loc: &mut Vec<Xf>, w: &mut Vec<Xf>, out: &mut Frame) {
        self.rig.local(t, loc);
        self.rig.world(t, loc, w);

        if let Some(root_j) = self.keep_on_floor
            && root_j < loc.len()
        {
            let mut low = f32::MAX;
            for p in &self.parts {
                if let Place3D::Joint { joint, xf } = &p.place
                    && let Some(mesh) = self.meshes.get(p.mesh as usize)
                {
                    let m = w[*joint].m4().mul(*xf);
                    for v in &mesh.pos {
                        low = low.min(m.point(*v).y);
                    }
                }
            }
            if low < f32::MAX && low.abs() > 1e-5 {
                loc[root_j].p.y -= low;
                self.rig.world(t, loc, w);
            }
        }

        let roll = self.roll.at(t);
        out.instances.clear();
        for p in &self.parts {
            let xf = match &p.place {
                Place3D::Joint { joint, xf } => w[*joint].m4().mul(*xf),
                Place3D::Link {
                    belt,
                    rear_joint,
                    front_joint,
                    offset,
                } => {
                    let b = &self.belts[*belt];
                    let s = *offset - roll;
                    let z = b.z_at(s);
                    let jn = if z < 0.0 { *rear_joint } else { *front_joint };
                    w[jn].m4().mul(b.at(s))
                }
                Place3D::Wheel {
                    joint,
                    center,
                    radius,
                    link_t,
                } => w[*joint]
                    .mul(Xf {
                        r: Quat::rot_x(roll / (radius + link_t)),
                        p: *center,
                    })
                    .m4(),
                Place3D::Ram(ends) => {
                    let a = w[ends[0].0].point(ends[0].1);
                    let b = w[ends[1].0].point(ends[1].1);
                    frame_y(a, b)
                }
            };
            let glow = self
                .mat_glow
                .get(p.material as usize)
                .map_or(1.0, |tr| tr.at(t));
            out.instances.push(Instance {
                mesh: p.mesh,
                material: p.material,
                xf,
                glow,
            });
        }

        let sample = |tr: &Track| {
            if self.cam.smooth {
                let mut s = 0.0;
                for k in 0..9 {
                    s += tr.at(t + (k as f32 - 4.0) * 0.125);
                }
                s * (1.0 / 9.0)
            } else {
                tr.at(t)
            }
        };
        let az = sample(&self.cam.az) * DEG;
        let d = sample(&self.cam.dist).max(0.2);
        let target = v3(
            sample(&self.cam.aim_x),
            sample(&self.cam.aim_y),
            sample(&self.cam.aim_z),
        );
        let eye = v3(
            target.x + d * sin(az),
            sample(&self.cam.y),
            target.z + d * cos(az),
        );
        out.camera = Camera {
            eye,
            target,
            up: V3::Y,
            tan_half_fov_y: self.cam.fov,
            near: 0.2,
            far: 120.0,
        };
        out.key = self.key.eval(t);
        out.fills.clear();
        for f in &self.fills {
            out.fills.push(f.eval(t));
        }
        out.ambient = self.ambient;
        out.floor = self.floor;
        out.exposure = self.exposure;
        out.bloom = self.bloom.at(t);
    }
}

#[derive(Clone, Debug)]
pub enum Stage {
    Custom(Box<CustomStage>),
}

/// Reusable per-renderer evaluation buffers so `Stage::eval` never allocates
/// in the frame loop.
pub struct Scratch {
    pub loc: Vec<Xf>,
    pub world: Vec<Xf>,
    pub frame: Frame,
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            loc: Vec::with_capacity(64),
            world: Vec::with_capacity(64),
            frame: Frame {
                instances: Vec::with_capacity(512),
                fills: Vec::with_capacity(4),
                ..Frame::default()
            },
        }
    }
}

impl Stage {
    pub fn meshes(&self) -> &[Mesh] {
        match self {
            Stage::Custom(c) => &c.meshes,
        }
    }

    pub fn materials(&self) -> &[Material] {
        match self {
            Stage::Custom(c) => &c.materials,
        }
    }

    pub fn eval(&self, t: f32, scratch: &mut Scratch) {
        match self {
            Stage::Custom(c) => {
                c.eval_into(t, &mut scratch.loc, &mut scratch.world, &mut scratch.frame)
            }
        }
    }

    /// Helper to parse a `[time, value, ease?]` track from a slice of tuples.
    pub fn make_track(keys: &[(f32, f32, Ease)]) -> Track {
        Track::new(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::studio::render::Renderer;
    use crate::studio::shapes::bbox;

    #[test]
    fn a_custom_stage_renders_the_same_twice() {
        let mut scratch = Scratch::new();
        let mut r = Renderer::new(160, 90);
        let mut custom = CustomStage::default();
        let mat = custom.add_material(
            Material::metal([0.8, 0.82, 0.86], 0.25),
            Track::constant(1.0),
        );
        let mesh = custom.add_mesh(bbox(v3(0.0, 1.0, 0.0), v3(0.6, 0.6, 0.6), 0.08));
        custom.parts.push(Part3D {
            mesh,
            material: mat,
            place: Place3D::Joint {
                joint: 0,
                xf: M4::IDENT,
            },
        });
        let cstage = Stage::Custom(Box::new(custom));
        cstage.eval(0.0, &mut scratch);
        let first = r
            .render(cstage.meshes(), cstage.materials(), &scratch.frame)
            .to_vec();
        assert_eq!(first.len(), 160 * 90 * 3);
        assert!(first.iter().any(|&b| b > 40));
        cstage.eval(0.0, &mut scratch);
        assert_eq!(
            first,
            r.render(cstage.meshes(), cstage.materials(), &scratch.frame)
        );
    }
}
