//! The full 329-part robot-to-tank transformation assembly, built in code
//! using `vgfx::studio` primitives, rig, belts, and hydraulic rams.

use super::math::{M4, Quat, TAU, V3, cos, quat_from_axes, sin, v3};
use super::rig::{Alt, Belt, Ch, DEG, Ease, NONE, Parts, Rig, Track, Xf};
use super::scene::{Camera, Film, Floor, Frame, Instance, Light, Material, Mesh};
use super::shapes::{
    add, along, at, ball, bolt, bx, cx, cyl, cyv, cz, frame_y, join, lathe, lathe_arc, p, prism,
    prism_x, prism_y, ring, scaled, slats, tz,
};

pub const HIP_Y: f32 = 2.35;
pub const THIGH: f32 = 0.95;
pub const SHIN: f32 = 1.10;
pub const TRACK_X: f32 = 1.26;
pub const HIP_OUT: f32 = 0.25;
pub const PARK_Z: f32 = 0.40;
pub const T_FIRE: f32 = 9.08;

pub const LINK_T: f32 = 0.03;
pub const ROAD_R: f32 = 0.23;
pub const ROAD_Y: f32 = 0.29;
pub const ROAD_Z: [f32; 6] = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5];
pub const END_R: f32 = 0.25;
pub const END_Y: f32 = 0.50;
pub const END_Z: f32 = 2.03;
pub const ROLL_R: f32 = 0.07;
pub const ROLL_Y: f32 = 0.70;
pub const ROLL_Z: f32 = 1.0;
pub const LINKS: usize = 50;

pub const SILVER: u32 = 0;
pub const CHROME: u32 = 1;
pub const GREY: u32 = 2;
pub const GUN: u32 = 3;
pub const STEEL: u32 = 4;
pub const BLACK: u32 = 5;
pub const RUBBER: u32 = 6;
pub const RED: u32 = 7;
pub const PURPLE: u32 = 8;
pub const EYE: u32 = 9;
pub const MUZZLE: u32 = 10;
pub const FLASH: u32 = 11;
pub const TRACK: u32 = 12;
pub const LAMP: u32 = 13;
pub const WHITE: u32 = 14;

#[derive(Clone, Debug)]
pub struct Jt {
    pub ground: usize,
    pub base: usize,
    pub sprung: usize,
    pub pelvis: usize,
    pub chest: usize,
    pub head: usize,
    pub hatch: usize,
    pub bib: usize,
    pub slide: [usize; 2],
    pub shoulder: [usize; 2],
    pub elbow: [usize; 2],
    pub hand: [usize; 2],
    pub cannon: usize,
    pub barrel1: usize,
    pub barrel2: usize,
    pub lift: usize,
    pub turret: usize,
    pub mantlet: usize,
    pub hip: [usize; 2],
    pub thigh: [usize; 2],
    pub shin: [usize; 2],
    pub foot: [usize; 2],
    pub runit: [usize; 2],
    pub funit: [usize; 2],
    pub deck: [usize; 2],
    pub deck2: [usize; 2],
}

#[derive(Clone, Debug)]
pub struct Motion {
    pub body: Track,
    pub thigh: Track,
    pub shin: Track,
    pub foot: Track,
    pub pin_z: Track,
    pub roll: Track,
    pub belt: Track,
    pub eye: Track,
    pub muzzle: Track,
    pub flash: Track,
    pub lamp: Track,
    pub cam_az: Track,
    pub cam_y: Track,
    pub cam_dist: Track,
    pub aim_x: Track,
    pub aim_y: Track,
    pub aim_z: Track,
}

#[derive(Clone, Debug)]
enum Place {
    Joint(usize),
    Link(usize, f32),
    Wheel(usize, V3, f32),
    Ram([(usize, V3); 2]),
}

#[derive(Clone, Debug)]
struct Part {
    mesh: u32,
    material: u32,
    place: Place,
}

#[derive(Clone, Debug)]
pub struct Transform {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub rig: Rig,
    pub jt: Jt,
    pub motion: Motion,
    pub belt: Belt,
    parts: Vec<Part>,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

fn mirror_q(q: Quat) -> Quat {
    Quat {
        x: q.x,
        y: -q.y,
        z: -q.z,
        w: q.w,
    }
}

fn pair(
    rig: &mut Rig,
    name: [&'static str; 2],
    parent: [usize; 2],
    at: V3,
    pre: Quat,
) -> [usize; 2] {
    let l = rig.add(name[0], parent[0], at, pre, false);
    let r = rig.add(
        name[1],
        parent[1],
        v3(-at.x, at.y, at.z),
        mirror_q(pre),
        false,
    );
    [l, r]
}

fn skeleton(rig: &mut Rig) -> Jt {
    let q0 = Quat::IDENT;
    let lay = Quat::rot_x(-1.570_796_4);
    let ground = rig.add("ground", NONE, V3::ZERO, q0, false);
    let base = rig.add("base", NONE, v3(0.0, HIP_Y, 0.0), q0, false);
    let sprung = rig.add("sprung", base, v3(0.0, HIP_Y, 0.0), q0, false);
    let pelvis = rig.add("pelvis", sprung, v3(0.0, HIP_Y, 0.0), q0, false);
    let chest = rig.add("chest", pelvis, v3(0.0, 3.0, 0.0), q0, false);
    let head = rig.add("head", chest, v3(0.0, 3.92, 0.02), q0, false);
    let hatch = rig.add("hatch", chest, v3(0.0, 2.95, -0.43), q0, false);
    let bib = rig.add("bib", chest, v3(0.0, 3.88, 0.47), q0, false);
    let slide = pair(
        rig,
        ["slide_l", "slide_r"],
        [chest; 2],
        v3(0.88, 3.6, 0.0),
        q0,
    );
    let shoulder = pair(
        rig,
        ["shoulder_l", "shoulder_r"],
        slide,
        v3(1.17, 3.6, 0.0),
        q0,
    );
    let elbow = pair(
        rig,
        ["elbow_l", "elbow_r"],
        shoulder,
        v3(1.19, 2.68, 0.0),
        q0,
    );
    let hand = pair(rig, ["hand_l", "hand_r"], elbow, v3(1.19, 1.86, 0.0), q0);
    let hip = pair(
        rig,
        ["hip_l", "hip_r"],
        [pelvis; 2],
        v3(0.45, HIP_Y, 0.0),
        q0,
    );
    let thigh = pair(rig, ["thigh_l", "thigh_r"], hip, v3(0.45, HIP_Y, 0.0), q0);
    let shin = pair(
        rig,
        ["shin_l", "shin_r"],
        thigh,
        v3(0.45, HIP_Y - THIGH, 0.0),
        q0,
    );
    let foot = pair(
        rig,
        ["foot_l", "foot_r"],
        shin,
        v3(0.45, HIP_Y - THIGH - SHIN, 0.0),
        q0,
    );
    let runit = pair(
        rig,
        ["runit_l", "runit_r"],
        shin,
        v3(TRACK_X - HIP_OUT, HIP_Y, 0.42),
        lay,
    );
    let deck = pair(rig, ["deck_l", "deck_r"], shin, v3(0.17, 0.85, -0.405), q0);
    let deck2 = pair(
        rig,
        ["deck2_l", "deck2_r"],
        thigh,
        v3(0.17, 1.9, -0.405),
        q0,
    );
    let lift = rig.add("lift", chest, v3(0.0, 3.1, -1.0), lay, false);
    let turret = rig.add("turret", lift, V3::ZERO, q0, true);
    let mantlet = rig.add("mantlet", turret, v3(0.0, 0.27, 0.58), q0, true);
    let stow = Quat::rot_y(1.570_796_4).mul(lay);
    let funit = pair(
        rig,
        ["funit_l", "funit_r"],
        [chest; 2],
        v3(0.94, 1.75, -0.73),
        stow,
    );
    let hang = quat_from_axes(v3(0.0, 0.0, 1.0), v3(-1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0));
    let cannon = rig.add("cannon", elbow[1], v3(-1.66, 3.05, 0.0), hang, false);
    let barrel1 = rig.add("barrel1", cannon, V3::ZERO, q0, true);
    let barrel2 = rig.add("barrel2", barrel1, V3::ZERO, q0, true);
    Jt {
        ground,
        base,
        sprung,
        pelvis,
        chest,
        head,
        hatch,
        bib,
        slide,
        shoulder,
        elbow,
        hand,
        cannon,
        barrel1,
        barrel2,
        lift,
        turret,
        mantlet,
        hip,
        thigh,
        shin,
        foot,
        runit,
        funit,
        deck,
        deck2,
    }
}

pub fn default_materials() -> Vec<Material> {
    let m = |base: [f32; 3], rough: f32, groove: f32, grain: f32| Material {
        base,
        metal: 1.0,
        rough,
        emit: [0.0; 3],
        groove,
        grain,
    };
    let p = |base: [f32; 3], rough: f32, grain: f32| Material {
        base,
        metal: 0.0,
        rough,
        emit: [0.0; 3],
        groove: 0.0,
        grain,
    };
    vec![
        Material {
            base: [0.80, 0.81, 0.84],
            metal: 0.75,
            rough: 0.30,
            emit: [0.0; 3],
            groove: 0.012,
            grain: 0.35,
        },
        m([0.90, 0.91, 0.93], 0.10, 0.0, 0.05),
        p([0.46, 0.47, 0.50], 0.42, 0.25),
        m([0.27, 0.28, 0.31], 0.38, 0.0, 0.4),
        m([0.11, 0.115, 0.13], 0.50, 0.008, 0.5),
        p([0.012, 0.012, 0.016], 0.22, 0.1),
        p([0.018, 0.018, 0.02], 0.85, 0.6),
        p([0.55, 0.025, 0.03], 0.35, 0.15),
        p([0.20, 0.04, 0.42], 0.30, 0.15),
        Material {
            base: [0.05, 0.0, 0.0],
            metal: 0.0,
            rough: 0.4,
            emit: [9.0, 0.5, 0.25],
            groove: 0.0,
            grain: 0.0,
        },
        Material {
            base: [0.02, 0.0, 0.04],
            metal: 0.0,
            rough: 0.4,
            emit: [5.0, 1.0, 12.0],
            groove: 0.0,
            grain: 0.0,
        },
        Material {
            base: [0.0, 0.0, 0.0],
            metal: 0.0,
            rough: 0.6,
            emit: [260.0, 100.0, 480.0],
            groove: 0.0,
            grain: 0.0,
        },
        m([0.16, 0.16, 0.17], 0.55, 0.0, 0.7),
        Material {
            base: [0.03, 0.0, 0.06],
            metal: 0.0,
            rough: 0.4,
            emit: [2.2, 0.5, 5.0],
            groove: 0.0,
            grain: 0.0,
        },
        Material {
            base: [0.58, 0.59, 0.62],
            metal: 0.3,
            rough: 0.42,
            emit: [0.0; 3],
            groove: 0.0,
            grain: 0.3,
        },
    ]
}

fn build_keys(rig: &mut Rig, j: &Jt) -> Motion {
    use Ease::*;
    let tr = |k: &[(f32, f32, Ease)]| Track::new(k);
    let ankle_back = PARK_Z - THIGH - SHIN;

    for s in 0..2 {
        rig.key(
            j.thigh[s],
            Ch::Rz,
            &[
                (0.0, -0.6, Lin),
                (1.2, 2.0, Smooth),
                (2.2, 0.6, Smooth),
                (2.6, 0.0, Smooth),
            ],
        );
        rig.key(
            j.foot[s],
            Ch::Rz,
            &[
                (0.0, 0.6, Lin),
                (1.2, -2.0, Smooth),
                (2.2, -0.6, Smooth),
                (2.6, 0.0, Smooth),
            ],
        );
    }
    rig.key(
        j.pelvis,
        Ch::Ry,
        &[
            (0.0, 3.0, Lin),
            (1.4, -4.0, Smooth),
            (2.3, -2.0, Smooth),
            (2.7, 0.0, Smooth),
        ],
    );
    rig.key(
        j.chest,
        Ch::Ry,
        &[
            (0.0, 2.0, Lin),
            (1.7, -7.0, Smooth),
            (2.4, -4.0, Smooth),
            (2.8, 0.0, Smooth),
        ],
    );
    rig.key(
        j.chest,
        Ch::Rx,
        &[
            (0.0, 0.0, Lin),
            (0.9, -1.6, Smooth),
            (1.9, 0.6, Smooth),
            (2.4, 3.0, Smooth),
            (2.9, 0.0, Smooth),
        ],
    );
    rig.key(
        j.head,
        Ch::Ry,
        &[
            (0.0, 6.0, Lin),
            (0.5, 6.0, Lin),
            (1.5, -30.0, Smooth),
            (2.3, -26.0, Smooth),
            (2.75, 0.0, Smooth),
        ],
    );
    rig.key(
        j.head,
        Ch::Rx,
        &[
            (0.0, 4.0, Lin),
            (0.5, 4.0, Lin),
            (1.5, -3.0, Smooth),
            (2.3, 4.0, Smooth),
            (2.75, 0.0, Smooth),
        ],
    );
    let (ls, rs) = (j.shoulder[0], j.shoulder[1]);
    rig.key(
        rs,
        Ch::Rx,
        &[
            (0.0, 0.0, Lin),
            (0.3, 0.0, Lin),
            (1.5, -20.0, Smooth),
            (2.2, -15.0, Smooth),
            (2.45, -8.0, Smooth),
            (3.6, -180.0, Clunk),
            (4.3, -180.0, Lin),
            (5.55, -360.0, Clunk),
        ],
    );
    rig.key(
        j.elbow[1],
        Ch::Rx,
        &[
            (0.0, -4.0, Lin),
            (0.3, -4.0, Lin),
            (1.5, -30.0, Smooth),
            (2.2, -24.0, Smooth),
            (2.6, -34.0, Smooth),
            (3.1, 0.0, Smooth),
        ],
    );
    rig.key(
        rs,
        Ch::Rz,
        &[
            (0.0, -16.0, Lin),
            (2.2, -16.0, Lin),
            (2.6, -27.0, Smooth),
            (2.9, -24.0, Smooth),
            (3.4, 0.0, Smooth),
        ],
    );
    rig.key(
        ls,
        Ch::Rx,
        &[
            (0.0, 2.0, Lin),
            (1.3, -3.0, Smooth),
            (2.2, 2.0, Smooth),
            (2.6, 6.0, Smooth),
            (3.2, -8.0, Smooth),
            (4.0, 0.0, Smooth),
        ],
    );
    rig.key(
        ls,
        Ch::Rz,
        &[
            (0.0, 16.0, Lin),
            (2.2, 16.0, Lin),
            (2.6, 27.0, Smooth),
            (3.5, 0.0, Smooth),
        ],
    );
    rig.key(
        j.elbow[0],
        Ch::Rx,
        &[
            (0.0, -6.0, Lin),
            (1.3, -10.0, Smooth),
            (2.2, -6.0, Smooth),
            (2.6, -30.0, Smooth),
            (3.1, -42.0, Smooth),
            (4.1, 0.0, Lock),
        ],
    );

    let body = tr(&[
        (2.2, 0.0, Lin),
        (2.6, 7.0, Smooth),
        (3.45, 26.0, Heavy),
        (4.55, 90.0, Heavy),
    ]);
    let thigh = tr(&[
        (2.2, 0.0, Lin),
        (2.6, -10.0, Smooth),
        (3.45, 4.0, Smooth),
        (4.55, 90.0, Heavy),
    ]);
    let shin = tr(&[(2.2, 0.0, Lin), (2.6, 9.0, Smooth), (3.45, 90.0, Heavy)]);
    let foot = tr(&[(2.6, 0.0, Lin), (2.72, 0.0, Lin), (3.5, 90.0, Heavy)]);
    let roll = tr(&[(6.6, 0.0, Lin), (8.7, 0.75, Smooth)]);
    let pin_z = tr(&[(2.6, 0.0, Lin), (3.5, ankle_back, Heavy)]);

    rig.key2(j.hip, Ch::Px, &[(2.7, 0.0, Lin), (3.3, HIP_OUT, Lock)]);
    rig.key2(j.runit, Ch::Pz, &[(2.75, 0.0, Lin), (3.25, 0.36, Lock)]);
    rig.key(j.head, Ch::Py, &[(2.75, 0.0, Lin), (3.2, -0.82, Lock)]);
    rig.key(j.bib, Ch::Rx, &[(2.95, 0.0, Lin), (3.75, -202.6, Clunk)]);

    rig.key(
        j.lift,
        Ch::Pz,
        &[
            (2.9, 0.0, Lin),
            (3.5, -0.35, Lock),
            (4.3, -0.35, Lin),
            (4.95, 0.53, Lock),
        ],
    );
    rig.key(j.lift, Ch::Py, &[(4.3, 0.0, Lin), (4.95, -0.40, Smooth)]);
    rig.key(
        j.turret,
        Ch::Ry,
        &[
            (0.0, 180.0, Lin),
            (2.95, 180.0, Lin),
            (3.7, 0.0, Clunk),
            (7.0, 0.0, Lin),
            (8.3, -16.0, Back),
        ],
    );
    rig.key(
        j.mantlet,
        Ch::Rx,
        &[
            (7.3, 0.0, Lin),
            (8.5, -7.0, Back),
            (T_FIRE, -7.0, Lin),
            (T_FIRE + 0.07, -9.5, Out),
            (T_FIRE + 0.6, -7.0, Smooth),
        ],
    );
    rig.joints[j.cannon].alt.as_mut().unwrap().w = tr(&[(3.62, 0.0, Lin), (4.25, 1.0, Heavy)]);

    rig.key2(
        j.funit,
        Ch::Px,
        &[
            (3.5, 0.0, Lin),
            (3.95, 0.86, Smooth),
            (4.95, 0.86, Lin),
            (5.3, 0.32, Lock),
        ],
    );
    rig.key2(j.funit, Ch::Rz, &[(3.9, 0.0, Lin), (4.4, -90.0, Lock)]);
    rig.key2(j.funit, Ch::Pz, &[(4.3, 0.0, Lin), (4.95, 1.51, Heavy)]);
    rig.key2(j.funit, Ch::Py, &[(4.3, 0.0, Lin), (4.95, 0.60, Heavy)]);
    for s in 0..2 {
        rig.joints[j.runit[s]].alt.as_mut().unwrap().w = tr(&[(3.6, 0.0, Lin), (3.7, 1.0, Lin)]);
        rig.joints[j.funit[s]].alt.as_mut().unwrap().w = tr(&[(5.42, 0.0, Lin), (5.56, 1.0, In)]);
    }
    rig.key2(j.deck, Ch::Ry, &[(3.8, 0.0, Lin), (4.3, 180.0, Clunk)]);
    rig.key2(j.deck2, Ch::Ry, &[(4.05, 0.0, Lin), (4.55, 180.0, Clunk)]);
    rig.key2(j.slide, Ch::Pz, &[(4.3, 0.0, Lin), (4.75, -0.40, Lock)]);
    rig.key(j.hatch, Ch::Py, &[(4.75, 0.0, Lin), (5.25, 0.95, Smooth)]);
    rig.key(j.hatch, Ch::Rx, &[(5.25, 0.0, Lin), (5.6, 38.0, Lock)]);
    rig.key(
        j.barrel1,
        Ch::Pz,
        &[
            (5.3, 0.0, Lin),
            (5.7, 0.5, Lock),
            (T_FIRE, 0.5, Lin),
            (T_FIRE + 0.06, 0.22, Out),
            (T_FIRE + 0.7, 0.5, Smooth),
        ],
    );
    rig.key(
        j.barrel2,
        Ch::Pz,
        &[
            (5.5, 0.0, Lin),
            (5.9, 0.55, Lock),
            (T_FIRE, 0.55, Lin),
            (T_FIRE + 0.05, 0.36, Out),
            (T_FIRE + 0.6, 0.55, Smooth),
        ],
    );

    rig.key(
        j.sprung,
        Ch::Py,
        &[
            (3.7, 0.0, Lin),
            (4.3, 0.07, Smooth),
            (6.2, 0.07, Lin),
            (6.6, 0.0, Drop),
        ],
    );
    rig.key(
        j.sprung,
        Ch::Rx,
        &[
            (6.2, 0.0, Lin),
            (6.34, 0.9, Out),
            (6.6, 0.0, Smooth),
            (6.95, -1.1, Smooth),
            (7.6, 0.0, Smooth),
            (8.45, 0.0, Lin),
            (8.75, 1.3, Smooth),
            (9.0, 0.0, Smooth),
            (T_FIRE, 0.0, Lin),
            (T_FIRE + 0.1, -2.6, Out),
            (T_FIRE + 0.42, 0.7, Smooth),
            (T_FIRE + 0.75, 0.0, Smooth),
        ],
    );
    rig.key(
        j.sprung,
        Ch::Pz,
        &[
            (T_FIRE, 0.0, Lin),
            (T_FIRE + 0.09, -0.07, Out),
            (T_FIRE + 0.6, 0.0, Smooth),
        ],
    );

    Motion {
        body,
        thigh,
        shin,
        foot,
        pin_z,
        roll,
        belt: tr(&[(5.6, 0.0, Lin), (5.76, 0.10, Out), (6.15, 0.0, Smooth)]),
        eye: tr(&[
            (0.0, 0.12, Lin),
            (0.5, 0.2, Smooth),
            (1.5, 1.0, Smooth),
            (2.6, 1.25, Smooth),
            (3.2, 0.4, Smooth),
        ]),
        muzzle: tr(&[
            (0.0, 0.06, Lin),
            (1.0, 0.12, Smooth),
            (2.1, 0.9, Smooth),
            (2.6, 1.0, Lin),
            (3.6, 0.25, Smooth),
            (5.9, 0.3, Lin),
            (6.6, 0.8, Smooth),
            (8.6, 1.0, Lin),
            (T_FIRE, 2.6, In),
            (T_FIRE + 0.05, 4.0, Lin),
            (T_FIRE + 0.3, 0.3, Out),
            (10.0, 0.7, Smooth),
        ]),
        flash: tr(&[
            (T_FIRE - 0.02, 0.0, Lin),
            (T_FIRE + 0.02, 1.0, Lin),
            (T_FIRE + 0.09, 0.45, Lin),
            (T_FIRE + 0.3, 0.0, Out),
        ]),
        lamp: tr(&[
            (0.0, 0.3, Lin),
            (1.6, 1.0, Smooth),
            (2.6, 1.0, Lin),
            (2.8, 0.3, Lin),
            (3.0, 1.3, Lin),
            (6.2, 0.6, Smooth),
            (6.6, 1.2, Lin),
            (10.0, 1.0, Lin),
        ]),
        cam_az: tr(&[
            (-1.0, -19.0, Lin),
            (2.6, -30.0, Lin),
            (6.4, -40.0, Lin),
            (11.0, -49.0, Lin),
        ]),
        cam_y: tr(&[
            (-1.0, 0.7, Lin),
            (2.6, 1.2, Lin),
            (6.4, 1.3, Lin),
            (11.0, 1.0, Lin),
        ]),
        cam_dist: tr(&[
            (-1.0, 6.5, Lin),
            (2.2, 7.9, Lin),
            (3.1, 9.8, Lin),
            (4.6, 8.8, Lin),
            (6.6, 6.6, Lin),
            (11.0, 7.1, Lin),
        ]),
        aim_x: tr(&[(0.0, 0.0, Lin)]),
        aim_y: tr(&[
            (-1.0, 2.45, Lin),
            (2.5, 2.45, Lin),
            (3.1, 3.05, Lin),
            (3.7, 2.85, Lin),
            (4.5, 1.9, Lin),
            (5.6, 1.1, Lin),
            (11.0, 0.88, Lin),
        ]),
        aim_z: tr(&[
            (-1.0, 0.0, Lin),
            (2.6, 0.0, Lin),
            (3.4, -1.0, Lin),
            (4.7, 0.3, Lin),
            (6.6, 0.9, Lin),
            (8.8, 1.6, Lin),
            (11.0, 1.75, Lin),
        ]),
    }
}

fn build_pelvis(q: &mut Parts, j: &Jt) {
    let pv = j.pelvis;
    q.w(pv, GUN, bx([0.0, 2.38, 0.0], [0.42, 0.19, 0.29], 0.05));
    q.w(pv, CHROME, cx(-0.74, 0.74, HIP_Y, 0.0, 0.085, 20));
    q.w(pv, STEEL, cx(-0.70, 0.70, HIP_Y - 0.02, -0.2, 0.05, 12));
    q.w(
        pv,
        SILVER,
        prism(
            &[
                [-0.36, 2.62],
                [0.36, 2.62],
                [0.27, 2.24],
                [0.11, 2.02],
                [-0.11, 2.02],
                [-0.27, 2.24],
            ],
            0.27,
            0.40,
            0.035,
        ),
    );
    q.w(
        pv,
        STEEL,
        prism(
            &[[-0.2, 2.5], [0.2, 2.5], [0.14, 2.3], [-0.14, 2.3]],
            0.39,
            0.415,
            0.008,
        ),
    );
    q.w(
        pv,
        PURPLE,
        prism(
            &[[-0.08, 2.24], [0.08, 2.24], [0.05, 2.09], [-0.05, 2.09]],
            0.39,
            0.418,
            0.008,
        ),
    );
    q.w(
        pv,
        RED,
        bx([0.0, 2.555, 0.405], [0.22, 0.018, 0.012], 0.005),
    );
    q.w(
        pv,
        GREY,
        prism(
            &[[-0.4, 2.6], [0.4, 2.6], [0.3, 2.2], [-0.3, 2.2]],
            -0.36,
            -0.27,
            0.03,
        ),
    );
    for (i, y) in [2.66f32, 2.78, 2.90].iter().enumerate() {
        let w = 0.40 + 0.03 * i as f32;
        q.w(
            pv,
            if i == 1 { GUN } else { STEEL },
            bx([0.0, *y, 0.0], [w, 0.05, 0.27], 0.025),
        );
    }
    q.w(pv, CHROME, cyv(0.0, 2.55, 3.0, -0.12, 0.09, 16));
    for sx in [-1.0f32, 1.0] {
        q.w(pv, CHROME, cyv(0.3 * sx, 2.56, 3.0, 0.2, 0.035, 10));
        q.w(pv, GUN, cyv(0.3 * sx, 2.56, 2.8, 0.2, 0.05, 10));
        for y in [2.3f32, 2.46] {
            q.w(
                pv,
                CHROME,
                bolt(p(0.3 * sx, y, 0.405), v3(0.0, 0.0, 1.0), 0.022, 0.012),
            );
        }
    }
}

fn build_leg(q: &mut Parts, j: &Jt) {
    let (hip, th, sh, ft) = (q.s(j.hip), q.s(j.thigh), q.s(j.shin), q.s(j.foot));
    let x = 0.45;
    q.w(hip, GUN, cx(0.17, 0.73, HIP_Y, 0.0, 0.21, 24));
    q.w(hip, CHROME, cx(0.14, 0.76, HIP_Y, 0.0, 0.12, 20));
    q.w(
        hip,
        STEEL,
        along(
            &ring(
                &[[0.215, 0.0], [0.24, 0.0], [0.24, 0.06], [0.215, 0.06]],
                24,
            ),
            p(x - 0.03, HIP_Y, 0.0),
            p(x + 1.0, HIP_Y, 0.0),
        ),
    );

    q.w(
        th,
        GUN,
        cyl(p(x, 1.46, 0.0), p(x, 2.26, 0.0), 0.17, 20, 0.02),
    );
    q.w(
        th,
        WHITE,
        prism_x(
            &[
                [-0.25, 1.56],
                [0.2, 1.56],
                [0.28, 1.72],
                [0.28, 2.12],
                [0.2, 2.22],
                [-0.25, 2.22],
                [-0.29, 2.05],
                [-0.29, 1.7],
            ],
            0.19,
            0.71,
            0.04,
        ),
    );
    q.w(
        th,
        SILVER,
        prism_x(
            &[[0.27, 1.66], [0.325, 1.74], [0.325, 2.08], [0.27, 2.16]],
            0.25,
            0.65,
            0.02,
        ),
    );
    q.w(
        th,
        STEEL,
        slats([x, 1.80, 0.33], [x, 2.02, 0.33], 4, [0.13, 0.018, 0.012]),
    );
    q.w(th, STEEL, bx([0.715, 1.9, 0.0], [0.012, 0.2, 0.16], 0.008));
    q.w(th, STEEL, bx([0.185, 1.9, 0.0], [0.012, 0.2, 0.16], 0.008));
    q.w(th, RED, bx([x, 2.185, 0.285], [0.2, 0.016, 0.012], 0.005));
    for (bx_, by) in [(0.24f32, 1.62f32), (0.66, 1.62), (0.24, 2.16), (0.66, 2.16)] {
        q.w(
            th,
            CHROME,
            bolt(p(bx_, by, 0.283), v3(0.0, 0.0, 1.0), 0.02, 0.012),
        );
    }
    q.w(
        th,
        GUN,
        cyl(p(x, 2.16, -0.31), p(x, 1.8, -0.33), 0.055, 12, 0.008),
    );
    q.w(
        th,
        CHROME,
        cyl(p(x, 1.85, -0.33), p(x, 1.52, -0.3), 0.03, 10, 0.004),
    );

    q.w(sh, GUN, cx(0.15, 0.75, 1.40, 0.0, 0.18, 24));
    q.w(sh, CHROME, cx(0.13, 0.77, 1.40, 0.0, 0.085, 16));
    q.w(
        sh,
        SILVER,
        prism_x(
            &[
                [-0.27, 0.36],
                [0.27, 0.36],
                [0.33, 0.54],
                [0.33, 1.10],
                [0.25, 1.30],
                [-0.16, 1.30],
                [-0.31, 1.06],
                [-0.31, 0.52],
            ],
            0.17,
            0.73,
            0.04,
        ),
    );
    q.w(
        sh,
        SILVER,
        prism_x(
            &[
                [0.20, 1.20],
                [0.33, 1.24],
                [0.385, 1.36],
                [0.36, 1.56],
                [0.26, 1.60],
                [0.19, 1.44],
            ],
            0.23,
            0.67,
            0.03,
        ),
    );
    q.w(sh, RED, bx([x, 1.40, 0.385], [0.13, 0.03, 0.01], 0.005));
    q.w(sh, STEEL, bx([x, 0.80, 0.33], [0.15, 0.25, 0.012], 0.008));
    q.w(
        sh,
        GUN,
        slats([x, 0.62, 0.345], [x, 0.98, 0.345], 7, [0.12, 0.014, 0.012]),
    );
    for dx in [-0.21f32, 0.21] {
        q.w(
            sh,
            WHITE,
            prism_x(
                &[[0.32, 0.5], [0.365, 0.56], [0.365, 1.04], [0.32, 1.12]],
                x + dx - 0.035,
                x + dx + 0.035,
                0.012,
            ),
        );
    }
    q.w(
        sh,
        PURPLE,
        bx([x, 1.115, 0.33], [0.15, 0.018, 0.012], 0.005),
    );
    q.w(sh, GUN, bx([x, 0.82, -0.3], [0.2, 0.3, 0.035], 0.02));
    q.w(
        sh,
        STEEL,
        slats([x, 0.62, -0.335], [x, 0.74, -0.335], 3, [0.14, 0.02, 0.012]),
    );
    q.w(sh, CHROME, cyv(x - 0.1, 0.9, 1.12, -0.345, 0.035, 10));
    q.w(sh, CHROME, cyv(x + 0.1, 0.9, 1.12, -0.345, 0.035, 10));
    q.w(sh, STEEL, bx([0.165, 0.8, 0.0], [0.012, 0.3, 0.2], 0.008));
    for y in [0.62f32, 1.08] {
        q.w(sh, CHROME, cz(0.745, y, -0.26, 0.44, 0.03, 10));
        q.w(sh, GUN, bx([0.74, y, 0.0], [0.03, 0.06, 0.1], 0.015));
    }
    for dx in [-0.23f32, 0.23] {
        q.w(sh, GUN, bx([x + dx, 0.34, 0.0], [0.035, 0.1, 0.13], 0.02));
    }

    let (d1, d2) = (q.s(j.deck), q.s(j.deck2));
    q.w(
        d1,
        SILVER,
        prism(
            &[[0.19, 0.4], [0.57, 0.4], [0.57, 1.3], [0.19, 1.3]],
            -0.42,
            -0.39,
            0.01,
        ),
    );
    q.w(
        d1,
        STEEL,
        slats(
            [0.38, 0.6, -0.424],
            [0.38, 0.9, -0.424],
            5,
            [0.13, 0.02, 0.006],
        ),
    );
    q.w(
        d1,
        WHITE,
        bx([0.38, 1.12, -0.424], [0.13, 0.08, 0.006], 0.004),
    );
    q.w(d1, GUN, cyv(0.17, 0.42, 1.28, -0.405, 0.02, 8));
    q.w(
        d2,
        SILVER,
        prism(
            &[[0.19, 1.52], [0.57, 1.52], [0.57, 2.22], [0.19, 2.22]],
            -0.42,
            -0.39,
            0.01,
        ),
    );
    q.w(
        d2,
        STEEL,
        slats(
            [0.38, 1.72, -0.424],
            [0.38, 2.02, -0.424],
            5,
            [0.13, 0.02, 0.006],
        ),
    );
    q.w(d2, GUN, cyv(0.17, 1.54, 2.2, -0.405, 0.02, 8));

    q.w(ft, GUN, cx(0.16, 0.74, 0.30, 0.0, 0.125, 20));
    q.w(ft, CHROME, cx(0.14, 0.76, 0.30, 0.0, 0.06, 12));
    q.w(
        ft,
        SILVER,
        prism_x(
            &[
                [-0.2, 0.02],
                [0.72, 0.02],
                [0.76, 0.06],
                [0.72, 0.17],
                [0.36, 0.3],
                [0.14, 0.3],
                [0.1, 0.2],
                [-0.16, 0.2],
                [-0.2, 0.14],
            ],
            0.15,
            0.75,
            0.03,
        ),
    );
    q.w(
        ft,
        SILVER,
        prism_x(
            &[[0.3, 0.26], [0.7, 0.13], [0.72, 0.19], [0.36, 0.335]],
            0.24,
            0.66,
            0.015,
        ),
    );
    q.w(ft, STEEL, bx([x, 0.07, 0.755], [0.25, 0.05, 0.012], 0.008));
    q.w(
        ft,
        GUN,
        prism_x(
            &[
                [-0.36, 0.02],
                [-0.18, 0.02],
                [-0.18, 0.2],
                [-0.3, 0.24],
                [-0.38, 0.14],
            ],
            0.24,
            0.66,
            0.025,
        ),
    );
    q.w(ft, RUBBER, bx([x, 0.016, 0.19], [0.27, 0.016, 0.56], 0.006));
    q.w(ft, RED, bx([x, 0.11, -0.385], [0.14, 0.03, 0.008], 0.004));
    q.w(
        ft,
        PURPLE,
        prism_x(
            &[[0.42, 0.262], [0.6, 0.197], [0.61, 0.215], [0.43, 0.28]],
            0.34,
            0.56,
            0.004,
        ),
    );
}

fn build_chest(q: &mut Parts, j: &Jt) {
    let c = j.chest;
    q.w(
        c,
        GUN,
        prism(
            &[
                [-0.5, 2.95],
                [0.5, 2.95],
                [0.86, 3.35],
                [0.86, 3.80],
                [0.62, 3.9],
                [-0.62, 3.9],
                [-0.86, 3.80],
                [-0.86, 3.35],
            ],
            -0.40,
            0.385,
            0.05,
        ),
    );
    q.w(
        c,
        STEEL,
        prism(
            &[
                [-0.4, 3.1],
                [0.4, 3.1],
                [0.7, 3.42],
                [0.7, 3.74],
                [-0.7, 3.74],
                [-0.7, 3.42],
            ],
            0.38,
            0.4,
            0.008,
        ),
    );
    q.w(
        c,
        CHROME,
        at(
            &along(
                &ring(&[[0.15, 0.0], [0.2, 0.0], [0.2, 0.03], [0.15, 0.03]], 28),
                p(0.0, 0.0, 0.0),
                p(0.0, 0.0, 1.0),
            ),
            p(0.0, 3.45, 0.385),
        ),
    );
    q.w(c, LAMP, cz(0.0, 3.45, 0.385, 0.405, 0.15, 24));
    q.w(
        c,
        GUN,
        slats(
            [-0.45, 3.45, 0.405],
            [-0.28, 3.45, 0.405],
            4,
            [0.016, 0.2, 0.01],
        ),
    );
    q.w(
        c,
        GUN,
        slats(
            [0.28, 3.45, 0.405],
            [0.45, 3.45, 0.405],
            4,
            [0.016, 0.2, 0.01],
        ),
    );
    q.w(c, CHROME, cx(-0.6, 0.6, 3.68, 0.4, 0.016, 8));
    q.w(c, CHROME, cx(-0.5, 0.5, 3.2, 0.4, 0.016, 8));
    q.w(
        c,
        CHROME,
        at(
            &ring(
                &[
                    [0.43, 0.0],
                    [0.49, 0.0],
                    [0.49, 0.05],
                    [0.47, 0.075],
                    [0.43, 0.075],
                ],
                36,
            ),
            p(0.0, 3.88, 0.05),
        ),
    );
    q.w(c, BLACK, cyv(0.0, 3.86, 3.905, 0.05, 0.43, 28));
    for sx in [-1.0f32, 1.0] {
        q.w(
            c,
            WHITE,
            prism(
                &[
                    [0.5 * sx, 2.98],
                    [0.56 * sx, 2.98],
                    [0.9 * sx, 3.36],
                    [0.9 * sx, 3.44],
                    [0.84 * sx, 3.44],
                ],
                -0.34,
                0.36,
                0.02,
            ),
        );
        q.w(
            c,
            STEEL,
            slats(
                [0.865 * sx, 3.5, -0.2],
                [0.865 * sx, 3.5, 0.2],
                5,
                [0.012, 0.045, 0.02],
            ),
        );
        q.w(c, CHROME, cz(0.9 * sx, 3.6, -0.6, 0.2, 0.04, 12));
        q.w(
            c,
            STEEL,
            bx([0.88 * sx, 3.6, -0.6], [0.05, 0.07, 0.03], 0.012),
        );
        q.w(
            c,
            WHITE,
            bx([0.56 * sx, 3.905, -0.02], [0.1, 0.018, 0.3], 0.01),
        );
        q.w(
            c,
            CHROME,
            bolt(p(0.56 * sx, 3.925, 0.2), v3(0.0, 1.0, 0.0), 0.022, 0.01),
        );
        q.w(
            c,
            CHROME,
            bolt(p(0.56 * sx, 3.925, -0.24), v3(0.0, 1.0, 0.0), 0.022, 0.01),
        );
    }

    let b = j.bib;
    q.w(
        b,
        SILVER,
        prism(
            &[
                [-0.8, 3.86],
                [0.8, 3.86],
                [0.82, 3.5],
                [0.46, 3.08],
                [-0.46, 3.08],
                [-0.82, 3.5],
            ],
            0.42,
            0.50,
            0.03,
        ),
    );
    q.w(
        b,
        WHITE,
        prism(
            &[[-0.31, 3.8], [0.31, 3.8], [0.21, 3.2], [-0.21, 3.2]],
            0.49,
            0.545,
            0.02,
        ),
    );
    for (i, y) in [3.68f32, 3.58, 3.48, 3.38].iter().enumerate() {
        q.w(
            b,
            STEEL,
            bx(
                [0.0, *y, 0.547],
                [0.2 - 0.017 * i as f32, 0.022, 0.008],
                0.004,
            ),
        );
    }
    q.w(
        b,
        PURPLE,
        bx([0.0, 3.27, 0.547], [0.15, 0.022, 0.008], 0.004),
    );
    for sx in [-1.0f32, 1.0] {
        q.w(
            b,
            STEEL,
            prism(
                &[
                    [0.37 * sx, 3.78],
                    [0.74 * sx, 3.78],
                    [0.755 * sx, 3.52],
                    [0.5 * sx, 3.22],
                    [0.31 * sx, 3.22],
                ],
                0.495,
                0.512,
                0.008,
            ),
        );
        q.w(
            b,
            GUN,
            slats(
                [0.55 * sx, 3.4, 0.515],
                [0.55 * sx, 3.7, 0.515],
                6,
                [0.16, 0.016, 0.012],
            ),
        );
        q.w(
            b,
            RED,
            bx([0.57 * sx, 3.815, 0.503], [0.13, 0.016, 0.008], 0.004),
        );
        q.w(
            b,
            LAMP,
            bx([0.4 * sx, 3.15, 0.503], [0.035, 0.02, 0.006], 0.003),
        );
        q.w(
            b,
            CHROME,
            bolt(p(0.74 * sx, 3.82, 0.5), v3(0.0, 0.0, 1.0), 0.022, 0.012),
        );
        q.w(
            b,
            CHROME,
            bolt(p(0.5 * sx, 3.16, 0.5), v3(0.0, 0.0, 1.0), 0.022, 0.012),
        );
    }
    q.w(b, GUN, cx(-0.72, 0.72, 3.88, 0.47, 0.035, 12));

    let h = j.hatch;
    q.w(
        h,
        SILVER,
        prism(
            &[
                [-0.74, 2.97],
                [0.74, 2.97],
                [0.74, 3.5],
                [0.6, 3.87],
                [-0.6, 3.87],
                [-0.74, 3.5],
            ],
            -0.452,
            -0.412,
            0.012,
        ),
    );
    q.w(
        h,
        WHITE,
        prism(
            &[
                [-0.3, 3.05],
                [0.3, 3.05],
                [0.3, 3.62],
                [0.22, 3.72],
                [-0.22, 3.72],
                [-0.3, 3.62],
            ],
            -0.468,
            -0.45,
            0.008,
        ),
    );
    q.w(
        h,
        STEEL,
        slats(
            [0.0, 3.16, -0.472],
            [0.0, 3.4, -0.472],
            5,
            [0.2, 0.016, 0.006],
        ),
    );
    for sx in [-1.0f32, 1.0] {
        q.w(
            h,
            WHITE,
            bx([0.52 * sx, 3.3, -0.46], [0.035, 0.3, 0.01], 0.006),
        );
        q.w(
            h,
            LAMP,
            bx([0.47 * sx, 3.74, -0.46], [0.08, 0.03, 0.008], 0.004),
        );
        q.w(
            h,
            GUN,
            bx([0.47 * sx, 3.74, -0.458], [0.1, 0.045, 0.006], 0.003),
        );
        q.w(
            h,
            CHROME,
            bolt(p(0.66 * sx, 3.05, -0.452), v3(0.0, 0.0, -1.0), 0.022, 0.012),
        );
        q.w(
            h,
            CHROME,
            bolt(p(0.66 * sx, 3.45, -0.452), v3(0.0, 0.0, -1.0), 0.022, 0.012),
        );
    }
}

fn build_head(q: &mut Parts, j: &Jt) {
    let h = j.head;
    let big = |m: Mesh| scaled(&m, p(0.0, 3.94, 0.02), 1.2, p(0.0, 0.0, 0.03));
    let o = p(0.0, 0.0, 0.02);
    q.w(h, GUN, big(cyv(0.0, 3.6, 4.0, 0.02, 0.15, 16)));
    q.w(
        h,
        SILVER,
        big(at(
            &lathe_arc(
                &[[0.262, 3.97], [0.312, 3.97], [0.33, 4.29], [0.275, 4.29]],
                30,
                true,
                0.82,
                2.2,
                7.225,
            ),
            o,
        )),
    );
    q.w(
        h,
        SILVER,
        big(at(
            &lathe(
                &[
                    [0.0, 4.27],
                    [0.335, 4.27],
                    [0.34, 4.32],
                    [0.318, 4.41],
                    [0.25, 4.475],
                    [0.12, 4.508],
                    [0.0, 4.515],
                ],
                32,
            ),
            o,
        )),
    );
    q.w(
        h,
        WHITE,
        big(prism_x(
            &[
                [0.3, 4.25],
                [0.36, 4.33],
                [0.33, 4.47],
                [0.12, 4.56],
                [-0.22, 4.52],
                [-0.26, 4.44],
                [0.1, 4.5],
                [0.27, 4.43],
            ],
            -0.05,
            0.05,
            0.015,
        )),
    );
    q.w(
        h,
        SILVER,
        big(bx([0.0, 4.28, 0.315], [0.21, 0.028, 0.03], 0.012)),
    );
    for sx in [-1.0f32, 1.0] {
        q.w(
            h,
            SILVER,
            big(bx([0.215 * sx, 4.1, 0.25], [0.04, 0.14, 0.06], 0.02)),
        );
        q.w(h, GUN, big(cx(0.3 * sx, 0.355 * sx, 4.2, 0.02, 0.075, 14)));
        q.w(
            h,
            CHROME,
            big(cx(0.34 * sx, 0.362 * sx, 4.2, 0.02, 0.035, 10)),
        );
        q.w(
            h,
            EYE,
            big(prism(
                &[
                    [0.035 * sx, 4.195],
                    [0.15 * sx, 4.222],
                    [0.15 * sx, 4.252],
                    [0.035 * sx, 4.228],
                ],
                0.245,
                0.262,
                0.0,
            )),
        );
    }
    q.w(
        h,
        STEEL,
        big(bx([0.0, 4.14, 0.12], [0.19, 0.15, 0.13], 0.04)),
    );
    q.w(
        h,
        GUN,
        big(prism_x(
            &[[0.24, 4.22], [0.3, 4.12], [0.24, 4.09]],
            -0.03,
            0.03,
            0.006,
        )),
    );
    q.w(
        h,
        GUN,
        big(bx([0.0, 4.045, 0.225], [0.085, 0.022, 0.03], 0.008)),
    );
    q.w(
        h,
        SILVER,
        big(prism_x(
            &[[0.16, 3.96], [0.27, 3.96], [0.29, 4.02], [0.16, 4.03]],
            -0.12,
            0.12,
            0.012,
        )),
    );
}

fn build_arm(q: &mut Parts, j: &Jt) {
    let (sl, sh, el, ha) = (q.s(j.slide), q.s(j.shoulder), q.s(j.elbow), q.s(j.hand));
    q.w(sl, STEEL, bx([0.92, 3.6, 0.0], [0.06, 0.15, 0.16], 0.03));
    q.w(sl, GUN, cx(0.9, 1.0, 3.6, 0.0, 0.18, 20));

    q.w(sh, GUN, cx(0.98, 1.2, 3.6, 0.0, 0.15, 16));
    q.w(sh, SILVER, bx([1.24, 3.63, 0.0], [0.28, 0.33, 0.29], 0.055));
    q.w(sh, WHITE, bx([1.24, 3.966, 0.0], [0.2, 0.012, 0.2], 0.006));
    q.w(sh, STEEL, bx([1.522, 3.64, 0.0], [0.008, 0.2, 0.2], 0.005));
    q.w(
        sh,
        GUN,
        slats(
            [1.532, 3.54, 0.0],
            [1.532, 3.74, 0.0],
            5,
            [0.01, 0.014, 0.16],
        ),
    );
    q.w(sh, RED, bx([1.523, 3.895, 0.0], [0.008, 0.018, 0.2], 0.004));
    q.w(
        sh,
        PURPLE,
        bx([1.24, 3.38, 0.292], [0.2, 0.022, 0.008], 0.004),
    );
    q.w(
        sh,
        STEEL,
        bx([1.24, 3.66, 0.292], [0.16, 0.14, 0.008], 0.005),
    );
    q.w(
        sh,
        LAMP,
        bx([1.24, 3.87, 0.293], [0.1, 0.016, 0.006], 0.003),
    );
    for (dx, dy) in [(-0.2f32, -0.2f32), (0.2, -0.2), (-0.2, 0.26), (0.2, 0.26)] {
        q.w(
            sh,
            CHROME,
            bolt(
                p(1.24 + dx, 3.63 + dy, 0.292),
                v3(0.0, 0.0, 1.0),
                0.022,
                0.012,
            ),
        );
        q.w(
            sh,
            CHROME,
            bolt(
                p(1.24 + dx, 3.63 + dy, -0.292),
                v3(0.0, 0.0, -1.0),
                0.022,
                0.012,
            ),
        );
    }
    q.w(sh, GUN, cyv(1.19, 2.7, 3.32, 0.0, 0.15, 16));
    q.w(sh, GREY, bx([1.19, 3.0, 0.0], [0.19, 0.24, 0.19], 0.04));
    q.w(
        sh,
        STEEL,
        bx([1.19, 3.0, 0.192], [0.11, 0.15, 0.008], 0.005),
    );
    q.w(
        sh,
        GUN,
        slats(
            [1.19, 2.92, 0.2],
            [1.19, 3.08, 0.2],
            4,
            [0.09, 0.012, 0.008],
        ),
    );
    q.w(sh, STEEL, bx([1.382, 3.0, 0.0], [0.008, 0.15, 0.11], 0.005));

    q.w(el, GUN, cx(0.975, 1.405, 2.68, 0.0, 0.165, 22));
    q.w(el, CHROME, cx(0.955, 1.425, 2.68, 0.0, 0.07, 14));
    q.w(
        el,
        SILVER,
        prism_x(
            &[
                [-0.2, 1.9],
                [0.2, 1.9],
                [0.235, 2.05],
                [0.235, 2.48],
                [0.17, 2.56],
                [-0.17, 2.56],
                [-0.235, 2.48],
                [-0.235, 2.05],
            ],
            0.975,
            1.405,
            0.04,
        ),
    );
    q.w(
        el,
        CHROME,
        bx([1.19, 1.915, 0.0], [0.225, 0.03, 0.215], 0.012),
    );
    q.w(
        el,
        STEEL,
        bx([1.19, 2.27, 0.236], [0.13, 0.16, 0.008], 0.005),
    );
    q.w(
        el,
        GUN,
        slats(
            [1.19, 2.16, 0.244],
            [1.19, 2.38, 0.244],
            5,
            [0.1, 0.014, 0.008],
        ),
    );
    q.w(
        el,
        STEEL,
        bx([1.19, 2.27, -0.236], [0.13, 0.16, 0.008], 0.005),
    );
    q.w(
        el,
        PURPLE,
        bx([1.19, 2.0, 0.237], [0.16, 0.016, 0.008], 0.004),
    );
    q.w(el, WHITE, bx([0.972, 2.26, 0.0], [0.008, 0.2, 0.15], 0.005));
    for y in [2.08f32, 2.46] {
        q.w(
            el,
            CHROME,
            bolt(p(1.05, y, 0.238), v3(0.0, 0.0, 1.0), 0.02, 0.01),
        );
        q.w(
            el,
            CHROME,
            bolt(p(1.33, y, 0.238), v3(0.0, 0.0, 1.0), 0.02, 0.01),
        );
    }

    q.w(ha, STEEL, cyv(1.19, 1.8, 1.9, 0.0, 0.12, 14));
    q.w(ha, BLACK, bx([1.19, 1.70, 0.0], [0.1, 0.12, 0.16], 0.035));
    q.w(
        ha,
        BLACK,
        slats(
            [1.13, 1.57, -0.12],
            [1.13, 1.57, 0.13],
            4,
            [0.09, 0.06, 0.036],
        ),
    );
    q.w(
        ha,
        BLACK,
        slats(
            [1.07, 1.66, -0.12],
            [1.07, 1.66, 0.13],
            4,
            [0.035, 0.07, 0.036],
        ),
    );
    q.w(
        ha,
        BLACK,
        bx([1.13, 1.69, 0.185], [0.06, 0.08, 0.04], 0.018),
    );
}

fn bore(m: &Mesh) -> Mesh {
    along(m, V3::ZERO, v3(0.0, 0.0, 1.0))
}

fn build_cannon(q: &mut Parts, j: &Jt) {
    let c = j.cannon;
    let mut prof = vec![[0.0, 0.0], [0.15, 0.0], [0.19, 0.04], [0.19, 0.09]];
    for i in 0..6 {
        let z = 0.10 + 0.068 * i as f32;
        prof.extend_from_slice(&[
            [0.25, z],
            [0.25, z + 0.03],
            [0.19, z + 0.038],
            [0.19, z + 0.06],
        ]);
    }
    prof.extend_from_slice(&[
        [0.2, 0.54],
        [0.2, 0.9],
        [0.235, 0.93],
        [0.235, 1.02],
        [0.215, 1.05],
        [0.13, 1.05],
        [0.13, 0.3],
        [0.0, 0.3],
    ]);
    q.l(c, BLACK, bore(&lathe(&prof, 36)));
    for z in [0.58f32, 0.82] {
        q.l(
            c,
            CHROME,
            bore(&ring(
                &[[0.198, z], [0.222, z], [0.222, z + 0.05], [0.198, z + 0.05]],
                32,
            )),
        );
    }
    q.l(
        c,
        PURPLE,
        bore(&ring(
            &[[0.232, 0.95], [0.243, 0.955], [0.243, 0.995], [0.232, 1.0]],
            32,
        )),
    );
    q.l(
        c,
        CHROME,
        bolt(p(0.0, 0.0, 0.0), v3(0.0, 0.0, -1.0), 0.06, 0.02),
    );
    q.l(c, BLACK, tz(0.0, 0.31, 0.26, 0.86, 0.06, 0.045, 16));
    q.l(c, LAMP, cz(0.0, 0.31, 0.78, 0.8, 0.045, 12));
    q.l(c, CHROME, tz(0.0, 0.31, 0.84, 0.875, 0.067, 0.045, 16));
    q.l(c, GUN, bx([0.0, 0.24, 0.36], [0.03, 0.05, 0.035], 0.01));
    q.l(c, GUN, bx([0.0, 0.24, 0.74], [0.03, 0.05, 0.035], 0.01));
    q.l(c, GUN, bx([0.0, -0.205, 0.62], [0.13, 0.03, 0.22], 0.015));
    q.l(c, STEEL, bx([0.0, -0.235, 0.62], [0.09, 0.022, 0.27], 0.01));
    for z in [0.44f32, 0.8] {
        q.l(
            c,
            CHROME,
            bolt(p(0.1, -0.2, z), v3(1.0, 0.0, 0.0), 0.02, 0.035),
        );
        q.l(
            c,
            CHROME,
            bolt(p(-0.1, -0.2, z), v3(-1.0, 0.0, 0.0), 0.02, 0.035),
        );
    }

    let b1 = j.barrel1;
    q.l(b1, GUN, tz(0.0, 0.0, 0.32, 1.45, 0.118, 0.092, 28));
    q.l(
        b1,
        CHROME,
        bore(&ring(
            &[[0.117, 1.36], [0.14, 1.37], [0.14, 1.44], [0.117, 1.45]],
            28,
        )),
    );
    for z in [0.55f32, 0.8, 1.05] {
        q.l(
            b1,
            STEEL,
            bore(&ring(
                &[
                    [0.117, z],
                    [0.127, z],
                    [0.127, z + 0.035],
                    [0.117, z + 0.035],
                ],
                28,
            )),
        );
    }
    let b2 = j.barrel2;
    q.l(b2, BLACK, tz(0.0, 0.0, 0.62, 1.6, 0.084, 0.06, 24));
    q.l(
        b2,
        BLACK,
        bore(&ring(
            &[
                [0.062, 1.55],
                [0.1, 1.55],
                [0.128, 1.6],
                [0.128, 1.71],
                [0.112, 1.75],
                [0.078, 1.75],
                [0.066, 1.7],
            ],
            28,
        )),
    );
    q.l(
        b2,
        CHROME,
        bore(&ring(
            &[[0.127, 1.63], [0.136, 1.635], [0.136, 1.675], [0.127, 1.68]],
            28,
        )),
    );
    for z in [0.85f32, 1.15] {
        q.l(
            b2,
            STEEL,
            bore(&ring(
                &[[0.083, z], [0.09, z], [0.09, z + 0.03], [0.083, z + 0.03]],
                24,
            )),
        );
    }
    q.l(b2, MUZZLE, cz(0.0, 0.0, 1.5, 1.56, 0.062, 16));
    q.l(b2, FLASH, ball(p(0.0, 0.0, 1.68), 0.07, 1.0, 12));
}

fn build_turret(q: &mut Parts, j: &Jt) {
    let t = j.turret;
    q.l(
        t,
        STEEL,
        lathe(
            &[
                [0.0, 0.0],
                [0.5, 0.0],
                [0.52, 0.02],
                [0.52, 0.075],
                [0.0, 0.075],
            ],
            36,
        ),
    );
    q.l(
        t,
        SILVER,
        prism_x(
            &[
                [-0.68, 0.07],
                [0.52, 0.07],
                [0.62, 0.16],
                [0.62, 0.36],
                [0.36, 0.5],
                [-0.5, 0.52],
                [-0.68, 0.42],
            ],
            -0.44,
            0.44,
            0.04,
        ),
    );
    for sx in [-1.0f32, 1.0] {
        q.l(
            t,
            WHITE,
            prism_y(
                &[
                    [0.43 * sx, -0.62],
                    [0.6 * sx, -0.5],
                    [0.66 * sx, -0.05],
                    [0.57 * sx, 0.42],
                    [0.43 * sx, 0.54],
                ],
                0.085,
                0.43,
                0.03,
            ),
        );
        q.l(
            t,
            PURPLE,
            prism_y(
                &[
                    [0.6 * sx, -0.48],
                    [0.612 * sx, -0.48],
                    [0.672 * sx, -0.06],
                    [0.66 * sx, -0.06],
                ],
                0.3,
                0.36,
                0.0,
            ),
        );
        for i in 0..3 {
            let z = 0.08 + 0.11 * i as f32;
            let a = p((0.62 - 0.02 * i as f32) * sx, 0.34, z);
            q.l(
                t,
                BLACK,
                cyl(a, a + v3(0.1 * sx, 0.07, 0.1), 0.04, 12, 0.006),
            );
        }
        q.l(
            t,
            RED,
            bx([0.36 * sx, 0.455, 0.37], [0.05, 0.012, 0.02], 0.004),
        );
        q.l(
            t,
            STEEL,
            bx([0.2 * sx, 0.2, 0.625], [0.1, 0.1, 0.01], 0.006),
        );
    }
    q.l(
        t,
        GUN,
        at(
            &lathe(
                &[
                    [0.0, 0.0],
                    [0.18, 0.0],
                    [0.18, 0.07],
                    [0.15, 0.115],
                    [0.0, 0.115],
                ],
                24,
            ),
            p(0.2, 0.5, -0.2),
        ),
    );
    q.l(
        t,
        CHROME,
        at(
            &ring(&[[0.1, 0.0], [0.13, 0.0], [0.13, 0.02], [0.1, 0.02]], 20),
            p(0.2, 0.615, -0.2),
        ),
    );
    q.l(t, LAMP, bx([0.2, 0.56, -0.02], [0.05, 0.02, 0.008], 0.003));
    q.l(
        t,
        STEEL,
        at(
            &lathe(
                &[
                    [0.0, 0.0],
                    [0.15, 0.0],
                    [0.15, 0.02],
                    [0.13, 0.035],
                    [0.0, 0.035],
                ],
                20,
            ),
            p(-0.2, 0.515, -0.1),
        ),
    );
    q.l(
        t,
        CHROME,
        bx([-0.2, 0.56, -0.1], [0.06, 0.01, 0.012], 0.004),
    );
    q.l(
        t,
        STEEL,
        slats(
            [0.0, 0.525, -0.42],
            [0.0, 0.525, -0.58],
            4,
            [0.3, 0.01, 0.014],
        ),
    );
    q.l(t, STEEL, bx([0.0, 0.28, -0.72], [0.4, 0.15, 0.07], 0.03));
    q.l(
        t,
        GUN,
        slats(
            [-0.24, 0.28, -0.795],
            [0.24, 0.28, -0.795],
            5,
            [0.03, 0.1, 0.01],
        ),
    );
    q.l(t, GUN, cyv(-0.34, 0.5, 0.58, -0.5, 0.03, 10));
    q.l(
        t,
        CHROME,
        cyl(p(-0.34, 0.58, -0.5), p(-0.34, 1.2, -0.56), 0.012, 8, 0.002),
    );
    q.l(t, GUN, bx([0.0, 0.27, 0.58], [0.34, 0.15, 0.06], 0.03));
    for (x, z) in [(-0.36f32, 0.3f32), (0.36, 0.3), (-0.4, -0.45), (0.4, -0.45)] {
        q.l(
            t,
            CHROME,
            bolt(
                p(x, 0.505 + 0.02 * (0.5 - z).min(0.5), z),
                v3(0.0, 1.0, 0.0),
                0.022,
                0.012,
            ),
        );
    }

    let m = j.mantlet;
    q.l(m, GUN, cx(-0.27, 0.27, 0.0, 0.06, 0.16, 24));
    q.l(m, CHROME, cx(-0.29, 0.29, 0.0, 0.06, 0.06, 12));
    q.l(m, STEEL, tz(0.0, 0.0, 0.1, 0.24, 0.285, 0.255, 32));
    q.l(m, STEEL, bx([0.0, -0.2, 0.5], [0.14, 0.02, 0.36], 0.01));
    q.l(m, CHROME, cz(0.09, -0.19, 0.2, 0.84, 0.018, 8));
    q.l(m, CHROME, cz(-0.09, -0.19, 0.2, 0.84, 0.018, 8));
}

fn build_track_frame(q: &mut Parts, joint: usize, sgn: f32) {
    let z = |v: f32| v * sgn;
    q.l(
        joint,
        STEEL,
        prism_x(
            &[
                [z(2.0), 0.36],
                [z(1.7), 0.2],
                [z(0.03), 0.2],
                [z(0.03), 0.62],
                [z(1.95), 0.62],
                [z(2.08), 0.5],
            ],
            -0.245,
            -0.14,
            0.02,
        ),
    );
    q.l(
        joint,
        GUN,
        prism_x(
            &[[z(1.6), 0.3], [z(0.2), 0.3], [z(0.2), 0.52], [z(1.7), 0.52]],
            -0.14,
            -0.11,
            0.01,
        ),
    );
    for w in [0.3f32, 0.9, 1.5] {
        q.l(
            joint,
            GUN,
            cyl(
                p(-0.1, 0.5, z(w - 0.16)),
                p(-0.1, 0.29, z(w)),
                0.04,
                10,
                0.006,
            ),
        );
        q.l(joint, CHROME, cx(-0.2, 0.2, 0.29, z(w), 0.04, 12));
        q.l(joint, CHROME, cx(-0.14, -0.06, 0.5, z(w - 0.16), 0.05, 12));
        q.l(
            joint,
            WHITE,
            cyl(
                p(-0.1, 0.57, z(w + 0.12)),
                p(-0.1, 0.34, z(w + 0.04)),
                0.025,
                8,
                0.004,
            ),
        );
    }
    q.l(joint, CHROME, cx(-0.2, 0.2, 0.5, z(2.03), 0.045, 12));
    q.l(
        joint,
        CHROME,
        cz(
            -0.19,
            0.5,
            z(2.0).min(z(1.55)),
            z(2.0).max(z(1.55)),
            0.028,
            10,
        ),
    );
    q.l(joint, GUN, cx(-0.2, 0.17, 0.70, z(1.0), 0.03, 10));
    q.l(
        joint,
        GUN,
        bx([-0.19, 0.64, z(1.0)], [0.04, 0.06, 0.05], 0.015),
    );
    q.l(
        joint,
        GUN,
        bx([-0.19, 0.41, z(0.06)], [0.05, 0.2, 0.05], 0.02),
    );
    q.l(
        joint,
        CHROME,
        cz(
            -0.19,
            0.3,
            z(0.0).min(z(0.14)),
            z(0.0).max(z(0.14)),
            0.022,
            8,
        ),
    );
    q.l(
        joint,
        CHROME,
        cz(
            -0.19,
            0.52,
            z(0.0).min(z(0.14)),
            z(0.0).max(z(0.14)),
            0.022,
            8,
        ),
    );
    q.l(
        joint,
        WHITE,
        prism_x(
            &[
                [z(0.02), 0.6],
                [z(1.9), 0.6],
                [z(1.9), 0.66],
                [z(0.02), 0.66],
            ],
            -0.255,
            -0.235,
            0.006,
        ),
    );
}

fn road_wheel() -> (Mesh, Mesh) {
    let ax = |m: &Mesh| along(m, v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0));
    let tyre = ax(&ring(
        &[
            [0.15, -0.17],
            [0.215, -0.17],
            [0.23, -0.15],
            [0.23, -0.02],
            [0.2, -0.01],
            [0.2, 0.01],
            [0.23, 0.02],
            [0.23, 0.15],
            [0.215, 0.17],
            [0.15, 0.17],
        ],
        28,
    ));
    let mut hub = ax(&lathe(
        &[
            [0.0, -0.15],
            [0.155, -0.15],
            [0.155, 0.11],
            [0.13, 0.145],
            [0.085, 0.15],
            [0.08, 0.19],
            [0.045, 0.2],
            [0.0, 0.2],
        ],
        24,
    ));
    for i in 0..6 {
        let xf = M4::rotate(Quat::rot_x(TAU * i as f32 / 6.0));
        add(
            &mut hub,
            &bolt(p(0.15, 0.1, 0.0), v3(1.0, 0.0, 0.0), 0.02, 0.012),
            &xf,
            false,
        );
        add(
            &mut hub,
            &bx([0.145, 0.062, 0.0], [0.012, 0.042, 0.014], 0.005),
            &xf,
            false,
        );
    }
    (tyre, hub)
}

fn end_wheel(teeth: bool) -> (Mesh, Mesh) {
    let ax = |m: &Mesh| along(m, v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0));
    let mut rim = Mesh::default();
    for x in [-0.15f32, 0.11] {
        add(
            &mut rim,
            &ax(&ring(
                &[[0.17, x], [0.25, x], [0.25, x + 0.04], [0.17, x + 0.04]],
                28,
            )),
            &M4::IDENT,
            false,
        );
        if teeth {
            for i in 0..13 {
                let xf = M4::rotate(Quat::rot_x(TAU * i as f32 / 13.0));
                add(
                    &mut rim,
                    &bx([x + 0.02, 0.262, 0.0], [0.018, 0.022, 0.022], 0.006),
                    &xf,
                    false,
                );
            }
        }
    }
    let mut hub = ax(&lathe(
        &[
            [0.0, -0.17],
            [0.12, -0.17],
            [0.12, 0.13],
            [0.09, 0.18],
            [0.04, 0.2],
            [0.0, 0.2],
        ],
        20,
    ));
    for i in 0..5 {
        let xf = M4::rotate(Quat::rot_x(TAU * i as f32 / 5.0));
        for x in [-0.13f32, 0.13] {
            add(
                &mut hub,
                &bx([x, 0.15, 0.0], [0.016, 0.06, 0.03], 0.008),
                &xf,
                false,
            );
        }
        add(
            &mut hub,
            &bolt(p(0.185, 0.07, 0.0), v3(1.0, 0.0, 0.0), 0.016, 0.01),
            &xf,
            false,
        );
    }
    (rim, hub)
}

fn roller() -> Mesh {
    let ax = |m: &Mesh| along(m, v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0));
    ax(&lathe(
        &[
            [0.0, -0.16],
            [0.07, -0.16],
            [0.07, -0.04],
            [0.04, -0.03],
            [0.04, 0.03],
            [0.07, 0.04],
            [0.07, 0.16],
            [0.0, 0.16],
        ],
        14,
    ))
}

fn link(pitch: f32) -> Mesh {
    let h = pitch * 0.5;
    join(&[
        bx([0.0, 0.0, 0.0], [0.235, 0.022, h - 0.012], 0.008),
        bx([0.0, 0.034, 0.0], [0.24, 0.02, 0.026], 0.008),
        bx([0.12, 0.028, h * 0.5], [0.09, 0.012, 0.03], 0.006),
        bx([-0.12, 0.028, h * 0.5], [0.09, 0.012, 0.03], 0.006),
        cx(-0.245, 0.245, 0.0, h - 0.002, 0.02, 8),
        prism_x(
            &[
                [-0.04, -0.02],
                [0.04, -0.02],
                [0.015, -0.085],
                [-0.015, -0.085],
            ],
            -0.02,
            0.02,
            0.004,
        ),
    ])
}

fn default_belt() -> Belt {
    let mut c: Vec<[f32; 3]> = ROAD_Z
        .iter()
        .map(|&z| [z, ROAD_Y, ROAD_R + LINK_T])
        .collect();
    c.push([END_Z, END_Y, END_R + LINK_T]);
    c.push([ROLL_Z, ROLL_Y, ROLL_R + LINK_T]);
    c.push([-ROLL_Z, ROLL_Y, ROLL_R + LINK_T]);
    c.push([-END_Z, END_Y, END_R + LINK_T]);
    Belt::from_wheels(&c)
}

impl Transform {
    pub fn new() -> Transform {
        let mut rig = Rig::default();
        let jt = skeleton(&mut rig);
        for s in 0..2 {
            let x = if s == 0 { TRACK_X } else { -TRACK_X };
            for j in [jt.runit[s], jt.funit[s]] {
                rig.joints[j].alt = Some(Alt {
                    parent: jt.ground,
                    pos: v3(x, 0.0, 0.0),
                    rot: Quat::IDENT,
                    w: Default::default(),
                });
            }
        }
        rig.joints[jt.cannon].alt = Some(Alt {
            parent: jt.mantlet,
            pos: v3(0.0, 0.0, 0.1),
            rot: Quat::IDENT,
            w: Default::default(),
        });
        let motion = build_keys(&mut rig, &jt);
        let belt = default_belt();

        let mut meshes: Vec<Mesh> = Vec::new();
        let mut parts: Vec<Part> = Vec::new();
        {
            let mut q = Parts::new(&rig);
            build_pelvis(&mut q, &jt);
            build_chest(&mut q, &jt);
            build_head(&mut q, &jt);
            build_cannon(&mut q, &jt);
            build_turret(&mut q, &jt);
            for right in [false, true] {
                q.right = right;
                build_leg(&mut q, &jt);
                build_arm(&mut q, &jt);
                build_track_frame(&mut q, jt.runit[right as usize], -1.0);
                build_track_frame(&mut q, jt.funit[right as usize], 1.0);
            }
            for b in q.buckets {
                meshes.push(b.mesh);
                parts.push(Part {
                    mesh: meshes.len() as u32 - 1,
                    material: b.mat,
                    place: Place::Joint(b.joint),
                });
            }
        }
        let pitch = belt.len / LINKS as f32;
        meshes.push(link(pitch));
        let link_mesh = meshes.len() as u32 - 1;
        for side in 0..2 {
            for i in 0..LINKS {
                parts.push(Part {
                    mesh: link_mesh,
                    material: TRACK,
                    place: Place::Link(side, (i as f32 + 0.5) * pitch),
                });
            }
        }
        let mut both = |m: Mesh| -> [u32; 2] {
            let mut r = Mesh::default();
            add(&mut r, &m, &M4::IDENT, true);
            meshes.push(m);
            meshes.push(r);
            [meshes.len() as u32 - 2, meshes.len() as u32 - 1]
        };
        let (tyre, hub) = road_wheel();
        let (tyre, hub) = (both(tyre), both(hub));
        let (rim_i, hub_i) = end_wheel(false);
        let (rim_i, hub_i) = (both(rim_i), both(hub_i));
        let (rim_s, hub_s) = end_wheel(true);
        let (rim_s, hub_s) = (both(rim_s), both(hub_s));
        let roller_m = both(roller());
        for s in 0..2 {
            let half = |z: f32| if z < 0.0 { jt.runit[s] } else { jt.funit[s] };
            for z in ROAD_Z {
                let c = v3(0.0, ROAD_Y, z);
                parts.push(Part {
                    mesh: tyre[s],
                    material: RUBBER,
                    place: Place::Wheel(half(z), c, ROAD_R),
                });
                parts.push(Part {
                    mesh: hub[s],
                    material: WHITE,
                    place: Place::Wheel(half(z), c, ROAD_R),
                });
            }
            for (z, rim, hb) in [(-END_Z, rim_i, hub_i), (END_Z, rim_s, hub_s)] {
                let c = v3(0.0, END_Y, z);
                parts.push(Part {
                    mesh: rim[s],
                    material: GUN,
                    place: Place::Wheel(half(z), c, END_R),
                });
                parts.push(Part {
                    mesh: hb[s],
                    material: SILVER,
                    place: Place::Wheel(half(z), c, END_R),
                });
            }
            for z in [-ROLL_Z, ROLL_Z] {
                parts.push(Part {
                    mesh: roller_m[s],
                    material: GUN,
                    place: Place::Wheel(half(z), v3(0.0, ROLL_Y, z), ROLL_R),
                });
            }
        }
        let mut film = Transform {
            meshes,
            materials: default_materials(),
            rig,
            jt,
            motion,
            belt,
            parts,
        };
        film.rams();
        film
    }

    fn rams(&mut self) {
        let j = &self.jt;
        let ub = |jn: usize, w: V3| (jn, self.rig.unbind[jn].point(w));
        let mut list: Vec<([(usize, V3); 2], f32)> = vec![
            (
                [
                    ub(j.chest, v3(0.0, 3.4, -0.40)),
                    (j.lift, v3(0.0, 0.0, 0.3)),
                ],
                0.06,
            ),
            (
                [
                    ub(j.chest, v3(0.0, 2.95, -0.40)),
                    (j.lift, v3(0.0, 0.0, -0.3)),
                ],
                0.06,
            ),
        ];
        for s in 0..2 {
            let x = if s == 0 { 1.0 } else { -1.0 };
            list.push((
                [
                    ub(j.pelvis, v3(0.3 * x, 2.52, -0.3)),
                    ub(j.thigh[s], v3(0.45 * x, 1.75, -0.33)),
                ],
                0.045,
            ));
            list.push((
                [
                    ub(j.chest, v3(0.45 * x, 3.2, -0.41)),
                    ub(j.hatch, v3(0.45 * x, 3.75, -0.44)),
                ],
                0.03,
            ));
            list.push((
                [
                    ub(j.chest, v3(0.84 * x, 3.38, -0.2)),
                    (j.funit[s], v3(-0.245 * x, 0.42, 1.3)),
                ],
                0.065,
            ));
            list.push((
                [
                    ub(j.chest, v3(0.62 * x, 3.08, -0.2)),
                    (j.funit[s], v3(-0.245 * x, 0.42, 0.6)),
                ],
                0.065,
            ));
        }
        let (mut loc, mut w) = (Vec::new(), Vec::new());
        for (ends, r) in list {
            let mut hi = 0.0f32;
            for k in 0..=400 {
                self.pose(k as f32 * 0.025, &mut loc, &mut w);
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
            self.meshes.push(rod(r, len));
            self.parts.push(Part {
                mesh: self.meshes.len() as u32 - 1,
                material: GUN,
                place: Place::Ram(ends),
            });
            self.meshes.push(rod(r * 0.6, len));
            self.parts.push(Part {
                mesh: self.meshes.len() as u32 - 1,
                material: CHROME,
                place: Place::Ram([ends[1], ends[0]]),
            });
        }
    }

    pub fn pose(&self, t: f32, loc: &mut Vec<Xf>, w: &mut Vec<Xf>) {
        let (rig, j, m) = (&self.rig, &self.jt, &self.motion);
        rig.local(t, loc);
        let (body, thigh, shin, foot) = (m.body.at(t), m.thigh.at(t), m.shin.at(t), m.foot.at(t));
        let lean = |a: f32, x: &mut Xf| x.r = Quat::rot_x(a * DEG).mul(x.r);
        lean(body, &mut loc[j.pelvis]);
        for s in 0..2 {
            lean(thigh - body, &mut loc[j.thigh[s]]);
            lean(shin - thigh, &mut loc[j.shin[s]]);
            lean(foot - shin, &mut loc[j.foot[s]]);
        }
        let mut mid = V3::ZERO;
        let mut low = f32::MAX;
        for s in 0..2 {
            let shin = loc[j.pelvis]
                .mul(loc[j.hip[s]])
                .mul(loc[j.thigh[s]])
                .mul(loc[j.shin[s]]);
            let foot = shin.mul(loc[j.foot[s]]);
            mid += foot.p * 0.5;
            for p in [
                v3(0.0, -0.3, 0.76),
                v3(0.0, -0.3, -0.2),
                v3(0.0, -0.16, -0.38),
            ] {
                low = low.min(foot.point(p).y);
            }
            let unit = shin.mul(loc[j.runit[s]]);
            let r = ROAD_R + 2.0 * LINK_T;
            low = low
                .min(unit.point(v3(0.0, ROAD_Y, -1.5)).y - r)
                .min(unit.point(v3(0.0, ROAD_Y, -0.3)).y - r);
            low = low.min(unit.point(v3(0.0, END_Y, -END_Z)).y - END_R - 2.0 * LINK_T);
        }
        let pin = v3(0.0, mid.y - low, m.pin_z.at(t) + m.roll.at(t));
        loc[j.base] = Xf {
            r: Quat::IDENT,
            p: pin - mid,
        };
        loc[j.ground] = Xf {
            r: Quat::IDENT,
            p: v3(0.0, 0.0, PARK_Z + m.roll.at(t)),
        };
        rig.world(t, loc, w);
    }

    pub fn counts(&self) -> (usize, usize) {
        (
            self.parts.len(),
            self.parts
                .iter()
                .map(|p| self.meshes[p.mesh as usize].idx.len() / 3)
                .sum(),
        )
    }

    pub fn eval_into(&self, t: f32, loc: &mut Vec<Xf>, w: &mut Vec<Xf>, out: &mut Frame) {
        self.pose(t, loc, w);
        let m = &self.motion;
        let roll = m.roll.at(t) + m.belt.at(t);
        let glow = [m.eye.at(t), m.muzzle.at(t), m.flash.at(t), m.lamp.at(t)];
        out.instances.clear();
        for p in &self.parts {
            let xf = match &p.place {
                Place::Joint(jn) => w[*jn].m4(),
                Place::Link(side, s) => {
                    let s = *s - roll;
                    let z = self.belt.z_at(s);
                    let jn = if z < 0.0 {
                        self.jt.runit[*side]
                    } else {
                        self.jt.funit[*side]
                    };
                    w[jn].m4().mul(self.belt.at(s))
                }
                Place::Wheel(jn, c, r) => w[*jn]
                    .mul(Xf {
                        r: Quat::rot_x(roll / (r + LINK_T)),
                        p: *c,
                    })
                    .m4(),
                Place::Ram(ends) => {
                    let a = w[ends[0].0].point(ends[0].1);
                    let b = w[ends[1].0].point(ends[1].1);
                    frame_y(a, b)
                }
            };
            let g = match p.material {
                EYE => glow[0],
                MUZZLE => glow[1],
                FLASH => glow[2],
                LAMP => glow[3],
                _ => 1.0,
            };
            out.instances.push(Instance {
                mesh: p.mesh,
                material: p.material,
                xf,
                glow: g,
            });
        }
        let muzzle = w[self.jt.barrel2].point(v3(0.0, 0.0, 1.9));
        stage_light(t, m, muzzle, out);
    }
}

fn stage_light(t: f32, m: &Motion, muzzle: V3, out: &mut Frame) {
    let soft = |tr: &Track| {
        let mut s = 0.0;
        for k in 0..9 {
            s += tr.at(t + (k as f32 - 4.0) * 0.125);
        }
        s * (1.0 / 9.0)
    };
    let az = soft(&m.cam_az) * DEG;
    let d = soft(&m.cam_dist);
    let target = v3(soft(&m.aim_x), soft(&m.aim_y), soft(&m.aim_z));
    let eye = v3(
        target.x + d * sin(az),
        soft(&m.cam_y),
        target.z + d * cos(az),
    );
    out.camera = Camera {
        eye,
        target,
        up: V3::Y,
        tan_half_fov_y: 0.40,
        near: 0.2,
        far: 120.0,
    };

    let at = v3(0.0, target.y * 0.8, target.z * 0.7 + 0.2);
    out.key = Light {
        pos: v3(-3.2, 10.0, 8.0),
        at,
        color: [1.20, 1.15, 1.06],
        size: 0.9,
    };
    out.fills.clear();
    out.fills.push(Light {
        pos: v3(-10.0, 2.5, 1.5),
        at: v3(0.0, 1.5, 0.5),
        color: [0.10, 0.15, 0.25],
        size: 4.0,
    });
    out.fills.push(Light {
        pos: v3(5.5, 6.0, -8.0),
        at: v3(0.0, 1.5, 0.0),
        color: [0.42, 0.16, 0.80],
        size: 1.6,
    });
    let f = m.flash.at(t);
    out.fills.push(Light {
        pos: muzzle,
        at: v3(0.0, 0.9, 1.0),
        color: [2.4 * f, 0.9 * f, 4.2 * f],
        size: 0.5,
    });
    out.ambient = [0.050, 0.052, 0.062];
    out.floor = Floor {
        base: [0.022, 0.022, 0.027],
        rough: 0.20,
        mirror: 0.5,
    };
    out.exposure = 1.1;
    out.bloom = 0.55 + 0.4 * f;
}

impl Film for Transform {
    fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }
    fn materials(&self) -> &[Material] {
        &self.materials
    }
    fn seconds(&self) -> f32 {
        10.0
    }
    fn frame(&self, t: f32, out: &mut Frame) {
        let (mut loc, mut w) = (Vec::with_capacity(64), Vec::with_capacity(64));
        self.eval_into(t, &mut loc, &mut w, out);
    }
}
