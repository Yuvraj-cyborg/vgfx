//! Tiny Doom-like shooter (macOS): one room, five demons, hitscan pistol.
//!
//! Controls:
//! - WASD / arrows: move (Shift: sprint)
//! - Drag mouse: look (click without dragging, or Space: shoot)
//! - R: restart, Esc: quit
//!
//! Run with: `cargo run --example doom` (`make run-doom`)

#[cfg(target_os = "macos")]
mod perf;

#[cfg(target_os = "macos")]
mod app {
    use std::sync::Arc;
    use std::time::Instant;

    use vgfx::backend::metal::attach_layer_to_view;
    use vgfx::math::{Mat4, Vec3};
    use vgfx::prelude::*;
    use vgfx::window::{
        ActiveEventLoop, ApplicationHandler, ElementState, EventLoop, PhysicalSize, Window,
        WindowAttributes, WindowEvent, WindowId,
    };
    use vgfx::window::{KeyCode, KeyEvent, MouseButton, PhysicalKey};

    use crate::perf::Perf;

    const EYE_H: f32 = 1.6;
    const PLAYER_R: f32 = 0.35;
    const WALK: f32 = 4.2;
    const FIRE_CD: f32 = 0.22;
    const SENS: f32 = 0.0035;

    /// Axis-aligned box: wall, floor, or ceiling.
    struct Wall {
        min: Vec3,
        max: Vec3,
        color: Color,
    }

    impl Wall {
        fn center(&self) -> Vec3 {
            (self.min + self.max) * 0.5
        }
        fn size(&self) -> Vec3 {
            self.max - self.min
        }
        fn draw(&self, r: &mut Renderer, shared_cube: &PreparedMesh) {
            let model = Mat4::from_translation(self.center()) * Mat4::from_scale(self.size());
            let mat = Material {
                color: self.color,
                ..Material::default()
            };
            r.draw_prepared_mesh(shared_cube, model, mat);
        }
    }

    struct Enemy {
        pos: Vec3, // center of body
        hp: i32,
        flash: f32,
        dying: f32, // >0 while death animation plays
        speed: f32,
        attack_cd: f32,
        phase: f32,
    }

    impl Enemy {
        fn alive(&self) -> bool {
            self.dying <= 0.0
        }
    }

    struct Impact {
        pos: Vec3,
        ttl: f32,
    }

    struct Game {
        pos: Vec3, // feet position (y = 0)
        yaw: f32,
        pitch: f32,
        hp: f32,
        kills: u32,
        time: f32,
        fire_cd: f32,
        muzzle: f32,
        hurt: f32,
        shake: f32,
        over: Option<bool>, // None = playing, Some(won)
        solids: Vec<Wall>,  // block movement + bullets
        decor: Vec<Wall>,   // block bullets only (floor, ceiling)
        enemies: Vec<Enemy>,
        impacts: Vec<Impact>,
        // input
        fwd: bool,
        back: bool,
        left: bool,
        right: bool,
        sprint: bool,
        dragging: bool,
        last_mouse: (f64, f64),
        press_time: Option<Instant>,
        press_pos: (f64, f64),
        last_title: String,
    }

    fn level() -> (Vec<Wall>, Vec<Wall>, Vec<Enemy>, Vec3) {
        let stone = Color::srgb(0.42, 0.39, 0.36, 1.0);
        let dark = Color::srgb(0.25, 0.23, 0.28, 1.0);
        // Room 24x24 (walls at +-12), height 3.
        let solids = vec![
            Wall {
                min: Vec3::new(-12.0, 0.0, -12.0),
                max: Vec3::new(12.0, 3.0, -11.0),
                color: stone,
            },
            Wall {
                min: Vec3::new(-12.0, 0.0, 11.0),
                max: Vec3::new(12.0, 3.0, 12.0),
                color: stone,
            },
            Wall {
                min: Vec3::new(-12.0, 0.0, -11.0),
                max: Vec3::new(-11.0, 3.0, 11.0),
                color: stone,
            },
            Wall {
                min: Vec3::new(11.0, 0.0, -11.0),
                max: Vec3::new(12.0, 3.0, 11.0),
                color: stone,
            },
            // Four pillars.
            Wall {
                min: Vec3::new(4.0, 0.0, 4.0),
                max: Vec3::new(6.0, 3.0, 6.0),
                color: dark,
            },
            Wall {
                min: Vec3::new(-6.0, 0.0, 4.0),
                max: Vec3::new(-4.0, 3.0, 6.0),
                color: dark,
            },
            Wall {
                min: Vec3::new(4.0, 0.0, -6.0),
                max: Vec3::new(6.0, 3.0, -4.0),
                color: dark,
            },
            Wall {
                min: Vec3::new(-6.0, 0.0, -6.0),
                max: Vec3::new(-4.0, 3.0, -4.0),
                color: dark,
            },
            // Low center cover (blocks movement, shoot over it).
            Wall {
                min: Vec3::new(-3.0, 0.0, -0.5),
                max: Vec3::new(3.0, 1.2, 0.5),
                color: dark,
            },
        ];
        let decor = vec![
            Wall {
                min: Vec3::new(-12.0, -0.1, -12.0),
                max: Vec3::new(12.0, 0.0, 12.0),
                color: Color::srgb(0.14, 0.14, 0.17, 1.0),
            },
            Wall {
                min: Vec3::new(-12.0, 3.0, -12.0),
                max: Vec3::new(12.0, 3.1, 12.0),
                color: Color::srgb(0.05, 0.05, 0.08, 1.0),
            },
        ];
        let spawns = [
            (-7.0, -7.0),
            (7.0, -7.0),
            (-7.0, 3.0),
            (7.0, 3.0),
            (0.0, -8.0),
        ];
        let enemies = spawns
            .iter()
            .enumerate()
            .map(|(i, &(x, z))| Enemy {
                pos: Vec3::new(x, 0.9, z),
                hp: 2,
                flash: 0.0,
                dying: 0.0,
                speed: 1.5 + (i as f32 % 3.0) * 0.35,
                attack_cd: 0.0,
                phase: i as f32 * 1.7,
            })
            .collect();
        (solids, decor, enemies, Vec3::new(0.0, 0.0, 8.0))
    }

    impl Game {
        fn new() -> Self {
            let (solids, decor, enemies, spawn) = level();
            Self {
                pos: spawn,
                yaw: 0.0,
                pitch: 0.0,
                hp: 100.0,
                kills: 0,
                time: 0.0,
                fire_cd: 0.0,
                muzzle: 0.0,
                hurt: 0.0,
                shake: 0.0,
                over: None,
                solids,
                decor,
                enemies,
                impacts: Vec::new(),
                fwd: false,
                back: false,
                left: false,
                right: false,
                sprint: false,
                dragging: false,
                last_mouse: (0.0, 0.0),
                press_time: None,
                press_pos: (0.0, 0.0),
                last_title: String::new(),
            }
        }

        fn dir(&self) -> Vec3 {
            let (y, p) = (self.yaw, self.pitch);
            Vec3::new(-y.sin() * p.cos(), p.sin(), -y.cos() * p.cos())
        }

        fn eye(&self) -> Vec3 {
            // Cheap screen shake: tiny random offset while `shake` lasts.
            let s = if self.shake > 0.0 {
                let t = self.time * 91.7;
                Vec3::new(
                    t.sin() * 0.05 * self.shake,
                    (t * 1.3).cos() * 0.05 * self.shake,
                    0.0,
                )
            } else {
                Vec3::ZERO
            };
            self.pos + Vec3::new(0.0, EYE_H, 0.0) + s
        }

        /// Push a circle (XZ) out of solid boxes.
        fn collide(pos: &mut Vec3, radius: f32, solids: &[Wall]) {
            for w in solids {
                if pos.y + EYE_H < w.min.y || pos.y + 0.3 > w.max.y {
                    continue;
                }
                let cx = pos.x.clamp(w.min.x, w.max.x);
                let cz = pos.z.clamp(w.min.z, w.max.z);
                let dx = pos.x - cx;
                let dz = pos.z - cz;
                let d2 = dx * dx + dz * dz;
                if d2 < radius * radius {
                    if d2 > 1e-9 {
                        let d = d2.sqrt();
                        pos.x = cx + dx / d * radius;
                        pos.z = cz + dz / d * radius;
                    } else {
                        // Center inside the box: push out along smallest exit.
                        let ex = (pos.x - w.min.x).min(w.max.x - pos.x);
                        let ez = (pos.z - w.min.z).min(w.max.z - pos.z);
                        if ex < ez {
                            pos.x = if pos.x - w.min.x < w.max.x - pos.x {
                                w.min.x - radius
                            } else {
                                w.max.x + radius
                            };
                        } else {
                            pos.z = if pos.z - w.min.z < w.max.z - pos.z {
                                w.min.z - radius
                            } else {
                                w.max.z + radius
                            };
                        }
                    }
                }
            }
        }

        fn update(&mut self, dt: f32) {
            self.time += dt;
            self.fire_cd = (self.fire_cd - dt).max(0.0);
            self.muzzle = (self.muzzle - dt).max(0.0);
            self.hurt = (self.hurt - dt * 1.5).max(0.0);
            self.shake = (self.shake - dt * 3.0).max(0.0);
            for e in &mut self.enemies {
                e.flash = (e.flash - dt).max(0.0);
                e.attack_cd = (e.attack_cd - dt).max(0.0);
            }
            for i in &mut self.impacts {
                i.ttl -= dt;
            }
            self.impacts.retain(|i| i.ttl > 0.0);
            if self.over.is_some() {
                return;
            }

            // Move relative to look direction.
            let f = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
            let r = Vec3::new(-f.z, 0.0, f.x);
            let mut wish = Vec3::ZERO;
            if self.fwd {
                wish += f;
            }
            if self.back {
                wish -= f;
            }
            if self.left {
                wish -= r;
            }
            if self.right {
                wish += r;
            }
            if wish.length_squared() > 0.0 {
                let speed = WALK * if self.sprint { 1.6 } else { 1.0 };
                self.pos += wish.normalize() * speed * dt;
                Self::collide(&mut self.pos, PLAYER_R, &self.solids);
            }

            // Enemies chase and maul.
            for i in 0..self.enemies.len() {
                if !self.enemies[i].alive() {
                    self.enemies[i].dying -= dt;
                    continue;
                }
                let to_player = self.pos - self.enemies[i].pos;
                let dist = Vec3::new(to_player.x, 0.0, to_player.z).length();
                if dist > 1.6 {
                    let step = Vec3::new(to_player.x, 0.0, to_player.z).normalize()
                        * self.enemies[i].speed
                        * dt;
                    self.enemies[i].pos += step;
                    Self::collide(&mut self.enemies[i].pos, 0.6, &self.solids);
                } else if self.enemies[i].attack_cd <= 0.0 {
                    self.enemies[i].attack_cd = 0.9;
                    self.hp -= 9.0;
                    self.hurt = 0.6;
                    self.shake = 0.6;
                    if self.hp <= 0.0 {
                        self.hp = 0.0;
                        self.over = Some(false);
                    }
                }
            }
            // Separation so demons don't stack.
            for i in 0..self.enemies.len() {
                for j in (i + 1)..self.enemies.len() {
                    if !self.enemies[i].alive() || !self.enemies[j].alive() {
                        continue;
                    }
                    let d = self.enemies[j].pos - self.enemies[i].pos;
                    let flat = Vec3::new(d.x, 0.0, d.z);
                    let dist = flat.length();
                    if dist < 1.2 && dist > 1e-4 {
                        let push = flat / dist * (1.2 - dist) * 0.5;
                        self.enemies[i].pos -= push;
                        self.enemies[j].pos += push;
                    }
                }
            }

            if self.enemies.iter().all(|e| !e.alive() && e.dying <= 0.0) {
                self.over = Some(true);
            }
        }

        fn ray_aabb(o: Vec3, d: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
            let mut t0 = 0.0f32;
            let mut t1 = f32::MAX;
            for a in 0..3 {
                let (o, d, mn, mx) = (o[a], d[a], min[a], max[a]);
                if d.abs() < 1e-9 {
                    if o < mn || o > mx {
                        return None;
                    }
                } else {
                    let (mut a, mut b) = ((mn - o) / d, (mx - o) / d);
                    if a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    t0 = t0.max(a);
                    t1 = t1.min(b);
                    if t0 > t1 {
                        return None;
                    }
                }
            }
            Some(t0)
        }

        fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, rad: f32) -> Option<f32> {
            let oc = o - c;
            let b = oc.dot(d);
            let cc = oc.length_squared() - rad * rad;
            let disc = b * b - cc;
            if disc < 0.0 {
                return None;
            }
            let t = -b - disc.sqrt();
            if t > 0.0 {
                Some(t)
            } else if -b + disc.sqrt() > 0.0 {
                Some(0.0)
            } else {
                None
            }
        }

        fn shoot(&mut self) {
            if self.fire_cd > 0.0 || self.over.is_some() {
                return;
            }
            self.fire_cd = FIRE_CD;
            self.muzzle = 0.09;
            self.shake = (self.shake + 0.25).min(0.5);
            let o = self.eye();
            let d = self.dir();

            let mut best = f32::MAX;
            let mut hit_enemy: Option<usize> = None;
            for (i, e) in self.enemies.iter().enumerate() {
                if !e.alive() {
                    continue;
                }
                if let Some(t) = Self::ray_sphere(o, d, e.pos, 0.8)
                    && t < best
                {
                    best = t;
                    hit_enemy = Some(i);
                }
            }
            for w in self.solids.iter().chain(self.decor.iter()) {
                if let Some(t) = Self::ray_aabb(o, d, w.min, w.max)
                    && t < best
                {
                    best = t;
                    hit_enemy = None;
                }
            }

            if best >= f32::MAX {
                return;
            }
            match hit_enemy {
                Some(i) => {
                    let e = &mut self.enemies[i];
                    e.hp -= 1;
                    e.flash = 0.18;
                    if e.hp <= 0 {
                        e.dying = 0.4;
                        self.kills += 1;
                    }
                }
                None => self.impacts.push(Impact {
                    pos: o + d * best,
                    ttl: 0.15,
                }),
            }
        }

        fn render(
            &mut self,
            r: &mut Renderer,
            cube: &PreparedMesh,
            ogre: &PreparedMesh,
            ogre_base: Mat4,
            eye_ball: &PreparedMesh,
            perf: &Perf,
        ) {
            r.begin_frame();
            r.set_background(Color::srgb(0.03, 0.03, 0.05, 1.0));
            let eye = self.eye();
            let target = eye + self.dir();
            r.set_camera_perspective(
                72.0,
                {
                    let (w, h) = r.size();
                    w as f32 / h.max(1) as f32
                },
                0.05,
                100.0,
                [eye.x, eye.y, eye.z],
                [target.x, target.y, target.z],
                [0.0, 1.0, 0.0],
            );

            for w in self.solids.iter().chain(self.decor.iter()) {
                w.draw(r, cube);
            }

            // Demons: loaded ogre meshes, hovering + bobbing, flashing on hit.
            // (Hit detection still uses an invisible sphere proxy at e.pos.)
            for e in &self.enemies {
                if !e.alive() && e.dying <= 0.0 {
                    continue;
                }
                let anim = if e.alive() {
                    1.0
                } else {
                    (e.dying / 0.4).clamp(0.0, 1.0)
                };
                let bob = (self.time * 3.0 + e.phase).sin() * 0.08;
                let feet = Vec3::new(e.pos.x, bob, e.pos.z);
                // Face the player. Assumes the model faces +Z (unverified —
                // flip the atan2 args if demons moonwalk at you).
                let to_player = self.pos - e.pos;
                let face_yaw = to_player.x.atan2(to_player.z);
                let body_color = if e.flash > 0.0 {
                    Color::srgb(1.0, 1.0, 1.0, 1.0)
                } else {
                    Color::srgb(0.72 * anim + 0.08, 0.16, 0.13, 1.0)
                };
                r.draw_prepared_mesh(
                    ogre,
                    Mat4::from_translation(feet)
                        * Mat4::from_rotation_y(face_yaw)
                        * Mat4::from_scale(Vec3::splat(anim))
                        * ogre_base,
                    Material {
                        color: body_color,
                        ..Material::default()
                    },
                );
            }

            // Bullet impacts.
            for i in &self.impacts {
                r.draw_prepared_mesh(
                    eye_ball,
                    Mat4::from_translation(i.pos),
                    Material {
                        color: Color::srgb(1.0, 0.85, 0.2, 1.0),
                        ..Material::default()
                    },
                );
            }

            self.hud(r);
            perf.draw(r);
            r.present().expect("present");
        }

        /// Pure-2D overlay: crosshair, health bar, enemy pips, flashes.
        fn hud(&self, r: &mut Renderer) {
            let (w, h) = r.size();
            let (w, h) = (w as f32, h as f32);
            let cx = w * 0.5;
            let cy = h * 0.5;

            // Crosshair.
            let cross = Color::srgb(1.0, 1.0, 1.0, 0.9);
            r.fill_rect(cx - 7.0, cy - 1.0, 14.0, 2.0, cross);
            r.fill_rect(cx - 1.0, cy - 7.0, 2.0, 14.0, cross);
            if self.muzzle > 0.0 {
                r.fill_circle(cx, cy + 26.0, 10.0, Color::srgb(1.0, 0.8, 0.2, 0.9));
            }

            // Health bar (bottom left).
            r.fill_rect(16.0, h - 40.0, 204.0, 20.0, Color::srgb(0.1, 0.0, 0.0, 0.7));
            let frac = (self.hp / 100.0).clamp(0.0, 1.0);
            r.fill_rect(
                18.0,
                h - 38.0,
                200.0 * frac,
                16.0,
                Color::srgb(1.0 - frac * 0.8, 0.15 + frac * 0.6, 0.15, 1.0),
            );

            // Enemy pips (top center): red alive, dark dead.
            let n = self.enemies.len() as f32;
            for (i, e) in self.enemies.iter().enumerate() {
                let x = cx - n * 9.0 + i as f32 * 18.0;
                let c = if e.alive() {
                    Color::srgb(0.8, 0.15, 0.12, 1.0)
                } else {
                    Color::srgb(0.2, 0.2, 0.22, 1.0)
                };
                r.fill_rect(x, 14.0, 14.0, 14.0, c);
            }

            // Damage vignette.
            if self.hurt > 0.0 {
                let a = (self.hurt * 0.7).min(0.6);
                let c = Color::srgb(0.8, 0.05, 0.05, a);
                r.fill_rect(0.0, 0.0, w, 36.0, c);
                r.fill_rect(0.0, h - 36.0, w, 36.0, c);
                r.fill_rect(0.0, 0.0, 36.0, h, c);
                r.fill_rect(w - 36.0, 0.0, 36.0, h, c);
            }

            // Win/lose tint (words live in the window title — no font engine).
            match self.over {
                Some(true) => r.fill_rect(0.0, 0.0, w, h, Color::srgb(0.1, 0.8, 0.2, 0.18)),
                Some(false) => r.fill_rect(0.0, 0.0, w, h, Color::srgb(0.8, 0.05, 0.05, 0.3)),
                None => {}
            }
        }

        fn title(&self) -> String {
            let alive = self.enemies.iter().filter(|e| e.alive()).count();
            match self.over {
                Some(true) => {
                    format!(
                        "vgfx doom — LEVEL CLEAR! kills {} — R to restart",
                        self.kills
                    )
                }
                Some(false) => "vgfx doom — YOU DIED — R to restart".into(),
                None => format!(
                    "vgfx doom — HP {}  demons left {}  (WASD + drag, click to shoot)",
                    self.hp as i32, alive
                ),
            }
        }
    }

    struct App {
        window: Option<Window>,
        renderer: Option<Renderer>,
        game: Game,
        last: Instant,
        perf: Perf,
        // Shared meshes (built once, not per frame).
        cube: Option<PreparedMesh>,
        ogre: Option<PreparedMesh>,
        ogre_base: Mat4,
        eye_ball: Option<PreparedMesh>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("vgfx doom")
                        .with_inner_size(PhysicalSize::new(960, 540)),
                )
                .expect("create window");
            let (w, h) = window.inner_size().into();
            let layer = unsafe { attach_layer_to_view(window.ns_view(), w, h) }
                .expect("attach CAMetalLayer");
            let mut renderer = Renderer::window(w, h, layer).expect("create renderer");
            self.cube = Some(
                renderer
                    .prepare_mesh(Arc::new(Mesh::cube(1.0)))
                    .expect("prepare cube"),
            );
            // Ogre asset (assets/ogre.obj): normalize to ~1.7m tall, feet at y=0.
            let ogre = Mesh::load_obj("assets/ogre.obj").expect("load assets/ogre.obj");
            let (mn, mx) = ogre.bounds();
            let s = 1.7 / (mx[1] - mn[1]).max(1e-6);
            self.ogre_base = Mat4::from_scale(Vec3::splat(s))
                * Mat4::from_translation(Vec3::new(
                    -(mn[0] + mx[0]) * 0.5,
                    -mn[1],
                    -(mn[2] + mx[2]) * 0.5,
                ));
            self.ogre = Some(renderer.prepare_mesh(Arc::new(ogre)).expect("prepare ogre"));
            self.eye_ball = Some(
                renderer
                    .prepare_mesh(Arc::new(Mesh::sphere(0.16, 12, 8)))
                    .expect("prepare sphere"),
            );
            self.window = Some(window);
            self.renderer = Some(renderer);
            self.last = Instant::now();
            self.window.as_ref().expect("window").request_redraw();
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state,
                            ..
                        },
                    ..
                } => {
                    let down = state == ElementState::Pressed;
                    match physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.game.fwd = down
                        }
                        PhysicalKey::Code(KeyCode::KeyS)
                        | PhysicalKey::Code(KeyCode::ArrowDown) => self.game.back = down,
                        PhysicalKey::Code(KeyCode::KeyA)
                        | PhysicalKey::Code(KeyCode::ArrowLeft) => self.game.left = down,
                        PhysicalKey::Code(KeyCode::KeyD)
                        | PhysicalKey::Code(KeyCode::ArrowRight) => self.game.right = down,
                        PhysicalKey::Code(KeyCode::ShiftLeft)
                        | PhysicalKey::Code(KeyCode::ShiftRight) => self.game.sprint = down,
                        PhysicalKey::Code(KeyCode::Space) => {
                            if down {
                                self.game.shoot()
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            if down {
                                self.game = Game::new();
                            }
                        }
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    if state == ElementState::Pressed {
                        self.game.dragging = true;
                        self.game.press_time = Some(Instant::now());
                        self.game.press_pos = self.game.last_mouse;
                    } else {
                        // Click (not drag) = shoot.
                        let quick = self
                            .game
                            .press_time
                            .map(|t| t.elapsed().as_secs_f32() < 0.25)
                            .unwrap_or(false);
                        let (x0, y0) = self.game.press_pos;
                        let (x1, y1) = self.game.last_mouse;
                        let still = (x1 - x0).hypot(y1 - y0) < 8.0;
                        self.game.dragging = false;
                        self.game.press_time = None;
                        if quick && still {
                            self.game.shoot();
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.game.dragging {
                        let dx = position.x - self.game.last_mouse.0;
                        let dy = position.y - self.game.last_mouse.1;
                        self.game.yaw -= dx as f32 * SENS;
                        self.game.pitch = (self.game.pitch - dy as f32 * SENS).clamp(-1.45, 1.45);
                    }
                    self.game.last_mouse = (position.x, position.y);
                }
                WindowEvent::Resized(size) => {
                    if size.width > 0
                        && size.height > 0
                        && let Some(r) = &mut self.renderer
                    {
                        r.resize(size.width, size.height);
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(self.last).as_secs_f32().min(0.05);
                    self.last = now;
                    self.perf.tick(dt);
                    if let (Some(r), Some(cube), Some(ogre), Some(eye)) =
                        (&mut self.renderer, &self.cube, &self.ogre, &self.eye_ball)
                    {
                        let base = self.ogre_base;
                        self.game.update(dt);
                        self.game.render(r, cube, ogre, base, eye, &self.perf);
                    }
                    if let Some(win) = &self.window {
                        let mut title = self.game.title();
                        title.push_str(&format!(
                            "  |  {} FPS  CPU {}%",
                            self.perf.fps() as i32,
                            self.perf.cpu() as i32
                        ));
                        if title != self.game.last_title {
                            win.set_title(&title);
                            self.game.last_title = title;
                        }
                        win.request_redraw();
                    }
                }
            }
        }
    }

    pub fn run() {
        let event_loop = EventLoop::new().expect("event loop");
        let mut app = App {
            window: None,
            renderer: None,
            game: Game::new(),
            last: Instant::now(),
            perf: Perf::new(),
            cube: None,
            ogre: None,
            ogre_base: Mat4::IDENTITY,
            eye_ball: None,
        };
        event_loop.run_app(&mut app).expect("run event loop");
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ray_hits_sphere() {
            let o = Vec3::new(0.0, 1.6, 8.0);
            let d = Vec3::new(0.0, 0.0, -1.0);
            let t = Game::ray_sphere(o, d, Vec3::new(0.0, 1.6, 0.0), 0.65).unwrap();
            assert!((t - (8.0 - 0.65)).abs() < 1e-4);
            assert!(Game::ray_sphere(o, d, Vec3::new(5.0, 1.6, 0.0), 0.65).is_none());
        }

        #[test]
        fn ray_hits_wall() {
            let o = Vec3::new(0.0, 1.6, 8.0);
            let d = Vec3::new(0.0, 0.0, -1.0);
            let t = Game::ray_aabb(
                o,
                d,
                Vec3::new(-12.0, 0.0, -12.0),
                Vec3::new(12.0, 3.0, -11.0),
            )
            .unwrap();
            assert!((t - 19.0).abs() < 1e-4);
            // Parallel ray missing the box.
            assert!(
                Game::ray_aabb(
                    Vec3::new(0.0, 5.0, 8.0),
                    Vec3::new(0.0, 0.0, -1.0),
                    Vec3::new(-12.0, 0.0, -12.0),
                    Vec3::new(12.0, 3.0, -11.0),
                )
                .is_none()
            );
        }

        #[test]
        fn collision_pushes_out() {
            let wall = Wall {
                min: Vec3::new(4.0, 0.0, 4.0),
                max: Vec3::new(6.0, 3.0, 6.0),
                color: Color::BLACK,
            };
            let mut p = Vec3::new(4.2, 0.0, 5.0);
            Game::collide(&mut p, PLAYER_R, &[wall]);
            assert!(p.x <= 4.0 - PLAYER_R + 1e-4, "pushed west, got {p:?}");
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    app::run();
    #[cfg(not(target_os = "macos"))]
    eprintln!("the doom example currently supports macOS (Metal) only");
}
