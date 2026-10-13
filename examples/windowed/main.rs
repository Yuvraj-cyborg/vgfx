//! Windowed rendering (macOS): a native AppKit window backed by a CAMetalLayer,
//! showing an animated 3D scene with a 2D HUD overlay.
//!
//! Drag with the mouse to orbit the camera.
//!
//! Run with: `cargo run --example windowed` (`make run-windowed`)

#[cfg(target_os = "macos")]
mod app {

    use vgfx::backend::metal::attach_layer_to_view;
    use vgfx::prelude::*;
    use vgfx::window::MouseButton;
    use vgfx::window::{
        ActiveEventLoop, ApplicationHandler, ElementState, EventLoop, PhysicalSize, Window,
        WindowAttributes, WindowEvent, WindowId,
    };

    struct App {
        window: Option<Window>,
        renderer: Option<Renderer>,
        frame: u64,
        cube: Option<PreparedMesh>,
        sphere: Option<PreparedMesh>,
        yaw: f32,
        pitch: f32,
        dragging: bool,
        last_mouse: (f64, f64),
    }

    impl App {
        fn draw(&mut self) {
            let Some(r) = &mut self.renderer else { return };
            let (w, h) = r.size();
            let t = (self.frame as f32) * 0.016;
            self.frame += 1;

            r.begin_frame();
            r.set_background(Color::srgb(0.08, 0.09, 0.12, 1.0));
            // Orbit camera (drag to rotate).
            let dist = 5.2;
            let eye = [
                self.yaw.sin() * self.pitch.cos() * dist,
                0.6 + self.pitch.sin() * dist,
                self.yaw.cos() * self.pitch.cos() * dist,
            ];
            r.set_camera_perspective(
                50.0,
                w as f32 / h.max(1) as f32,
                0.1,
                100.0,
                eye,
                [0.0, 0.2, 0.0],
                [0.0, 1.0, 0.0],
            );

            let cube = self.cube.as_ref().expect("prepared cube");
            let spin =
                vgfx::math::Mat4::from_rotation_z(t * 0.8) * vgfx::math::Mat4::from_rotation_y(t);
            r.draw_prepared_mesh(
                cube,
                spin * vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(-1.6, 0.0, 0.0)),
                Material {
                    color: Color::srgb(0.35, 0.55, 0.95, 1.0),
                    ..Material::default()
                },
            );
            let sphere = self.sphere.as_ref().expect("prepared sphere");
            r.draw_prepared_mesh(
                sphere,
                vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(
                    1.5,
                    (t * 0.5).sin() * 0.4,
                    0.0,
                )),
                Material {
                    color: Color::srgb(0.95, 0.45, 0.25, 1.0),
                    ..Material::default()
                },
            );

            // 2D HUD.
            r.fill_rounded_rect(
                16.0,
                16.0,
                240.0,
                34.0,
                8.0,
                Color::srgb(0.0, 0.0, 0.0, 0.45),
            );
            r.stroke_circle(
                w as f32 - 60.0,
                60.0,
                24.0 + (t * 2.0).sin() * 8.0,
                3.0,
                Color::srgb(0.3, 0.9, 0.7, 1.0),
            );

            if let Err(e) = r.present() {
                eprintln!("present failed: {e}");
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("vgfx window")
                        .with_inner_size(PhysicalSize::new(960, 540)),
                )
                .expect("create window");
            let (w, h) = window.inner_size().into();
            let layer = unsafe { attach_layer_to_view(window.ns_view(), w, h) }
                .expect("attach CAMetalLayer");
            let mut renderer = Renderer::window(w, h, layer).expect("create renderer");
            println!(
                "windowed renderer on backend '{}' ({}x{})",
                renderer.backend_name(),
                w,
                h
            );
            self.cube = Some(
                renderer
                    .prepare_mesh(std::sync::Arc::new(Mesh::cube(1.4)))
                    .expect("prepare cube"),
            );
            self.sphere = Some(
                renderer
                    .prepare_mesh(std::sync::Arc::new(Mesh::sphere(1.1, 32, 24)))
                    .expect("prepare sphere"),
            );
            self.window = Some(window);
            self.renderer = Some(renderer);
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
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    self.dragging = state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.dragging {
                        let dx = position.x - self.last_mouse.0;
                        let dy = position.y - self.last_mouse.1;
                        self.yaw -= dx as f32 * 0.005;
                        self.pitch = (self.pitch + dy as f32 * 0.005).clamp(-1.2, 1.2);
                    }
                    self.last_mouse = (position.x, position.y);
                }
                WindowEvent::Resized(size) => {
                    // `size` is already in physical pixels; pass it through
                    // as-is (multiplying by the scale factor again would
                    // double-count on Retina displays).
                    if size.width > 0
                        && size.height > 0
                        && let Some(r) = &mut self.renderer
                    {
                        r.resize(size.width, size.height);
                    }
                }
                WindowEvent::RedrawRequested => {
                    self.draw();
                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    pub fn run() {
        let event_loop = EventLoop::new().expect("event loop");
        let mut app = App {
            window: None,
            renderer: None,
            frame: 0,
            cube: None,
            sphere: None,
            yaw: 0.0,
            pitch: 0.1,
            dragging: false,
            last_mouse: (0.0, 0.0),
        };
        event_loop.run_app(&mut app).expect("run event loop");
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    app::run();
    #[cfg(not(target_os = "macos"))]
    eprintln!("the windowed example currently supports macOS (Metal) only");
}
