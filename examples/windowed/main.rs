//! Windowed rendering (macOS): a winit window backed by a CAMetalLayer,
//! showing an animated 3D scene with a 2D HUD overlay.
//!
//! Run with: `cargo run --example windowed`

#[cfg(target_vendor = "apple")]
mod app {
    use std::os::raw::c_void;

    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::{Window, WindowAttributes, WindowId};
    use winit::dpi::PhysicalSize;

    use vgfx::backend::metal::attach_layer_to_view;
    use vgfx::prelude::*;

    struct App {
        window: Option<Window>,
        renderer: Option<Renderer>,
        frame: u64,
    }

    impl App {
        fn draw(&mut self) {
            let Some(r) = &mut self.renderer else { return };
            let (w, h) = r.size();
            let t = (self.frame as f32) * 0.016;
            self.frame += 1;

            r.begin_frame();
            r.set_background(Color::srgb(0.08, 0.09, 0.12, 1.0));
            r.set_camera_perspective(
                50.0,
                w as f32 / h.max(1) as f32,
                0.1,
                100.0,
                [0.0, 0.6, 4.5],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            );

            let cube = Mesh::cube(1.4);
            let spin = glam::Mat4::from_rotation_z(t * 0.8) * glam::Mat4::from_rotation_y(t);
            r.draw_mesh(
                &cube,
                spin * glam::Mat4::from_translation(glam::Vec3::new(-1.6, 0.0, 0.0)),
                Color::srgb(0.35, 0.55, 0.95, 1.0),
            );
            let sphere = Mesh::sphere(1.1, 32, 24);
            r.draw_mesh(
                &sphere,
                glam::Mat4::from_translation(glam::Vec3::new(1.5, (t * 0.5).sin() * 0.4, 0.0)),
                Color::srgb(0.95, 0.45, 0.25, 1.0),
            );

            // 2D HUD.
            r.fill_rounded_rect(16.0, 16.0, 240.0, 34.0, 8.0, Color::srgb(0.0, 0.0, 0.0, 0.45));
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
            let ns_view = match window.window_handle().expect("window handle").as_raw() {
                RawWindowHandle::AppKit(h) => h.ns_view.as_ptr() as *mut c_void,
                _ => panic!("expected an AppKit window"),
            };
            let layer = attach_layer_to_view(ns_view, w, h).expect("attach CAMetalLayer");
            let renderer = Renderer::window(w, h, layer).expect("create renderer");
            println!(
                "windowed renderer on backend '{}' ({}x{})",
                renderer.backend_name(),
                w,
                h
            );
            self.window = Some(window);
            self.renderer = Some(renderer);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if size.width > 0 && size.height > 0 {
                        if let Some(r) = &mut self.renderer {
                            r.resize(size.width, size.height);
                        }
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
        let mut app = App { window: None, renderer: None, frame: 0 };
        event_loop.run_app(&mut app).expect("run event loop");
    }
}

fn main() {
    #[cfg(target_vendor = "apple")]
    app::run();
    #[cfg(not(target_vendor = "apple"))]
    eprintln!("the windowed example currently supports macOS (Metal) only");
}
