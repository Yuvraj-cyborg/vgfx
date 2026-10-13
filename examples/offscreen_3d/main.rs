//! Headless 3D mesh rendering to a PNG.
//!
//! Renders a lit cube and sphere with a perspective camera offscreen and
//! writes the result to `out.png`.

use vgfx::prelude::*;

fn main() -> vgfx::Result<()> {
    let (w, h) = (960u32, 540u32);
    let mut r = Renderer::offscreen(w, h)?;
    println!("backend: {} ({w}x{h})", r.backend_name());

    r.begin_frame();
    r.set_background(Color::srgb(0.10, 0.10, 0.14, 1.0));
    r.set_camera_perspective(
        50.0,
        w as f32 / h as f32,
        0.1,
        100.0,
        [0.0, 0.6, 4.5], // eye
        [0.0, 0.0, 0.0], // target
        [0.0, 1.0, 0.0], // up
    );

    let cube = Mesh::cube(1.4);
    let spin = vgfx::math::Mat4::from_rotation_z(0.7) * vgfx::math::Mat4::from_rotation_y(0.9);
    r.draw_mesh(
        cube,
        spin * vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(-1.6, 0.0, 0.0)),
        Color::srgb(0.35, 0.55, 0.95, 1.0),
    );

    let sphere = Mesh::sphere(1.1, 32, 24);
    r.draw_mesh(
        sphere,
        vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(1.5, 0.0, 0.0)),
        Color::srgb(0.95, 0.45, 0.25, 1.0),
    );

    // 2D overlay on top of the 3D scene.
    r.fill_rounded_rect(
        20.0,
        h as f32 - 60.0,
        320.0,
        40.0,
        10.0,
        Color::srgb(0.0, 0.0, 0.0, 0.45),
    );

    r.present()?;
    let out = "out.png";
    r.save_png(out)?;
    println!("wrote {out}");
    Ok(())
}
