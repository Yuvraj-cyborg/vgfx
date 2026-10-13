//! Headless OBJ rendering proof: loads `assets/ogre.obj` (62k verts,
//! no normals — computed on load) and renders it turntable-style.
//!
//! Run with: `cargo run --example offscreen_model` (writes `out.png`).
//! The ogre is CC0 — see `assets/README.md`.

use vgfx::prelude::*;

fn main() -> vgfx::Result<()> {
    let (w, h) = (960u32, 540u32);
    let mut r = Renderer::offscreen(w, h)?;
    println!("backend: {} ({w}x{h})", r.backend_name());

    let mesh = Mesh::load_obj("assets/ogre.obj")?;
    let (mn, mx) = mesh.bounds();
    println!(
        "ogre: {} verts, {} tris, bounds {mn:?} -> {mx:?}",
        mesh.vertex_count(),
        mesh.index_count() / 3
    );
    // Normalize: stand the model ~2m tall, feet on y=0.
    let height = (mx[1] - mn[1]).max(1e-6);
    let s = 2.0 / height;
    let cx = (mn[0] + mx[0]) * 0.5;
    let cz = (mn[2] + mx[2]) * 0.5;
    let model = vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(0.0, 0.0, 0.0))
        * vgfx::math::Mat4::from_scale(vgfx::math::Vec3::splat(s))
        * vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(-cx, -mn[1], -cz))
        * vgfx::math::Mat4::from_rotation_y(0.6);

    r.begin_frame();
    r.set_background(Color::srgb(0.07, 0.08, 0.11, 1.0));
    r.set_camera_perspective(
        45.0,
        w as f32 / h as f32,
        0.1,
        100.0,
        [0.0, 1.4, 5.2],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    // Ground slab for context.
    r.draw_mesh(
        Mesh::cube(1.0),
        vgfx::math::Mat4::from_translation(vgfx::math::Vec3::new(0.0, -0.05, 0.0))
            * vgfx::math::Mat4::from_scale(vgfx::math::Vec3::new(8.0, 0.1, 8.0)),
        Color::srgb(0.16, 0.16, 0.2, 1.0),
    );
    let ogre = std::sync::Arc::new(mesh);
    r.draw_mesh_material(
        ogre,
        model,
        Material {
            color: Color::srgb(0.55, 0.75, 0.45, 1.0),
            ..Material::default()
        },
    );

    r.present()?;
    r.save_png("out.png")?;
    println!("wrote out.png");
    Ok(())
}
