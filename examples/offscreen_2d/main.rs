//! Headless 2D vector rendering to a PNG.
//!
//! Renders a set of filled/stroked vector shapes offscreen via the platform
//! GPU backend and writes the result to `out.png` next to this example.

use vgfx::prelude::*;

fn main() -> vgfx::Result<()> {
    let (w, h) = (960u32, 540u32);
    let mut r = Renderer::offscreen(w, h)?;
    println!("backend: {} ({w}x{h})", r.backend_name());

    r.begin_frame();
    r.set_background(Color::srgb(0.07, 0.08, 0.11, 1.0));

    // Filled shapes.
    r.fill_rect(40.0, 40.0, 220.0, 140.0, Color::srgb(0.20, 0.75, 0.60, 1.0));
    r.fill_rounded_rect(300.0, 60.0, 180.0, 100.0, 24.0, Color::srgb(0.90, 0.55, 0.20, 1.0));
    r.fill_circle(560.0, 120.0, 90.0, Color::srgb(0.85, 0.30, 0.35, 1.0));

    // Stroked shapes.
    r.stroke_rect(40.0, 240.0, 220.0, 140.0, 6.0, Color::srgb(0.30, 0.55, 0.95, 1.0));
    r.stroke_circle(560.0, 320.0, 90.0, 8.0, Color::srgb(0.95, 0.80, 0.30, 1.0));

    // An arbitrary bezier path, filled.
    let mut heart = Path::new();
    heart.move_to(glam::Vec2::new(820.0, 300.0));
    heart.cubic_to(
        glam::Vec2::new(820.0, 260.0),
        glam::Vec2::new(760.0, 240.0),
        glam::Vec2::new(760.0, 300.0),
    );
    heart.cubic_to(
        glam::Vec2::new(760.0, 360.0),
        glam::Vec2::new(820.0, 400.0),
        glam::Vec2::new(820.0, 440.0),
    );
    heart.cubic_to(
        glam::Vec2::new(820.0, 400.0),
        glam::Vec2::new(880.0, 360.0),
        glam::Vec2::new(880.0, 300.0),
    );
    heart.cubic_to(
        glam::Vec2::new(880.0, 240.0),
        glam::Vec2::new(820.0, 260.0),
        glam::Vec2::new(820.0, 300.0),
    );
    r.fill_2d(heart, TwoDStyle::Fill(Color::srgb(0.80, 0.35, 0.70, 1.0)));

    r.present()?;
    let out = "out.png";
    r.save_png(out)?;
    println!("wrote {out}");
    Ok(())
}
