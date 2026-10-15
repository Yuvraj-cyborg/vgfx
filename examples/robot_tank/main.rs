//! Renders the robot-to-tank film with the studio renderer, on the CPU, as
//! raw RGB frames.
//!
//!   cargo run --release --example robot_tank -- [--size 1920x1080] [--fps 30] [--out film.rgb]
//!   ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i film.rgb -c:v libx264 -pix_fmt yuv420p film.mp4
// The film's coordinates are measured, not constants (0.318 is a height,
// not 1/pi), and it was written inside the studio module, under its rules.
#![allow(
    clippy::approx_constant,
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::neg_multiply,
    clippy::too_many_arguments
)]

mod film;

use std::io::Write;
use std::time::Instant;
use vgfx::studio::{Film, Frame, StudioRenderer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let size = arg("--size").unwrap_or_else(|| "1920x1080".into());
    let (w, h) = size
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .expect("--size WxH");
    let fps: f32 = arg("--fps").map_or(30.0, |v| v.parse().expect("--fps"));
    let out = arg("--out").unwrap_or_else(|| "film.rgb".into());

    let film = film::Transform::new();
    let (parts, triangles) = film.counts();
    eprintln!("{parts} parts, {triangles} triangles");
    let mut renderer = StudioRenderer::new(w, h);
    let mut frame = Frame::default();
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("--out"));
    let frames = (film.seconds() * fps).round() as usize;
    let start = Instant::now();
    for i in 0..frames {
        film.frame(i as f32 / fps, &mut frame);
        file.write_all(renderer.render(film.meshes(), film.materials(), &frame))
            .expect("write");
    }
    file.flush().expect("write");
    let s = start.elapsed().as_secs_f64();
    eprintln!(
        "{frames} frames of {w}x{h} in {s:.1} s, {:.0} ms a frame, one thread: {out}",
        s * 1000.0 / frames as f64
    );
}
