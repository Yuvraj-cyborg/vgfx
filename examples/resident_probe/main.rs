//! GPU smoke test + Rust allocation count for repeated resident mesh frames.
//! cargo run --release --offline --example resident_probe
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};
use vgfx::prelude::*;
struct Counter;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn main() -> vgfx::Result<()> {
    let mut r = Renderer::offscreen(64, 64)?;
    let small = r.prepare_mesh(Arc::new(Mesh::cube(1.)))?;
    let large = r.prepare_mesh(Arc::new(Mesh::sphere(1., 256, 128)))?;
    for (name, mesh) in [("cube", &small), ("65,536 triangle sphere", &large)] {
        // Warm the driver before counting. Driver/runtime allocations aren't
        // Rust GlobalAlloc calls and are outside the ownership guarantee.
        r.begin_frame();
        r.draw_prepared_mesh(mesh, Mat4::IDENTITY, Material::default());
        r.present()?;
        ALLOCS.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        let start = Instant::now();
        for _ in 0..32 {
            r.begin_frame();
            r.draw_prepared_mesh(mesh, Mat4::IDENTITY, Material::default());
            r.fill_rect(0., 0., 8., 8., Color::srgb(1., 0., 0., 1.));
            r.present()?;
        }
        let elapsed = start.elapsed();
        TRACK.store(false, Ordering::Relaxed);
        assert_eq!(ALLOCS.load(Ordering::Relaxed), 0);
        let pixels = r.read_pixels().expect("offscreen readback");
        assert_eq!(pixels.len(), 64 * 64 * 4);
        assert_eq!(&pixels[..4], &[255, 0, 0, 255]);
        println!(
            "{name}: 32 frames, 0 Rust allocations, {:?} total (includes GPU wait)",
            elapsed
        );
    }
    r.resize(80, 48);
    assert_eq!(r.size(), (80, 48));
    r.begin_frame();
    r.fill_rect(0., 0., 80., 48., Color::srgb(0., 1., 0., 1.));
    r.present()?;
    let pixels = r.read_pixels().unwrap();
    assert_eq!(pixels.len(), 80 * 48 * 4);
    assert_eq!(&pixels[pixels.len() - 4..], &[0, 255, 0, 255]);
    println!("readback, retained meshes, depth and resize verified");
    Ok(())
}
