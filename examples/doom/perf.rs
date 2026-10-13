//! FPS meter + process CPU readout (top-right overlay).
//!
//! Frame times feed a 120-sample history drawn as color-banded bars
//! (green < 17ms, yellow < 34ms, red above); the numbers use the bitmap
//! font from `vgfx::text` since the framework has no font engine.

use std::fmt::Write;
use std::time::Instant;

use vgfx::prelude::*;
use vgfx::text;

pub struct Perf {
    history: [f32; 120],
    cursor: usize,
    ema_fps: f32,
    last_process: f64,
    cpu: f32,
    last_cpu: Instant,
}

impl Perf {
    pub fn new() -> Self {
        Self {
            history: [1.0 / 60.0; 120],
            cursor: 0,
            ema_fps: 60.0,
            last_process: process_seconds(),
            cpu: 0.0,
            last_cpu: Instant::now(),
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.history[self.cursor] = dt;
        self.cursor = (self.cursor + 1) % 120;
        self.ema_fps += (1.0 / dt.max(1e-6) - self.ema_fps) * 0.05;
        let elapsed = self.last_cpu.elapsed().as_secs_f64();
        if elapsed > 0.5 {
            let now = process_seconds();
            self.cpu = ((now - self.last_process) / elapsed * 100.0) as f32;
            self.last_process = now;
            self.last_cpu = Instant::now();
        }
    }

    pub fn fps(&self) -> f32 {
        self.ema_fps
    }

    pub fn cpu(&self) -> f32 {
        self.cpu
    }

    pub fn draw(&self, r: &mut Renderer) {
        let (w, _) = r.size();
        let (x0, y0) = (w as f32 - 178.0, 10.0);
        r.fill_rounded_rect(x0, y0, 168.0, 92.0, 8.0, Color::srgb(0.0, 0.0, 0.0, 0.55));

        let mut label = Label {
            bytes: [0; 32],
            len: 0,
        };
        write!(label, "FPS {:>5}", self.fps() as i32).expect("fixed label capacity");
        text::draw_text(r, x0 + 8.0, y0 + 6.0, 2.0, Color::WHITE, label.as_str());
        let cpu_color = if self.cpu < 50.0 {
            Color::srgb(0.3, 0.9, 0.4, 1.0)
        } else if self.cpu < 100.0 {
            Color::srgb(0.95, 0.8, 0.25, 1.0)
        } else {
            Color::srgb(0.95, 0.3, 0.25, 1.0)
        };
        label.len = 0;
        write!(label, "CPU {:>5}%", self.cpu as i32).expect("fixed label capacity");
        text::draw_text(r, x0 + 8.0, y0 + 20.0, 2.0, cpu_color, label.as_str());

        let base = y0 + 86.0;
        let colors = [
            Color::srgb(0.3, 0.85, 0.4, 0.9),
            Color::srgb(0.9, 0.75, 0.2, 0.9),
            Color::srgb(0.9, 0.25, 0.2, 0.9),
        ];
        for i in 0..120 {
            let dt = self.history[(self.cursor + 119 - i) % 120];
            let ms = dt * 1000.0;
            let h = (ms / 50.0 * 44.0).clamp(1.0, 44.0);
            let band = usize::from(ms >= 17.0) + usize::from(ms >= 34.0);
            r.fill_rect(x0 + 160.0 - i as f32, base - h, 1.0, h, colors[band]);
        }
    }
}

struct Label {
    bytes: [u8; 32],
    len: usize,
}
impl Write for Label {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        if text.len() > self.bytes.len() - self.len {
            return Err(std::fmt::Error);
        }
        self.bytes[self.len..self.len + text.len()].copy_from_slice(text.as_bytes());
        self.len += text.len();
        Ok(())
    }
}
impl Label {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("ASCII numeric label")
    }
}

// Standard C process clock; no process enumeration or heap allocation.
fn process_seconds() -> f64 {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn clock() -> std::ffi::c_ulong;
        }
        unsafe { clock() as f64 / 1_000_000.0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}
