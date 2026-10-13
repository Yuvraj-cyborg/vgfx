//! Tiny 3x5 bitmap font for HUD text (FPS meters, labels).
//!
//! No font files, no dependencies: each glyph is 5 rows of 3 bits.
//! Each character records 15 cells (transparent when unset), coalesced into
//! the renderer's 2D batch. Glyph content never changes the cell count.
//!
//! ```text
//! use vgfx::text;
//! // inside a frame, after begin_frame():
//! // text::draw_text(&mut renderer, 10.0, 10.0, 2.0, Color::WHITE, "FPS 60");
//! ```

use crate::scene::Color;

/// One glyph: 5 rows, low 3 bits per row.
type Glyph = [u8; 5];

fn glyph(c: char) -> Glyph {
    match c {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        '(' => [0b001, 0b010, 0b010, 0b010, 0b001],
        ')' => [0b100, 0b010, 0b010, 0b010, 0b100],
        '!' => [0b010, 0b010, 0b010, 0b000, 0b010],
        _ => [0b000, 0b000, 0b000, 0b000, 0b000],
    }
}

/// Pixel width of `text` at `scale` (3px glyph + 1px gap per char).
#[must_use]
pub fn text_width(text: &str, scale: f32) -> f32 {
    let n = text.chars().count().max(1) as f32;
    (n * 4.0 - 1.0) * scale
}

/// Draw one line of text (uppercased; unknown chars render as space).
/// `(x, y)` is the top-left corner, `scale` the pixel size.
pub fn draw_text(
    r: &mut crate::backend::Renderer,
    x: f32,
    y: f32,
    scale: f32,
    color: Color,
    text: &str,
) {
    for (i, c) in text.chars().map(|c| c.to_ascii_uppercase()).enumerate() {
        let ox = x + i as f32 * 4.0 * scale;
        for (row, bits) in glyph(c).iter().enumerate() {
            for col in 0..3 {
                {
                    let color = Color {
                        a: color.a * ((bits >> (2 - col)) & 1) as f32,
                        ..color
                    };
                    let px = ox + col as f32 * scale;
                    let py = y + row as f32 * scale;
                    r.fill_rect(px, py, scale, scale, color);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_counts_cells() {
        assert_eq!(text_width("FPS 60", 2.0), (6.0 * 4.0 - 1.0) * 2.0);
    }

    #[test]
    fn glyphs_have_pixels() {
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:%-+()!/".chars() {
            assert!(glyph(c).iter().any(|&r| r != 0), "empty glyph {c}");
        }
    }
}
