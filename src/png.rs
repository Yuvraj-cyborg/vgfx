//! RGBA8 PNG writer: filter 0, stored DEFLATE blocks, fixed scratch space.
use crate::{Error, Result};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
const fn crc_table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut b = 0;
        while b < 8 {
            c = (c >> 1) ^ ((0u32.wrapping_sub(c & 1)) & 0xedb88320);
            b += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}
const CRC: [u32; 256] = crc_table();
fn crc_update(mut c: u32, data: &[u8]) -> u32 {
    for &b in data {
        c = (c >> 8) ^ CRC[((c as u8) ^ b) as usize];
    }
    c
}
fn chunk(w: &mut impl Write, kind: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    w.write_all(&(data.len() as u32).to_be_bytes())?;
    w.write_all(kind)?;
    w.write_all(data)?;
    w.write_all(&(!crc_update(crc_update(!0, kind), data)).to_be_bytes())
}
pub fn save(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let row = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| Error::Io("PNG size overflow".into()))?;
    let pixels = row
        .checked_mul(height as usize)
        .ok_or_else(|| Error::Io("PNG size overflow".into()))?;
    if width == 0 || height == 0 || pixels != rgba.len() {
        return Err(Error::Io("invalid PNG dimensions/pixels".into()));
    }
    let raw = (row + 1)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::Io("PNG size overflow".into()))?;
    let blocks = raw.div_ceil(65535);
    let length = raw
        .checked_add(blocks * 5 + 6)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::Io("PNG IDAT too large".into()))?;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"\x89PNG\r\n\x1a\n")?;
    let mut ihdr = [0; 13];
    ihdr[..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = 6;
    chunk(&mut w, b"IHDR", &ihdr)?;
    w.write_all(&length.to_be_bytes())?;
    w.write_all(b"IDAT")?;
    let mut crc = crc_update(!0, b"IDAT");
    let mut emit = |data: &[u8]| -> std::io::Result<()> {
        crc = crc_update(crc, data);
        w.write_all(data)
    };
    emit(&[0x78, 0x01])?;
    let (mut a, mut b) = (1u32, 0u32);
    let mut scratch = [0u8; 4096];
    let mut offset = 0;
    while offset < raw {
        let len = (raw - offset).min(65535);
        let n = len as u16;
        emit(&[
            u8::from(offset + len == raw),
            n as u8,
            (n >> 8) as u8,
            (!n) as u8,
            ((!n) >> 8) as u8,
        ])?;
        let end = offset + len;
        while offset < end {
            let count = (end - offset).min(scratch.len());
            for (i, byte) in scratch[..count].iter_mut().enumerate() {
                let pos = offset + i;
                let x = pos % (row + 1);
                let y = pos / (row + 1);
                *byte = if x == 0 { 0 } else { rgba[y * row + x - 1] };
                a = (a + *byte as u32) % 65521;
                b = (b + a) % 65521;
            }
            emit(&scratch[..count])?;
            offset += count;
        }
    }
    emit(&((b << 16) | a).to_be_bytes())?;
    w.write_all(&(!crc).to_be_bytes())?;
    chunk(&mut w, b"IEND", &[])?;
    w.flush()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crc_known_vector() {
        assert_eq!(!crc_update(!0, b"123456789"), 0xcbf43926);
    }
    #[test]
    fn invalid_pixels_rejected() {
        assert!(save(Path::new("unused.png"), 2, 2, &[0; 4]).is_err());
    }
}
