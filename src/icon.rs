//! Aegist's icon, drawn in code: a four-point star in the session's aurora
//! colors (cyan, violet, pink) on a dark rounded square. Used for the app
//! window and for the shortcuts `aegist install` makes.

/// The icon, `size`×`size`, as RGBA bytes row by row.
pub fn rgba(size: u32) -> Vec<u8> {
    let n = size as f32;
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    const SS: u32 = 4; // samples per pixel side, for smooth edges
    let lerp = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    let (cyan, violet, pink) = ([94.0, 234.0, 212.0], [167.0, 139.0, 250.0], [244.0, 114.0, 182.0]);
    for py in 0..size {
        for px in 0..size {
            let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for sy in 0..SS {
                for sx in 0..SS {
                    // -1..1 across the icon
                    let x = ((px * SS + sx) as f32 + 0.5) / (n * SS as f32) * 2.0 - 1.0;
                    let y = ((py * SS + sy) as f32 + 0.5) / (n * SS as f32) * 2.0 - 1.0;
                    // rounded square
                    let (qx, qy, rad) = (x.abs() - (0.94 - 0.36), y.abs() - (0.94 - 0.36), 0.36);
                    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - rad;
                    if outside > 0.0 {
                        continue;
                    }
                    let t = (y + 1.0) / 2.0;
                    let mut c = lerp([30.0, 34.0, 56.0], [12.0, 13.0, 22.0], t);
                    // the star: an astroid, |x|^(2/3) + |y|^(2/3) <= r^(2/3)
                    let star = x.abs().powf(2.0 / 3.0) + y.abs().powf(2.0 / 3.0);
                    let r0 = 0.72f32.powf(2.0 / 3.0);
                    let d = (x - y) * 0.5 + 0.5; // along the diagonal
                    let aurora = if d < 0.5 { lerp(cyan, violet, d * 2.0) } else { lerp(violet, pink, (d - 0.5) * 2.0) };
                    if star <= r0 {
                        c = lerp(aurora, [255.0, 255.0, 255.0], (1.0 - (x * x + y * y).sqrt() / 0.25).clamp(0.0, 1.0) * 0.55);
                    } else {
                        // a soft glow around it
                        let glow = (1.0 - (star - r0) / 0.55).clamp(0.0, 1.0).powi(3) * 0.45;
                        c = lerp(c, aurora, glow);
                    }
                    r += c[0];
                    g += c[1];
                    b += c[2];
                    a += 1.0;
                }
            }
            let k = (SS * SS) as f32;
            if a == 0.0 {
                out.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                out.extend_from_slice(&[(r / a) as u8, (g / a) as u8, (b / a) as u8, (a / k * 255.0).round() as u8]);
            }
        }
    }
    out
}

/// The icon as a PNG file's bytes.
pub fn png(size: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut enc = ::png::Encoder::new(&mut bytes, size, size);
        enc.set_color(::png::ColorType::Rgba);
        enc.set_depth(::png::BitDepth::Eight);
        let mut w = enc.write_header().expect("in-memory PNG");
        w.write_image_data(&rgba(size)).expect("in-memory PNG");
    }
    bytes
}

/// The icon as a Windows .ico file (PNG images at 256, 48, 32 and 16 pixels).
pub fn ico() -> Vec<u8> {
    let sizes = [256u32, 48, 32, 16];
    let images: Vec<Vec<u8>> = sizes.iter().map(|&s| png(s)).collect();
    let mut out = vec![0, 0, 1, 0];
    out.extend_from_slice(&(sizes.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * sizes.len() as u32;
    for (s, img) in sizes.iter().zip(&images) {
        let side = if *s >= 256 { 0 } else { *s as u8 }; // 0 means 256
        out.extend_from_slice(&[side, side, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes()); // color planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        out.extend_from_slice(&(img.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += img.len() as u32;
    }
    for img in images {
        out.extend_from_slice(&img);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_icon_is_a_star_on_a_rounded_square() {
        let n = 64;
        let px = rgba(n);
        assert_eq!(px.len(), (n * n * 4) as usize);
        let at = |x: u32, y: u32| &px[((y * n + x) * 4) as usize..((y * n + x) * 4 + 4) as usize];
        assert_eq!(at(0, 0)[3], 0, "the corners are clear");
        assert_eq!(at(32, 32)[3], 255);
        assert!(at(32, 32)[0] > 150 && at(32, 32)[2] > 150, "the star's middle is bright: {:?}", at(32, 32));
        assert!(at(10, 10)[0] < 90, "the background is dark: {:?}", at(10, 10));
        let ico = ico();
        assert_eq!(&ico[..6], &[0, 0, 1, 0, 4, 0]);
        assert_eq!(&ico[6 + 64..6 + 64 + 8], b"\x89PNG\r\n\x1a\n");
    }
}
