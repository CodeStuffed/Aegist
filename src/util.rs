//! Small shared helpers: timestamps, file writes, hashing.

use anyhow::Result;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// UTC timestamp like 2026-09-24T12:34:56+00:00.
pub fn iso_now() -> String {
    iso_from_unix(unix_now() as i64)
}

pub fn iso_from_unix(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // civil-from-days (Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Write via a temp file and rename, so a crash never leaves half a file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// FNV-1a: a small stable hash (unlike std's, it never changes between versions).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

pub fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Greedy word wrap for terminal output: lines at most `width` characters,
/// the first starting with indent + bullet, the rest hanging under it.
pub fn wrap(text: &str, width: usize, indent: &str, bullet: &str) -> String {
    let hang = " ".repeat(indent.chars().count() + bullet.chars().count());
    let mut lines = Vec::new();
    let mut line = format!("{indent}{bullet}");
    let mut empty = true;
    for word in text.split_whitespace() {
        if !empty && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::replace(&mut line, hang.clone()));
            empty = true;
        }
        if !empty {
            line.push(' ');
        }
        line.push_str(word);
        empty = false;
    }
    lines.push(line);
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates() {
        assert_eq!(iso_from_unix(0), "1970-01-01T00:00:00+00:00");
        assert_eq!(iso_from_unix(1_790_000_000), "2026-09-21T14:13:20+00:00");
        assert_eq!(iso_from_unix(951_782_400), "2000-02-29T00:00:00+00:00");
    }

    #[test]
    fn wrapping() {
        let w = wrap("one two three four five six", 12, "  ", "- ");
        assert_eq!(w, "  - one two\n    three\n    four\n    five six");
        assert_eq!(wrap("hi", 80, "", ""), "hi");
    }

    #[test]
    fn f32_roundtrip() {
        let v = vec![1.5f32, -2.25, 1e-8];
        assert_eq!(bytes_to_f32s(&f32s_to_bytes(&v)), v);
    }
}
