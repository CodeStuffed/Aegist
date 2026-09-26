//! The screen: finished output scrolls up into the terminal's history like
//! any program's, while a live region at the bottom - the input box, the
//! spinner, code being written - is redrawn in place. Nothing takes over
//! the whole terminal, so scrollback, copy and paste keep working.

use super::text;
use std::io::Write;

pub struct Screen {
    /// Rows the live region took when last drawn.
    live: usize,
    /// Row (within the live region) the cursor was left on.
    cursor_row: usize,
    pub width: usize,
    pub height: usize,
}

impl Default for Screen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen {
    pub fn new() -> Screen {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 24));
        Screen { live: 0, cursor_row: 0, width: w.max(20) as usize, height: h.max(8) as usize }
    }

    pub fn resize(&mut self, w: u16, h: u16) {
        self.width = w.max(20) as usize;
        self.height = h.max(8) as usize;
    }

    /// Move to the top of the live region and clear it.
    fn clear_live(&mut self, buf: &mut String) {
        buf.push('\r');
        if self.cursor_row > 0 {
            buf.push_str(&format!("\x1b[{}A", self.cursor_row));
        }
        buf.push_str("\x1b[J");
        self.live = 0;
        self.cursor_row = 0;
    }

    /// Write `permanent` lines above the live region (they scroll into
    /// history), then draw `live` with the cursor at `cursor` (row, column)
    /// or hidden.
    pub fn frame(&mut self, out: &mut impl Write, permanent: &[String], live: &[String], cursor: Option<(usize, usize)>) {
        let mut buf = String::from("\x1b[?2026h"); // hold the redraw until it's complete (no flicker)
        self.clear_live(&mut buf);
        for line in permanent {
            buf.push_str(line);
            buf.push_str("\x1b[0m\x1b[K\r\n");
        }
        // the live region must fit on screen: keep its bottom (the input box)
        let max = self.height.saturating_sub(1).max(1);
        let skip = live.len().saturating_sub(max);
        let shown = &live[skip..];
        let cursor = cursor.and_then(|(r, c)| r.checked_sub(skip).map(|r| (r, c)));
        for (i, line) in shown.iter().enumerate() {
            if i > 0 {
                buf.push_str("\r\n");
            }
            buf.push_str(&text::truncate(line, self.width.saturating_sub(1)));
            buf.push_str("\x1b[0m\x1b[K");
        }
        self.live = shown.len();
        match cursor {
            Some((row, col)) if row < shown.len() => {
                let up = shown.len() - 1 - row;
                if up > 0 {
                    buf.push_str(&format!("\x1b[{up}A"));
                }
                buf.push_str(&format!("\r\x1b[{}C", col).replace("\x1b[0C", ""));
                buf.push_str("\x1b[?25h");
                self.cursor_row = row;
            }
            _ => {
                buf.push_str("\x1b[?25l");
                self.cursor_row = self.live.saturating_sub(1);
            }
        }
        buf.push_str("\x1b[?2026l");
        let _ = out.write_all(buf.as_bytes());
        let _ = out.flush();
    }

    /// Remove the live region (before leaving, or before another program
    /// writes to the terminal).
    pub fn close(&mut self, out: &mut impl Write) {
        let mut buf = String::new();
        self.clear_live(&mut buf);
        buf.push_str("\x1b[?25h");
        let _ = out.write_all(buf.as_bytes());
        let _ = out.flush();
    }

    /// Clear the whole visible screen (history stays in the scrollback).
    pub fn clear_all(&mut self, out: &mut impl Write) {
        let _ = out.write_all(b"\x1b[2J\x1b[3J\x1b[H");
        let _ = out.flush();
        self.live = 0;
        self.cursor_row = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_redraw_in_place() {
        let mut s = Screen { live: 0, cursor_row: 0, width: 40, height: 10 };
        let mut out: Vec<u8> = Vec::new();
        s.frame(&mut out, &["done".into()], &["live 1".into(), "live 2".into()], Some((0, 3)));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("done\x1b[0m\x1b[K\r\nlive 1"));
        assert!(text.contains("\x1b[1A\r\x1b[3C\x1b[?25h"));
        assert_eq!((s.live, s.cursor_row), (2, 0));
        let mut out: Vec<u8> = Vec::new();
        s.frame(&mut out, &[], &["x".into()], None);
        let text = String::from_utf8(out).unwrap();
        // back to the top of the old live region (cursor was on its row 0), cleared
        assert!(text.starts_with("\x1b[?2026h\r\x1b[J"));
        // a live region taller than the screen keeps its bottom
        let many: Vec<String> = (0..30).map(|i| format!("row {i}")).collect();
        let mut out: Vec<u8> = Vec::new();
        s.frame(&mut out, &[], &many, Some((29, 0)));
        assert_eq!(s.live, 9);
        assert!(String::from_utf8(out).unwrap().contains("row 29"));
    }
}
