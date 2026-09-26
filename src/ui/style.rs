//! Color and text style: 24-bit color where the terminal has it, the
//! nearest 256- or 16-color match where it doesn't, and none at all when
//! output isn't a terminal or NO_COLOR is set.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const fn hex(h: u32) -> Rgb {
        Rgb((h >> 16) as u8, (h >> 8) as u8, h as u8)
    }

    pub fn lerp(self, o: Rgb, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let m = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
        Rgb(m(self.0, o.0), m(self.1, o.1), m(self.2, o.2))
    }

    /// Mixed toward black (0) or its full self (1).
    pub fn shade(self, t: f32) -> Rgb {
        Rgb(0, 0, 0).lerp(self, t)
    }
}

/// The palette: "aurora" - cool cyans and violets warming into pink, on
/// whatever background the terminal has.
pub mod pal {
    use super::Rgb;
    pub const CYAN: Rgb = Rgb::hex(0x5EEAD4);
    pub const SKY: Rgb = Rgb::hex(0x7DD3FC);
    pub const BLUE: Rgb = Rgb::hex(0x7AA2F7);
    pub const VIOLET: Rgb = Rgb::hex(0xA78BFA);
    pub const PINK: Rgb = Rgb::hex(0xF472B6);
    pub const GREEN: Rgb = Rgb::hex(0x34D399);
    pub const YELLOW: Rgb = Rgb::hex(0xFBBF24);
    pub const ORANGE: Rgb = Rgb::hex(0xFB923C);
    pub const RED: Rgb = Rgb::hex(0xF87171);
    pub const DIM: Rgb = Rgb::hex(0x9AA0B4);
    pub const FAINT: Rgb = Rgb::hex(0x646A80);
    pub const BORDER: Rgb = Rgb::hex(0x4B5068);
    pub const ADD_BG: Rgb = Rgb::hex(0x0E2A20);
    pub const DEL_BG: Rgb = Rgb::hex(0x3A1520);
    pub const ADD_FG: Rgb = Rgb::hex(0x6EE7B7);
    pub const DEL_FG: Rgb = Rgb::hex(0xFDA4AF);
    pub const UNSURE_BG: Rgb = Rgb::hex(0x3B2A0F);

    /// The signature gradient, 0 = cyan, 1 = pink.
    pub fn aurora(t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        if t < 0.5 { CYAN.lerp(VIOLET, t * 2.0) } else { VIOLET.lerp(PINK, (t - 0.5) * 2.0) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    None = 0,
    Ansi16 = 1,
    Ansi256 = 2,
    True = 3,
}

static DEPTH: AtomicU8 = AtomicU8::new(255);

fn detect() -> Depth {
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
        return Depth::None;
    }
    let forced = std::env::var_os("FORCE_COLOR").is_some_and(|v| v != "0");
    if !forced && !std::io::stdout().is_terminal() {
        return Depth::None;
    }
    let term = env("TERM");
    if term == "dumb" && !forced {
        return Depth::None;
    }
    let colorterm = env("COLORTERM").to_ascii_lowercase();
    if colorterm.contains("truecolor") || colorterm.contains("24bit") || std::env::var_os("WT_SESSION").is_some() || cfg!(windows) {
        return Depth::True;
    }
    match env("TERM_PROGRAM").as_str() {
        "iTerm.app" | "WezTerm" | "vscode" | "Hyper" | "ghostty" | "Tabby" | "rio" => return Depth::True,
        "Apple_Terminal" => return Depth::Ansi256,
        _ => {}
    }
    if term.contains("kitty") || term.contains("alacritty") || term.contains("foot") || term.contains("direct") {
        return Depth::True;
    }
    if term.contains("256") {
        Depth::Ansi256
    } else {
        Depth::Ansi16
    }
}

pub fn depth() -> Depth {
    match DEPTH.load(Ordering::Relaxed) {
        0 => Depth::None,
        1 => Depth::Ansi16,
        2 => Depth::Ansi256,
        3 => Depth::True,
        _ => {
            let d = detect();
            DEPTH.store(d as u8, Ordering::Relaxed);
            d
        }
    }
}

pub fn set_depth(d: Depth) {
    DEPTH.store(d as u8, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    depth() != Depth::None
}

fn to_256(c: Rgb) -> u8 {
    let (r, g, b) = (c.0 as i32, c.1 as i32, c.2 as i32);
    // gray ramp when the color is nearly gray
    if (r - g).abs() < 10 && (g - b).abs() < 10 {
        let avg = (r + g + b) / 3;
        if avg < 8 {
            return 16;
        }
        if avg > 238 {
            return 231;
        }
        return (232 + (avg - 8) * 24 / 231) as u8;
    }
    let q = |v: i32| if v < 48 { 0 } else if v < 115 { 1 } else { ((v - 35) / 40) as u8 as i32 };
    (16 + 36 * q(r) + 6 * q(g) + q(b)) as u8
}

fn to_16(c: Rgb, bg: bool) -> u8 {
    const BASE: [(u8, Rgb); 16] = [
        (0, Rgb(0, 0, 0)), (1, Rgb(205, 49, 49)), (2, Rgb(13, 188, 121)), (3, Rgb(229, 229, 16)), (4, Rgb(36, 114, 200)),
        (5, Rgb(188, 63, 188)), (6, Rgb(17, 168, 205)), (7, Rgb(229, 229, 229)), (60, Rgb(102, 102, 102)), (61, Rgb(241, 76, 76)),
        (62, Rgb(35, 209, 139)), (63, Rgb(245, 245, 67)), (64, Rgb(59, 142, 234)), (65, Rgb(214, 112, 214)), (66, Rgb(41, 184, 219)),
        (67, Rgb(255, 255, 255)),
    ];
    let d = |a: Rgb, b: Rgb| {
        let (x, y, z) = (a.0 as i32 - b.0 as i32, a.1 as i32 - b.1 as i32, a.2 as i32 - b.2 as i32);
        x * x + y * y + z * z
    };
    let code = BASE.iter().min_by_key(|(_, rgb)| d(c, *rgb)).map_or(7, |b| b.0);
    code + if bg { 40 } else { 30 }
}

fn color_code(c: Rgb, bg: bool) -> String {
    match depth() {
        Depth::True => format!("{};2;{};{};{}", if bg { 48 } else { 38 }, c.0, c.1, c.2),
        Depth::Ansi256 => format!("{};5;{}", if bg { 48 } else { 38 }, to_256(c)),
        Depth::Ansi16 => to_16(c, bg).to_string(),
        Depth::None => String::new(),
    }
}

pub const RESET: &str = "\x1b[0m";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Style {
    pub fg: Option<Rgb>,
    pub bg: Option<Rgb>,
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underline: bool,
}

impl Style {
    pub const fn new() -> Style {
        Style { fg: None, bg: None, bold: false, dim: false, italic: false, underline: false }
    }
    pub const fn fg(mut self, c: Rgb) -> Style {
        self.fg = Some(c);
        self
    }
    pub const fn bg(mut self, c: Rgb) -> Style {
        self.bg = Some(c);
        self
    }
    pub const fn bold(mut self) -> Style {
        self.bold = true;
        self
    }
    pub const fn dim(mut self) -> Style {
        self.dim = true;
        self
    }
    pub const fn italic(mut self) -> Style {
        self.italic = true;
        self
    }
    pub const fn underline(mut self) -> Style {
        self.underline = true;
        self
    }

    /// The escape sequence that switches to this style (from a reset).
    pub fn prefix(&self) -> String {
        if !enabled() {
            return String::new();
        }
        let mut parts: Vec<String> = Vec::new();
        if self.bold {
            parts.push("1".into());
        }
        if self.dim {
            parts.push("2".into());
        }
        if self.italic {
            parts.push("3".into());
        }
        if self.underline {
            parts.push("4".into());
        }
        if let Some(c) = self.fg {
            parts.push(color_code(c, false));
        }
        if let Some(c) = self.bg {
            parts.push(color_code(c, true));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!("\x1b[{}m", parts.join(";"))
        }
    }

    pub fn paint(&self, text: &str) -> String {
        let p = self.prefix();
        if p.is_empty() || text.is_empty() {
            text.to_string()
        } else {
            format!("{p}{text}{RESET}")
        }
    }
}

/// Shorthands for the common styles.
pub fn fg(c: Rgb, text: &str) -> String {
    Style::new().fg(c).paint(text)
}
pub fn bold(text: &str) -> String {
    Style::new().bold().paint(text)
}
pub fn dim(text: &str) -> String {
    Style::new().fg(pal::DIM).paint(text)
}
pub fn faint(text: &str) -> String {
    Style::new().fg(pal::FAINT).paint(text)
}
pub fn accent(text: &str) -> String {
    Style::new().fg(pal::VIOLET).bold().paint(text)
}

/// Text painted along the aurora gradient, one step per character;
/// `phase` shifts it (for animation).
pub fn gradient(text: &str, phase: f32) -> String {
    gradient_styled(text, phase, false)
}

pub fn gradient_styled(text: &str, phase: f32, bold: bool) -> String {
    if !enabled() {
        return text.to_string();
    }
    let n = text.chars().count().max(2) as f32;
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if ch == ' ' {
            out.push(' ');
            continue;
        }
        let t = ((i as f32 / (n - 1.0)) + phase).rem_euclid(2.0);
        let t = if t > 1.0 { 2.0 - t } else { t };
        let mut s = Style::new().fg(pal::aurora(t));
        s.bold = bold;
        out.push_str(&s.prefix());
        out.push(ch);
    }
    out.push_str(RESET);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_degrade_gracefully() {
        set_depth(Depth::True);
        assert_eq!(Style::new().fg(Rgb(1, 2, 3)).bold().paint("x"), "\x1b[1;38;2;1;2;3mx\x1b[0m");
        set_depth(Depth::Ansi256);
        assert!(Style::new().fg(pal::VIOLET).paint("x").starts_with("\x1b[38;5;"));
        assert_eq!(to_256(Rgb(128, 128, 128)), 244);
        set_depth(Depth::Ansi16);
        assert_eq!(Style::new().fg(Rgb(250, 250, 250)).paint("x"), "\x1b[97mx\x1b[0m");
        set_depth(Depth::None);
        assert_eq!(gradient("hello", 0.0), "hello");
        assert_eq!(Style::new().fg(pal::RED).paint("plain"), "plain");
        assert_eq!(pal::aurora(0.0), pal::CYAN);
        assert_eq!(pal::aurora(1.0), pal::PINK);
    }
}
