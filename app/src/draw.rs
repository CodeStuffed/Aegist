//! Painting the window: a header bar with Aegist's icon and name, an aurora
//! line under it, then the terminal's cells - text through the system's
//! monospace font, and box-drawing and block characters drawn by hand so
//! boxes join up and pictures (half blocks) fill their cells exactly.

use crate::term::{Cell, Color, Term};
use fontdue::{Font, FontSettings, Metrics};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    const fn hex(h: u32) -> Rgb {
        Rgb((h >> 16) as u8, (h >> 8) as u8, h as u8)
    }
    fn u32(self) -> u32 {
        (self.0 as u32) << 16 | (self.1 as u32) << 8 | self.2 as u32
    }
    fn from_u32(v: u32) -> Rgb {
        Rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
    }
    pub fn mix(self, o: Rgb, t: f32) -> Rgb {
        let f = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8;
        Rgb(f(self.0, o.0), f(self.1, o.1), f(self.2, o.2))
    }
}

/// Aegist's colors (the same ones the session uses for its text).
pub mod theme {
    use super::Rgb;
    pub const BG: Rgb = Rgb::hex(0x0E1018);
    pub const HEADER: Rgb = Rgb::hex(0x141726);
    pub const FG: Rgb = Rgb::hex(0xE2E4F0);
    pub const FAINT: Rgb = Rgb::hex(0x646A80);
    pub const CYAN: Rgb = Rgb::hex(0x5EEAD4);
    pub const VIOLET: Rgb = Rgb::hex(0xA78BFA);
    pub const PINK: Rgb = Rgb::hex(0xF472B6);
    pub const SELECTION: Rgb = Rgb::hex(0x34386A);
    pub const ANSI: [Rgb; 16] = [
        Rgb::hex(0x1B1E2B), Rgb::hex(0xF87171), Rgb::hex(0x34D399), Rgb::hex(0xFBBF24),
        Rgb::hex(0x7AA2F7), Rgb::hex(0xA78BFA), Rgb::hex(0x5EEAD4), Rgb::hex(0xC8CBD9),
        Rgb::hex(0x4B5068), Rgb::hex(0xFCA5A5), Rgb::hex(0x6EE7B7), Rgb::hex(0xFDE68A),
        Rgb::hex(0x93C5FD), Rgb::hex(0xC4B5FD), Rgb::hex(0x99F6E4), Rgb::hex(0xF5F6FA),
    ];

    pub fn aurora(t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        if t < 0.5 { CYAN.mix(VIOLET, t * 2.0) } else { VIOLET.mix(PINK, (t - 0.5) * 2.0) }
    }
}

pub fn color(c: Color, default: Rgb) -> Rgb {
    match c {
        Color::Default => default,
        Color::Rgb(r, g, b) => Rgb(r, g, b),
        Color::Idx(i @ 0..=15) => theme::ANSI[i as usize],
        Color::Idx(i @ 16..=231) => {
            let i = i - 16;
            let level = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
            Rgb(level(i / 36), level(i / 6 % 6), level(i % 6))
        }
        Color::Idx(i) => {
            let v = 8 + (i - 232) * 10;
            Rgb(v, v, v)
        }
    }
}

// ---------------------------------------------------------------- fonts

/// Font files to try, best first: (path, index in a collection).
type Candidates = Vec<(String, u32)>;

/// Monospace fonts to try (regular, bold), and fonts for symbols they lack.
fn font_candidates() -> (Candidates, Candidates, Candidates) {
    let s = |p: &str| (p.to_string(), 0u32);
    let win = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
    let local = std::env::var("LOCALAPPDATA").unwrap_or_default();
    let w = |f: &str| (format!("{win}\\Fonts\\{f}"), 0u32);
    let wl = |f: &str| (format!("{local}\\Microsoft\\Windows\\Fonts\\{f}"), 0u32);
    let mut regular = Vec::new();
    let mut bold = Vec::new();
    let mut symbols = Vec::new();
    if let Ok(p) = std::env::var("AEGIST_FONT") {
        regular.push(s(&p));
    }
    if cfg!(windows) {
        regular.extend([w("CascadiaMono.ttf"), wl("CascadiaMono.ttf"), w("CascadiaCode.ttf"), w("consola.ttf"), w("lucon.ttf"), w("cour.ttf")]);
        bold.extend([w("consolab.ttf"), w("courbd.ttf")]);
        symbols.extend([w("seguisym.ttf"), w("CascadiaMono.ttf"), w("segoeui.ttf"), w("msgothic.ttc"), w("arialuni.ttf")]);
    } else if cfg!(target_os = "macos") {
        regular.extend([s("/System/Library/Fonts/SFNSMono.ttf"), s("/System/Library/Fonts/Menlo.ttc"), s("/System/Library/Fonts/Monaco.ttf")]);
        symbols.extend([s("/System/Library/Fonts/Apple Symbols.ttf"), s("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")]);
    } else {
        for dir in ["/usr/share/fonts/truetype", "/usr/share/fonts/TTF", "/usr/share/fonts", "/usr/local/share/fonts"] {
            regular.extend([s(&format!("{dir}/dejavu/DejaVuSansMono.ttf")), s(&format!("{dir}/DejaVuSansMono.ttf")),
                            s(&format!("{dir}/jetbrains-mono/JetBrainsMono-Regular.ttf")), s(&format!("{dir}/liberation/LiberationMono-Regular.ttf")),
                            s(&format!("{dir}/noto/NotoSansMono-Regular.ttf"))]);
            bold.extend([s(&format!("{dir}/dejavu/DejaVuSansMono-Bold.ttf")), s(&format!("{dir}/DejaVuSansMono-Bold.ttf")),
                         s(&format!("{dir}/liberation/LiberationMono-Bold.ttf"))]);
            symbols.extend([s(&format!("{dir}/dejavu/DejaVuSans.ttf")), s(&format!("{dir}/noto/NotoSansSymbols2-Regular.ttf")),
                            s(&format!("{dir}/noto/NotoSansSymbols-Regular.ttf")), s(&format!("{dir}/freefont/FreeSerif.ttf")),
                            s(&format!("{dir}/noto/NotoSansCJK-Regular.ttc"))]);
        }
    }
    (regular, bold, symbols)
}

fn load(path: &str, index: u32) -> Option<Font> {
    let bytes = std::fs::read(path).ok()?;
    Font::from_bytes(bytes, FontSettings { collection_index: index, ..FontSettings::default() }).ok()
}

pub struct Fonts {
    pub regular: Font,
    pub bold: Option<Font>,
    pub fallbacks: Vec<Font>,
}

impl Fonts {
    pub fn load() -> Result<Fonts, String> {
        let (regular, bold, symbols) = font_candidates();
        let r = regular.iter().find_map(|(p, i)| load(p, *i)).ok_or_else(|| {
            format!("no monospace font found (looked for {}); set AEGIST_FONT to a .ttf file", regular.iter().take(3).map(|x| x.0.as_str())
                .collect::<Vec<_>>().join(", "))
        })?;
        let b = bold.iter().find_map(|(p, i)| load(p, *i));
        let mut seen = std::collections::HashSet::new();
        let fallbacks = symbols.iter().filter(|(p, _)| seen.insert(p.clone())).filter_map(|(p, i)| load(p, *i)).take(4).collect();
        Ok(Fonts { regular: r, bold: b, fallbacks })
    }
}

struct Glyph {
    m: Metrics,
    bitmap: Vec<u8>,
    /// No font had it: nothing to draw (a box is drawn instead).
    missing: bool,
}

// ---------------------------------------------------------------- painting

/// What to show besides the terminal itself.
#[derive(Clone, Debug, Default)]
pub struct View {
    /// Lines scrolled back from the bottom.
    pub offset: usize,
    /// Selected text: (line, column) from, to - lines counted from the oldest scrollback line.
    pub selection: Option<((usize, usize), (usize, usize))>,
    pub focused: bool,
    /// The cursor's blink phase.
    pub cursor_on: bool,
    pub hint: String,
}

pub struct Painter {
    fonts: Fonts,
    pub scale: f32,
    pub font_px: f32,
    pub cell_w: usize,
    pub cell_h: usize,
    ascent: f32,
    cache: HashMap<(char, bool, u32), Glyph>,
    icon: Vec<u8>,
    icon_side: u32,
}

pub const PAD: f32 = 12.0;
pub const HEADER: f32 = 38.0;

impl Painter {
    pub fn new(fonts: Fonts, font_pt: f32, scale: f32) -> Painter {
        let mut p = Painter { fonts, scale, font_px: 0.0, cell_w: 8, cell_h: 16, ascent: 12.0, cache: HashMap::new(), icon: Vec::new(), icon_side: 0 };
        p.set_font_size(font_pt);
        p
    }

    pub fn set_font_size(&mut self, pt: f32) {
        self.font_px = (pt * self.scale).round().clamp(8.0, 72.0);
        let m = self.fonts.regular.metrics('M', self.font_px);
        self.cell_w = m.advance_width.round().max(4.0) as usize;
        let lm = self.fonts.regular.horizontal_line_metrics(self.font_px);
        let (ascent, descent) = lm.map_or((self.font_px * 0.8, -self.font_px * 0.2), |l| (l.ascent, l.descent));
        self.cell_h = ((ascent - descent) * 1.12).ceil().max(6.0) as usize;
        self.ascent = ascent + ((self.cell_h as f32 - (ascent - descent)) / 2.0).floor();
        self.cache.clear();
        let side = (20.0 * self.scale).round() as u32;
        if side != self.icon_side {
            self.icon = aegist::icon::rgba(side);
            self.icon_side = side;
        }
    }

    pub fn set_scale(&mut self, scale: f32, pt: f32) {
        self.scale = scale;
        self.set_font_size(pt);
    }

    fn pad(&self) -> usize {
        (PAD * self.scale).round() as usize
    }

    pub fn header(&self) -> usize {
        (HEADER * self.scale).round() as usize
    }

    /// Where the grid starts, in pixels.
    pub fn origin(&self) -> (usize, usize) {
        (self.pad(), self.header() + self.pad() / 2 + (2.0 * self.scale) as usize)
    }

    /// How many cells fit in a window this size.
    pub fn grid_size(&self, w: usize, h: usize) -> (usize, usize) {
        let (ox, oy) = self.origin();
        let cols = w.saturating_sub(ox + self.pad()) / self.cell_w;
        let rows = h.saturating_sub(oy + self.pad() / 2) / self.cell_h;
        (cols.max(2), rows.max(2))
    }

    /// The cell under a point, if any: (column, row on screen).
    pub fn cell_at(&self, x: f64, y: f64, cols: usize, rows: usize) -> Option<(usize, usize)> {
        let (ox, oy) = self.origin();
        if x < ox as f64 || y < oy as f64 {
            return None;
        }
        let c = ((x - ox as f64) / self.cell_w as f64) as usize;
        let r = ((y - oy as f64) / self.cell_h as f64) as usize;
        (r < rows).then_some((c.min(cols - 1), r))
    }

    fn glyph(&mut self, c: char, bold: bool, wide: bool) -> &Glyph {
        let key = (c, bold, if wide { 2 } else { 1 });
        if !self.cache.contains_key(&key) {
            let has = |f: &Font| f.lookup_glyph_index(c) != 0;
            let (font, synthetic) = match (&self.fonts.bold, bold) {
                (Some(b), true) if has(b) => (Some(b), false),
                _ if has(&self.fonts.regular) => (Some(&self.fonts.regular), false),
                _ => (self.fonts.fallbacks.iter().find(|f| has(f)), true),
            };
            let g = match font {
                Some(f) => {
                    let mut px = self.font_px;
                    // a symbol from another font may be wider than its cell: shrink it to fit
                    let room = self.cell_w as f32 * if wide { 2.0 } else { 1.0 } * 1.05;
                    let adv = f.metrics(c, px).advance_width;
                    if synthetic && adv > room {
                        px *= room / adv;
                    }
                    let (m, bitmap) = f.rasterize(c, px);
                    Glyph { m, bitmap, missing: false }
                }
                None => Glyph { m: Metrics::default(), bitmap: Vec::new(), missing: true },
            };
            self.cache.insert(key, g);
        }
        &self.cache[&key]
    }

    /// Paint everything into `buf` (0RGB, `w`×`h`).
    pub fn draw(&mut self, buf: &mut [u32], w: usize, h: usize, term: &Term, view: &View) {
        let mut fb = Frame { buf, w, h };
        fb.fill(0, 0, w, h, theme::BG);
        self.draw_header(&mut fb, term, view);
        let (ox, oy) = self.origin();
        let total = term.total_lines();
        let first = total.saturating_sub(term.rows + view.offset);
        let sel = view.selection.map(|(a, b)| if a <= b { (a, b) } else { (b, a) });
        for row in 0..term.rows {
            let li = first + row;
            let Some(line) = term.line(li) else { continue };
            let y = oy + row * self.cell_h;
            for (col, cell) in line.iter().enumerate().take(term.cols) {
                if cell.spacer {
                    continue;
                }
                let x = ox + col * self.cell_w;
                let selected = sel.is_some_and(|(a, b)| (li, col) >= a && (li, col) <= b);
                let wide = line.get(col + 1).is_some_and(|n| n.spacer);
                self.draw_cell(&mut fb, x, y, cell, selected, wide);
            }
        }
        // the cursor: a violet bar when the window has the keyboard, an outline when not
        if term.cursor_visible && view.offset == 0 && term.cy < term.rows {
            let (x, y) = (ox + term.cx * self.cell_w, oy + term.cy * self.cell_h);
            let t = (2.0 * self.scale).round().max(1.0) as usize;
            if view.focused {
                if view.cursor_on {
                    fb.fill(x, y, t, self.cell_h, theme::VIOLET);
                }
            } else {
                fb.outline(x, y, self.cell_w, self.cell_h, theme::VIOLET.mix(theme::BG, 0.4));
            }
        }
        // where you are in the scrollback
        if view.offset > 0 && total > term.rows {
            let track_h = term.rows * self.cell_h;
            let thumb = (track_h * term.rows / total).max(8);
            let pos = (track_h - thumb) * first / (total - term.rows).max(1);
            let x = w.saturating_sub((5.0 * self.scale) as usize);
            fb.fill(x, oy + pos, (3.0 * self.scale).max(2.0) as usize, thumb, theme::VIOLET.mix(theme::BG, 0.3));
        }
    }

    fn draw_header(&mut self, fb: &mut Frame, term: &Term, view: &View) {
        let hh = self.header();
        for y in 0..hh {
            fb.fill(0, y, fb.w, 1, theme::HEADER.mix(theme::BG, y as f32 / hh as f32 * 0.6));
        }
        // the aurora line
        let line_h = (2.0 * self.scale).round().max(1.0) as usize;
        for x in 0..fb.w {
            fb.fill(x, hh, 1, line_h, theme::aurora(x as f32 / fb.w.max(1) as f32).mix(theme::BG, 0.15));
        }
        let pad = self.pad();
        let side = self.icon_side as usize;
        let iy = (hh.saturating_sub(side)) / 2;
        for y in 0..side {
            for x in 0..side {
                let i = (y * side + x) * 4;
                let a = self.icon[i + 3] as f32 / 255.0;
                if a > 0.0 {
                    fb.blend(pad + x, iy + y, Rgb(self.icon[i], self.icon[i + 1], self.icon[i + 2]), a);
                }
            }
        }
        let ty = (hh.saturating_sub(self.cell_h)) / 2;
        let mut x = pad + side + (10.0 * self.scale) as usize;
        x = self.text(fb, x, ty, "Aegist", theme::FG, true);
        let sub = match &term.title {
            Some(t) if !t.trim().is_empty() && !t.contains("Aegist") => format!("  ·  {t}"),
            _ => "  ·  a coding AI grown from scratch".into(),
        };
        let sub_end = self.text(fb, x, ty, &sub, theme::FAINT, false);
        let hint_w = view.hint.chars().count() * self.cell_w;
        if !view.hint.is_empty() && sub_end + hint_w + 3 * pad < fb.w {
            self.text(fb, fb.w - pad - hint_w, ty, &view.hint, theme::FAINT, false);
        }
    }

    /// Plain text at a pixel position; returns where it ended.
    fn text(&mut self, fb: &mut Frame, mut x: usize, y: usize, s: &str, fg: Rgb, bold: bool) -> usize {
        for c in s.chars() {
            let cell = Cell { ch: c, style: Default::default(), spacer: false };
            self.draw_glyph(fb, x, y, &cell, fg, theme::HEADER, bold, false);
            x += self.cell_w;
        }
        x
    }

    fn draw_cell(&mut self, fb: &mut Frame, x: usize, y: usize, cell: &Cell, selected: bool, wide: bool) {
        let s = cell.style;
        let (mut fg, mut bg) = (color(s.fg, theme::FG), color(s.bg, theme::BG));
        if s.bold {
            if let Color::Idx(i @ 0..=7) = s.fg {
                fg = theme::ANSI[i as usize + 8];
            }
        }
        if s.inverse {
            std::mem::swap(&mut fg, &mut bg);
        }
        if s.dim {
            fg = fg.mix(bg, 0.45);
        }
        if selected {
            bg = theme::SELECTION;
        }
        let cw = self.cell_w * if wide { 2 } else { 1 };
        if bg != theme::BG {
            fb.fill(x, y, cw, self.cell_h, bg);
        }
        if cell.ch != ' ' && !draw_special(fb, cell.ch, x, y, self.cell_w, self.cell_h, fg, bg, self.scale) {
            self.draw_glyph(fb, x, y, cell, fg, bg, s.bold, wide);
        }
        if s.underline {
            let t = self.scale.round().max(1.0) as usize;
            fb.fill(x, y + self.cell_h - 2 * t, cw, t, fg);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_glyph(&mut self, fb: &mut Frame, x: usize, y: usize, cell: &Cell, fg: Rgb, _bg: Rgb, bold: bool, wide: bool) {
        let ascent = self.ascent;
        let (cell_w, cell_h) = (self.cell_w, self.cell_h);
        let fake_bold = bold && self.fonts.bold.is_none();
        let g = self.glyph(cell.ch, bold, wide);
        if g.missing {
            // no font has it: a small hollow box, like other terminals
            fb.outline(x + cell_w / 5, y + cell_h / 4, cell_w * 3 / 5, cell_h / 2, fg.mix(theme::BG, 0.5));
            return;
        }
        let gx = x as i64 + g.m.xmin as i64 + if wide { 0 } else { ((cell_w as f32 - g.m.advance_width) / 2.0).max(0.0) as i64 };
        let gy = y as i64 + ascent.round() as i64 - g.m.height as i64 - g.m.ymin as i64;
        for passes in 0..if fake_bold { 2 } else { 1 } {
            for row in 0..g.m.height {
                for col in 0..g.m.width {
                    let a = g.bitmap[row * g.m.width + col];
                    if a == 0 {
                        continue;
                    }
                    let (px, py) = (gx + col as i64 + passes, gy + row as i64);
                    if px >= 0 && py >= 0 {
                        fb.blend(px as usize, py as usize, fg, a as f32 / 255.0);
                    }
                }
            }
        }
    }
}

pub struct Frame<'a> {
    pub buf: &'a mut [u32],
    pub w: usize,
    pub h: usize,
}

impl Frame<'_> {
    pub fn fill(&mut self, x: usize, y: usize, w: usize, h: usize, c: Rgb) {
        let v = c.u32();
        for yy in y..(y + h).min(self.h) {
            let row = yy * self.w;
            for xx in x..(x + w).min(self.w) {
                self.buf[row + xx] = v;
            }
        }
    }

    pub fn blend(&mut self, x: usize, y: usize, c: Rgb, a: f32) {
        if x >= self.w || y >= self.h {
            return;
        }
        let i = y * self.w + x;
        let under = Rgb::from_u32(self.buf[i]);
        self.buf[i] = under.mix(c, a.clamp(0.0, 1.0)).u32();
    }

    fn fill_alpha(&mut self, x: usize, y: usize, w: usize, h: usize, c: Rgb, a: f32) {
        for yy in y..(y + h).min(self.h) {
            for xx in x..(x + w).min(self.w) {
                self.blend(xx, yy, c, a);
            }
        }
    }

    pub fn outline(&mut self, x: usize, y: usize, w: usize, h: usize, c: Rgb) {
        self.fill(x, y, w, 1, c);
        self.fill(x, y + h.saturating_sub(1), w, 1, c);
        self.fill(x, y, 1, h, c);
        self.fill(x + w.saturating_sub(1), y, 1, h, c);
    }
}

/// Box-drawing and block characters, drawn to fill the cell exactly.
/// Returns false for anything else (the font draws it).
#[allow(clippy::too_many_arguments)]
pub fn draw_special(fb: &mut Frame, c: char, x: usize, y: usize, w: usize, h: usize, fg: Rgb, bg: Rgb, scale: f32) -> bool {
    let code = c as u32;
    if (0x2580..=0x259F).contains(&code) {
        let (hw, hh) = (w / 2, h / 2);
        let eighth_h = |n: usize| (h * n + 4) / 8;
        let eighth_w = |n: usize| (w * n + 4) / 8;
        match c {
            '▀' => fb.fill(x, y, w, hh, fg),
            '▄' => fb.fill(x, y + hh, w, h - hh, fg),
            '█' => fb.fill(x, y, w, h, fg),
            '▌' => fb.fill(x, y, hw, h, fg),
            '▐' => fb.fill(x + hw, y, w - hw, h, fg),
            '▔' => fb.fill(x, y, w, eighth_h(1), fg),
            '▕' => fb.fill(x + w - eighth_w(1), y, eighth_w(1), h, fg),
            '▁' | '▂' | '▃' | '▅' | '▆' | '▇' => {
                let n = match c { '▁' => 1, '▂' => 2, '▃' => 3, '▅' => 5, '▆' => 6, _ => 7 };
                let t = eighth_h(n);
                fb.fill(x, y + h - t, w, t, fg);
            }
            '▉' | '▊' | '▋' | '▍' | '▎' | '▏' => {
                let n = match c { '▉' => 7, '▊' => 6, '▋' => 5, '▍' => 3, '▎' => 2, _ => 1 };
                fb.fill(x, y, eighth_w(n), h, fg);
            }
            '░' => fb.fill_alpha(x, y, w, h, fg, 0.25),
            '▒' => fb.fill_alpha(x, y, w, h, fg, 0.5),
            '▓' => fb.fill_alpha(x, y, w, h, fg, 0.75),
            _ => {
                // quadrants: bits for upper-left, upper-right, lower-left, lower-right
                let q = match c {
                    '▖' => 0b0010, '▗' => 0b0001, '▘' => 0b1000, '▝' => 0b0100, '▙' => 0b1011, '▚' => 0b1001,
                    '▛' => 0b1110, '▜' => 0b1101, '▞' => 0b0110, '▟' => 0b0111, _ => return false,
                };
                if q & 0b1000 != 0 { fb.fill(x, y, hw, hh, fg) }
                if q & 0b0100 != 0 { fb.fill(x + hw, y, w - hw, hh, fg) }
                if q & 0b0010 != 0 { fb.fill(x, y + hh, hw, h - hh, fg) }
                if q & 0b0001 != 0 { fb.fill(x + hw, y + hh, w - hw, h - hh, fg) }
            }
        }
        return true;
    }
    if !(0x2500..=0x257F).contains(&code) {
        return false;
    }
    let light = (scale.round() as usize).max(1).max(w / 9);
    let heavy = light * 2;
    let (cx, cy) = (x + w / 2, y + h / 2);
    // one arm from the middle to an edge, `t` thick
    let arm = |fb: &mut Frame, dir: u8, t: usize, off: isize| {
        let (ox, oy) = ((cx as isize + off) as usize, (cy as isize + off) as usize);
        match dir {
            0 => fb.fill(ox - t / 2, y, t, cy - y + t - t / 2, fg),     // up
            1 => fb.fill(ox - t / 2, oy - t / 2, t, y + h - oy + t / 2, fg), // down
            2 => fb.fill(x, oy - t / 2, cx - x + t - t / 2, t, fg),     // left
            _ => fb.fill(ox - t / 2, oy - t / 2, x + w - ox + t / 2, t, fg), // right
        }
    };
    // (up, down, left, right): 0 none, 1 light, 2 heavy
    let arms: Option<[u8; 4]> = match c {
        '─' => Some([0, 0, 1, 1]), '━' => Some([0, 0, 2, 2]), '│' => Some([1, 1, 0, 0]), '┃' => Some([2, 2, 0, 0]),
        '┌' => Some([0, 1, 0, 1]), '┐' => Some([0, 1, 1, 0]), '└' => Some([1, 0, 0, 1]), '┘' => Some([1, 0, 1, 0]),
        '┏' => Some([0, 2, 0, 2]), '┓' => Some([0, 2, 2, 0]), '┗' => Some([2, 0, 0, 2]), '┛' => Some([2, 0, 2, 0]),
        '├' => Some([1, 1, 0, 1]), '┤' => Some([1, 1, 1, 0]), '┬' => Some([0, 1, 1, 1]), '┴' => Some([1, 0, 1, 1]), '┼' => Some([1, 1, 1, 1]),
        '┣' => Some([2, 2, 0, 2]), '┫' => Some([2, 2, 2, 0]), '┳' => Some([0, 2, 2, 2]), '┻' => Some([2, 0, 2, 2]), '╋' => Some([2, 2, 2, 2]),
        '╴' => Some([0, 0, 1, 0]), '╵' => Some([1, 0, 0, 0]), '╶' => Some([0, 0, 0, 1]), '╷' => Some([0, 1, 0, 0]),
        _ => None,
    };
    if let Some(a) = arms {
        for (dir, &weight) in a.iter().enumerate() {
            if weight > 0 {
                arm(fb, dir as u8, if weight == 2 { heavy } else { light }, 0);
            }
        }
        return true;
    }
    // double lines: two light lines `gap` apart
    let gap = (light + 1).max(w / 6) as isize;
    let dbl = |fb: &mut Frame, horizontal: bool, from: usize, to: usize, at: isize| {
        let t = light;
        if horizontal {
            let yy = (cy as isize + at) as usize;
            fb.fill(from, yy - t / 2, to.saturating_sub(from), t, fg);
        } else {
            let xx = (cx as isize + at) as usize;
            fb.fill(xx - t / 2, from, t, to.saturating_sub(from), fg);
        }
    };
    let (l, r, t, b) = (x, x + w, y, y + h);
    let p = |off: isize, c: usize| (c as isize + off) as usize;
    match c {
        '═' => {
            dbl(fb, true, l, r, -gap);
            dbl(fb, true, l, r, gap);
        }
        '║' => {
            dbl(fb, false, t, b, -gap);
            dbl(fb, false, t, b, gap);
        }
        // corners: an outer and an inner line, each turning the corner
        '╔' => {
            dbl(fb, true, p(-gap, cx), r, -gap);
            dbl(fb, false, p(-gap, cy), b, -gap);
            dbl(fb, true, p(gap, cx), r, gap);
            dbl(fb, false, p(gap, cy), b, gap);
        }
        '╗' => {
            dbl(fb, true, l, p(gap, cx) + light, -gap);
            dbl(fb, false, p(-gap, cy), b, gap);
            dbl(fb, true, l, p(-gap, cx) + light, gap);
            dbl(fb, false, p(gap, cy), b, -gap);
        }
        '╚' => {
            dbl(fb, true, p(-gap, cx), r, gap);
            dbl(fb, false, t, p(gap, cy) + light, -gap);
            dbl(fb, true, p(gap, cx), r, -gap);
            dbl(fb, false, t, p(-gap, cy) + light, gap);
        }
        '╝' => {
            dbl(fb, true, l, p(gap, cx) + light, gap);
            dbl(fb, false, t, p(gap, cy) + light, gap);
            dbl(fb, true, l, p(-gap, cx) + light, -gap);
            dbl(fb, false, t, p(-gap, cy) + light, -gap);
        }
        '╭' | '╮' | '╰' | '╯' => {
            // a quarter circle joining the two arms
            let rad = (w as f32 / 2.0).min(h as f32 / 2.0);
            let (right, down) = match c { '╭' => (true, true), '╮' => (false, true), '╰' => (true, false), _ => (false, false) };
            // centered on the middle of the line pixels, so the arc meets the straight lines
            let ccx = cx as f32 + 0.5 + if right { rad } else { -rad };
            let ccy = cy as f32 + 0.5 + if down { rad } else { -rad };
            let half = light as f32 / 2.0;
            for yy in y..y + h {
                for xx in x..x + w {
                    let (fx, fy) = (xx as f32 + 0.5, yy as f32 + 0.5);
                    let in_quadrant = (if right { fx <= ccx } else { fx >= ccx }) && (if down { fy <= ccy } else { fy >= ccy });
                    if !in_quadrant {
                        continue;
                    }
                    let d = ((fx - ccx).powi(2) + (fy - ccy).powi(2)).sqrt();
                    let cover = (half + 0.5 - (d - rad).abs()).clamp(0.0, 1.0);
                    if cover > 0.0 {
                        fb.blend(xx, yy, fg, cover);
                    }
                }
            }
            // the straight part down (or up) to the edge
            let t2 = light;
            if down {
                let from = (ccy as usize).min(b);
                fb.fill(cx - t2 / 2, from, t2, b - from, fg);
            } else {
                let to = (ccy as usize).max(t);
                fb.fill(cx - t2 / 2, t, t2, to - t, fg);
            }
        }
        _ => {
            let _ = bg;
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::Emulator;

    #[test]
    fn palette_colors() {
        assert_eq!(color(Color::Idx(16), theme::FG), Rgb(0, 0, 0));
        assert_eq!(color(Color::Idx(231), theme::FG), Rgb(255, 255, 255));
        assert_eq!(color(Color::Idx(196), theme::FG), Rgb(255, 0, 0));
        assert_eq!(color(Color::Idx(232), theme::FG), Rgb(8, 8, 8));
        assert_eq!(color(Color::Default, theme::BG), theme::BG);
    }

    #[test]
    fn half_blocks_fill_exactly_and_boxes_join() {
        let mut buf = vec![0u32; 20 * 20];
        let mut fb = Frame { buf: &mut buf, w: 20, h: 20 };
        assert!(draw_special(&mut fb, '▀', 0, 0, 10, 20, Rgb(255, 0, 0), Rgb(0, 0, 0), 1.0));
        assert_eq!(fb.buf[0], 0xFF0000);
        assert_eq!(fb.buf[9 * 20 + 9], 0xFF0000);
        assert_eq!(fb.buf[10 * 20], 0, "the bottom half is left to the background");
        // a horizontal line reaches both edges of its cell, at the same height in every cell
        let mut buf = vec![0u32; 20 * 20];
        let mut fb = Frame { buf: &mut buf, w: 20, h: 20 };
        draw_special(&mut fb, '─', 0, 0, 10, 20, Rgb(0, 255, 0), Rgb(0, 0, 0), 1.0);
        draw_special(&mut fb, '╮', 10, 0, 10, 20, Rgb(0, 255, 0), Rgb(0, 0, 0), 1.0);
        assert!(fb.buf[10 * 20] != 0 && fb.buf[10 * 20 + 9] != 0 && fb.buf[10 * 20 + 11] != 0);
        assert!(fb.buf[19 * 20 + 15] != 0, "the corner runs down to the bottom edge");
        assert!(!draw_special(&mut fb, 'a', 0, 0, 10, 20, Rgb(0, 0, 0), Rgb(0, 0, 0), 1.0));
    }

    #[test]
    fn a_whole_frame_paints() {
        let Ok(fonts) = Fonts::load() else { return }; // no fonts on this machine: nothing to check
        let mut p = Painter::new(fonts, 13.0, 1.0);
        let (w, h) = (640, 400);
        let (cols, rows) = p.grid_size(w, h);
        let mut e = Emulator::new(cols, rows);
        e.feed("\x1b[38;2;167;139;250m╭─ ✦ plan ─╮\x1b[0m\r\n│ hello 日本 │\r\n\x1b[1mbold\x1b[0m".as_bytes());
        let mut buf = vec![0u32; w * h];
        p.draw(&mut buf, w, h, &e.term, &View { focused: true, cursor_on: true, ..View::default() });
        let (ox, oy) = p.origin();
        let violet = theme::VIOLET.u32();
        let row0: Vec<u32> = (0..cols * p.cell_w).map(|x| buf[(oy + p.cell_h / 2) * w + ox + x]).collect();
        assert!(row0.iter().filter(|&&v| v == violet).count() >= p.cell_w * 2, "the box's top line is drawn");
        assert!(buf.iter().any(|&v| v != theme::BG.u32() && v != violet), "text and the header are drawn");
        assert_eq!(p.cell_at(ox as f64 + 1.0, oy as f64 + 1.0, cols, rows), Some((0, 0)));
    }
}
