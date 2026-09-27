//! The terminal inside the Aegist window: reads the escape codes the
//! session writes (colors, cursor moves, clearing) and keeps the grid of
//! cells to draw, plus the lines that scrolled off the top.
//!
//! It understands what a modern console sends - including Windows'
//! pseudo-console, which re-renders everything as plain VT sequences - and
//! answers the few questions programs ask a terminal (where's the cursor,
//! what are you).

use std::collections::VecDeque;
use unicode_width::UnicodeWidthChar;
use vte::{Params, Parser, Perform};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Color {
    #[default]
    Default,
    /// The 256-color palette (0-15 are the 16 basic colors).
    Idx(u8),
    Rgb(u8, u8, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underline: bool,
    pub inverse: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub style: Style,
    /// The right half of a wide character (nothing to draw).
    pub spacer: bool,
}

impl Cell {
    fn blank(style: Style) -> Cell {
        Cell { ch: ' ', style: Style { bg: style.bg, ..Style::default() }, spacer: false }
    }
}

pub type Line = Vec<Cell>;

pub struct Term {
    pub cols: usize,
    pub rows: usize,
    pub screen: Vec<Line>,
    pub scrollback: VecDeque<Line>,
    pub max_scrollback: usize,
    /// The main screen, kept while the alternate screen is up.
    saved_screen: Option<(Vec<Line>, usize, usize)>,
    pub cx: usize,
    pub cy: usize,
    /// The last column was written: the next character wraps first.
    pending_wrap: bool,
    pub style: Style,
    saved_cursor: (usize, usize, Style),
    top: usize,
    bottom: usize,
    pub cursor_visible: bool,
    pub bracketed_paste: bool,
    pub app_cursor: bool,
    autowrap: bool,
    pub title: Option<String>,
    /// Answers to send back to the program.
    pub replies: Vec<u8>,
    /// Something changed since the last draw.
    pub dirty: bool,
}

impl Term {
    pub fn new(cols: usize, rows: usize) -> Term {
        let (cols, rows) = (cols.max(2), rows.max(2));
        Term { cols, rows, screen: vec![vec![Cell::blank(Style::default()); cols]; rows], scrollback: VecDeque::new(), max_scrollback: 10_000,
               saved_screen: None, cx: 0, cy: 0, pending_wrap: false, style: Style::default(), saved_cursor: (0, 0, Style::default()), top: 0,
               bottom: rows - 1, cursor_visible: true, bracketed_paste: false, app_cursor: false, autowrap: true, title: None,
               replies: Vec::new(), dirty: true }
    }

    pub fn in_alt_screen(&self) -> bool {
        self.saved_screen.is_some()
    }

    /// New size: lines are cut or padded; shrinking pushes the top lines into
    /// the scrollback so the cursor's line stays in view.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        let (cols, rows) = (cols.max(2), rows.max(2));
        if cols == self.cols && rows == self.rows {
            return;
        }
        for line in self.screen.iter_mut() {
            line.resize(cols, Cell::blank(Style::default()));
            if let Some(last) = line.last_mut() {
                if last.spacer {
                    *last = Cell::blank(Style::default());
                }
            }
        }
        if rows < self.rows {
            let excess = self.rows - rows;
            let from_top = excess.min(self.cy);
            for _ in 0..from_top {
                let line = self.screen.remove(0);
                if !self.in_alt_screen() {
                    self.push_scrollback(line);
                }
            }
            self.screen.truncate(rows);
            self.cy -= from_top;
        } else {
            self.screen.resize(rows, vec![Cell::blank(Style::default()); cols]);
        }
        if let Some((saved, _, _)) = self.saved_screen.as_mut() {
            saved.resize(rows, vec![Cell::blank(Style::default()); cols]);
            for l in saved.iter_mut() {
                l.resize(cols, Cell::blank(Style::default()));
            }
        }
        self.cols = cols;
        self.rows = rows;
        self.cx = self.cx.min(cols - 1);
        self.cy = self.cy.min(rows - 1);
        self.top = 0;
        self.bottom = rows - 1;
        self.pending_wrap = false;
        self.dirty = true;
    }

    fn push_scrollback(&mut self, line: Line) {
        self.scrollback.push_back(line);
        while self.scrollback.len() > self.max_scrollback {
            self.scrollback.pop_front();
        }
    }

    fn blank(&self) -> Cell {
        Cell::blank(self.style)
    }

    fn scroll_up(&mut self, n: usize) {
        for _ in 0..n {
            let line = self.screen.remove(self.top);
            if self.top == 0 && !self.in_alt_screen() {
                self.push_scrollback(line);
            }
            self.screen.insert(self.bottom, vec![self.blank(); self.cols]);
        }
    }

    fn scroll_down(&mut self, n: usize) {
        for _ in 0..n {
            self.screen.remove(self.bottom);
            self.screen.insert(self.top, vec![self.blank(); self.cols]);
        }
    }

    fn linefeed(&mut self) {
        self.pending_wrap = false;
        if self.cy == self.bottom {
            self.scroll_up(1);
        } else if self.cy + 1 < self.rows {
            self.cy += 1;
        }
    }

    fn put(&mut self, c: char) {
        let w = c.width().unwrap_or(0);
        if w == 0 {
            return; // combining marks and the like: drawn as nothing
        }
        if self.pending_wrap && self.autowrap {
            self.cx = 0;
            self.linefeed();
        }
        if w == 2 && self.cx + 1 >= self.cols {
            if self.autowrap {
                self.screen[self.cy][self.cx] = self.blank();
                self.cx = 0;
                self.linefeed();
            } else {
                return;
            }
        }
        // writing over half of a wide character clears the other half
        if self.screen[self.cy][self.cx].spacer && self.cx > 0 {
            self.screen[self.cy][self.cx - 1] = self.blank();
        }
        self.screen[self.cy][self.cx] = Cell { ch: c, style: self.style, spacer: false };
        if w == 2 {
            self.screen[self.cy][self.cx + 1] = Cell { ch: ' ', style: self.style, spacer: true };
        }
        if self.cx + w >= self.cols {
            self.cx = self.cols - 1;
            self.pending_wrap = true;
        } else {
            self.cx += w;
        }
    }

    fn erase(&mut self, y: usize, from: usize, to: usize) {
        let b = self.blank();
        for x in from.min(self.cols)..to.min(self.cols) {
            self.screen[y][x] = b;
        }
    }

    fn goto(&mut self, x: usize, y: usize) {
        self.cx = x.min(self.cols - 1);
        self.cy = y.min(self.rows - 1);
        self.pending_wrap = false;
    }

    fn sgr(&mut self, params: &[u16]) {
        if params.is_empty() {
            self.style = Style::default();
            return;
        }
        let mut i = 0;
        while i < params.len() {
            let p = params[i];
            match p {
                0 => self.style = Style::default(),
                1 => self.style.bold = true,
                2 => self.style.dim = true,
                3 => self.style.italic = true,
                4 => self.style.underline = true,
                7 => self.style.inverse = true,
                21 | 24 => self.style.underline = false,
                22 => {
                    self.style.bold = false;
                    self.style.dim = false;
                }
                23 => self.style.italic = false,
                27 => self.style.inverse = false,
                30..=37 => self.style.fg = Color::Idx((p - 30) as u8),
                39 => self.style.fg = Color::Default,
                40..=47 => self.style.bg = Color::Idx((p - 40) as u8),
                49 => self.style.bg = Color::Default,
                90..=97 => self.style.fg = Color::Idx((p - 90 + 8) as u8),
                100..=107 => self.style.bg = Color::Idx((p - 100 + 8) as u8),
                38 | 48 => {
                    let color = match params.get(i + 1) {
                        Some(5) => {
                            let c = params.get(i + 2).map(|&n| Color::Idx(n.min(255) as u8));
                            i += 2;
                            c
                        }
                        Some(2) => {
                            let get = |k: usize| params.get(i + k).map_or(0, |&n| n.min(255) as u8);
                            let c = Some(Color::Rgb(get(2), get(3), get(4)));
                            i += 4;
                            c
                        }
                        _ => None,
                    };
                    if let Some(c) = color {
                        if p == 38 {
                            self.style.fg = c;
                        } else {
                            self.style.bg = c;
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn set_mode(&mut self, private: bool, params: &[u16], on: bool) {
        for &p in params {
            match (private, p) {
                (true, 1) => self.app_cursor = on,
                (true, 7) => self.autowrap = on,
                (true, 25) => self.cursor_visible = on,
                (true, 2004) => self.bracketed_paste = on,
                (true, 47 | 1047 | 1049) => {
                    if on && self.saved_screen.is_none() {
                        let main = std::mem::replace(&mut self.screen, vec![vec![Cell::blank(Style::default()); self.cols]; self.rows]);
                        self.saved_screen = Some((main, self.cx, self.cy));
                    } else if !on {
                        if let Some((main, x, y)) = self.saved_screen.take() {
                            self.screen = main;
                            self.goto(x, y);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Every line, scrollback first: (the lines, how many are scrollback).
    pub fn total_lines(&self) -> usize {
        self.scrollback.len() + self.rows
    }

    /// Line `i` counting from the oldest scrollback line.
    pub fn line(&self, i: usize) -> Option<&Line> {
        if i < self.scrollback.len() {
            self.scrollback.get(i)
        } else {
            self.screen.get(i - self.scrollback.len())
        }
    }

    /// The text between two points (line, column), inclusive, one line per line.
    pub fn text_between(&self, a: (usize, usize), b: (usize, usize)) -> String {
        let (start, end) = if a <= b { (a, b) } else { (b, a) };
        let mut out = String::new();
        for y in start.0..=end.0 {
            let Some(line) = self.line(y) else { break };
            let from = if y == start.0 { start.1 } else { 0 };
            let to = if y == end.0 { (end.1 + 1).min(line.len()) } else { line.len() };
            let s: String = line[from.min(line.len())..to].iter().filter(|c| !c.spacer).map(|c| c.ch).collect();
            out.push_str(s.trim_end());
            if y != end.0 {
                out.push('\n');
            }
        }
        out
    }

    /// A line's text (for tests).
    #[cfg(test)]
    pub fn row_text(&self, y: usize) -> String {
        self.screen[y].iter().filter(|c| !c.spacer).map(|c| c.ch).collect::<String>().trim_end().to_string()
    }
}

fn flat(params: &Params) -> Vec<u16> {
    params.iter().flat_map(|p| p.iter().copied()).collect()
}

fn first(params: &[u16], default: usize) -> usize {
    match params.first() {
        Some(&0) | None => default,
        Some(&n) => n as usize,
    }
}

impl Perform for Term {
    fn print(&mut self, c: char) {
        self.put(c);
        self.dirty = true;
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\n' | 0x0b | 0x0c => self.linefeed(),
            b'\r' => {
                self.cx = 0;
                self.pending_wrap = false;
            }
            0x08 => {
                self.cx = self.cx.saturating_sub(1);
                self.pending_wrap = false;
            }
            b'\t' => {
                self.cx = ((self.cx / 8 + 1) * 8).min(self.cols - 1);
            }
            _ => {}
        }
        self.dirty = true;
    }

    fn osc_dispatch(&mut self, params: &[&[u8]], _bell_terminated: bool) {
        if let [kind, title, ..] = params {
            if matches!(*kind, b"0" | b"2") {
                self.title = Some(String::from_utf8_lossy(title).into_owned());
            }
        }
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], _ignore: bool, action: char) {
        let p = flat(params);
        let private = intermediates.first() == Some(&b'?');
        let n = first(&p, 1);
        match (action, intermediates) {
            ('h', _) => self.set_mode(private, &p, true),
            ('l', _) => self.set_mode(private, &p, false),
            ('m', []) => self.sgr(&p),
            ('A', []) => self.goto(self.cx, self.cy.saturating_sub(n)),
            ('B' | 'e', []) => self.goto(self.cx, self.cy + n),
            ('C' | 'a', []) => self.goto(self.cx + n, self.cy),
            ('D', []) => self.goto(self.cx.saturating_sub(n), self.cy),
            ('E', []) => self.goto(0, self.cy + n),
            ('F', []) => self.goto(0, self.cy.saturating_sub(n)),
            ('G' | '`', []) => self.goto(n - 1, self.cy),
            ('d', []) => self.goto(self.cx, n - 1),
            ('H' | 'f', []) => {
                let row = first(&p, 1);
                let col = p.get(1).map_or(1, |&c| (c as usize).max(1));
                self.goto(col - 1, row - 1);
            }
            ('J', []) => {
                let (cx, cy, rows) = (self.cx, self.cy, self.rows);
                match p.first().copied().unwrap_or(0) {
                    0 => {
                        self.erase(cy, cx, self.cols);
                        for y in cy + 1..rows {
                            self.erase(y, 0, self.cols);
                        }
                    }
                    1 => {
                        for y in 0..cy {
                            self.erase(y, 0, self.cols);
                        }
                        self.erase(cy, 0, cx + 1);
                    }
                    2 => (0..rows).for_each(|y| self.erase(y, 0, self.cols)),
                    3 => self.scrollback.clear(),
                    _ => {}
                }
            }
            ('K', []) => {
                let (cx, cy) = (self.cx, self.cy);
                match p.first().copied().unwrap_or(0) {
                    0 => self.erase(cy, cx, self.cols),
                    1 => self.erase(cy, 0, cx + 1),
                    2 => self.erase(cy, 0, self.cols),
                    _ => {}
                }
            }
            ('X', []) => {
                let (cx, cy) = (self.cx, self.cy);
                self.erase(cy, cx, cx + n);
            }
            ('P', []) => {
                let (cx, cy, b) = (self.cx, self.cy, self.blank());
                let line = &mut self.screen[cy];
                for _ in 0..n.min(self.cols - cx) {
                    line.remove(cx);
                    line.push(b);
                }
            }
            ('@', []) => {
                let (cx, cy, b) = (self.cx, self.cy, self.blank());
                let line = &mut self.screen[cy];
                for _ in 0..n.min(self.cols - cx) {
                    line.insert(cx, b);
                    line.pop();
                }
            }
            ('L', []) if (self.top..=self.bottom).contains(&self.cy) => {
                let saved = self.top;
                self.top = self.cy;
                self.scroll_down(n.min(self.bottom - self.cy + 1));
                self.top = saved;
            }
            ('M', []) if (self.top..=self.bottom).contains(&self.cy) => {
                let saved = self.top;
                self.top = self.cy;
                // lines deleted inside the screen don't go to the scrollback
                for _ in 0..n.min(self.bottom - self.cy + 1) {
                    self.screen.remove(self.top);
                    self.screen.insert(self.bottom, vec![self.blank(); self.cols]);
                }
                self.top = saved;
            }
            ('S', []) => self.scroll_up(n),
            ('T', []) => self.scroll_down(n),
            ('r', []) => {
                let top = first(&p, 1) - 1;
                let bottom = p.get(1).map_or(self.rows, |&b| if b == 0 { self.rows } else { b as usize }).min(self.rows) - 1;
                if top < bottom {
                    self.top = top;
                    self.bottom = bottom;
                    self.goto(0, 0);
                }
            }
            ('s', []) => self.saved_cursor = (self.cx, self.cy, self.style),
            ('u', []) => {
                let (x, y, s) = self.saved_cursor;
                self.style = s;
                self.goto(x, y);
            }
            ('n', []) if p.first() == Some(&6) => {
                self.replies.extend_from_slice(format!("\x1b[{};{}R", self.cy + 1, self.cx + 1).as_bytes());
            }
            ('n', []) if p.first() == Some(&5) => self.replies.extend_from_slice(b"\x1b[0n"),
            // "what are you": a VT220-class terminal with color
            ('c', []) => self.replies.extend_from_slice(b"\x1b[?62;22c"),
            ('c', [b'>']) => self.replies.extend_from_slice(b"\x1b[>0;10;1c"),
            _ => {}
        }
        self.dirty = true;
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], _ignore: bool, byte: u8) {
        match (intermediates, byte) {
            ([], b'7') => self.saved_cursor = (self.cx, self.cy, self.style),
            ([], b'8') => {
                let (x, y, s) = self.saved_cursor;
                self.style = s;
                self.goto(x, y);
            }
            ([], b'D') => self.linefeed(),
            ([], b'E') => {
                self.cx = 0;
                self.linefeed();
            }
            ([], b'M') => {
                if self.cy == self.top {
                    self.scroll_down(1);
                } else {
                    self.cy = self.cy.saturating_sub(1);
                }
            }
            ([], b'c') => {
                let (c, r) = (self.cols, self.rows);
                *self = Term::new(c, r);
            }
            _ => {}
        }
        self.dirty = true;
    }
}

/// A terminal plus the parser that feeds it.
pub struct Emulator {
    parser: Parser,
    pub term: Term,
}

impl Emulator {
    pub fn new(cols: usize, rows: usize) -> Emulator {
        Emulator { parser: Parser::new(), term: Term::new(cols, rows) }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        self.parser.advance(&mut self.term, bytes);
    }

    /// Answers owed to the program since last asked.
    pub fn take_replies(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.term.replies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emu(cols: usize, rows: usize, input: &str) -> Emulator {
        let mut e = Emulator::new(cols, rows);
        e.feed(input.as_bytes());
        e
    }

    #[test]
    fn text_wraps_scrolls_and_keeps_history() {
        let e = emu(10, 3, "hello\r\nworld\r\nthird\r\nfourth");
        assert_eq!(e.term.row_text(0), "world");
        assert_eq!(e.term.row_text(2), "fourth");
        assert_eq!(e.term.scrollback.len(), 1);
        let e = emu(5, 3, "abcdefgh");
        assert_eq!((e.term.row_text(0).as_str(), e.term.row_text(1).as_str()), ("abcde", "fgh"));
        // the last column doesn't wrap until the next character comes
        let e = emu(5, 3, "abcde\r\nx");
        assert_eq!((e.term.row_text(0).as_str(), e.term.row_text(1).as_str()), ("abcde", "x"));
    }

    #[test]
    fn colors_and_styles() {
        let e = emu(20, 2, "\x1b[1;38;2;167;139;250mhi\x1b[0m \x1b[48;5;236;33mx\x1b[m");
        let c = &e.term.screen[0];
        assert_eq!(c[0].style.fg, Color::Rgb(167, 139, 250));
        assert!(c[0].style.bold);
        assert_eq!(c[2].style, Style::default());
        assert_eq!((c[3].style.bg, c[3].style.fg), (Color::Idx(236), Color::Idx(3)));
        let e = emu(20, 2, "\x1b[38:2:1:2:3mz");
        assert_eq!(e.term.screen[0][0].style.fg, Color::Rgb(1, 2, 3));
    }

    #[test]
    fn cursor_moves_and_erasing() {
        let e = emu(10, 4, "0123456789\x1b[2;3Hab\x1b[1;1H\x1b[K\x1b[4;1Hlast\x1b[2D\x1b[0K");
        assert_eq!(e.term.row_text(0), "");
        assert_eq!(e.term.row_text(1), "  ab");
        assert_eq!(e.term.row_text(3), "la");
        let e = emu(10, 3, "aaa\r\nbbb\r\nccc\x1b[2J\x1b[Hz");
        assert_eq!((e.term.row_text(0).as_str(), e.term.row_text(1).as_str()), ("z", ""));
        let e = emu(10, 3, "x\x1b[3J");
        assert!(e.term.scrollback.is_empty());
    }

    #[test]
    fn wide_characters_take_two_cells() {
        let e = emu(6, 2, "a日b");
        assert_eq!(e.term.screen[0][1].ch, '日');
        assert!(e.term.screen[0][2].spacer);
        assert_eq!(e.term.screen[0][3].ch, 'b');
        assert_eq!(e.term.row_text(0), "a日b");
        let e = emu(4, 2, "abc日");
        assert_eq!((e.term.row_text(0).as_str(), e.term.row_text(1).as_str()), ("abc", "日"));
    }

    #[test]
    fn it_answers_what_programs_ask() {
        let mut e = emu(10, 5, "\x1b[3;4H\x1b[6n\x1b[c");
        assert_eq!(e.take_replies(), b"\x1b[3;4R\x1b[?62;22c");
        let e = emu(10, 5, "\x1b[?25l\x1b[?2004h\x1b]0;Aegist\x07");
        assert!(!e.term.cursor_visible && e.term.bracketed_paste);
        assert_eq!(e.term.title.as_deref(), Some("Aegist"));
    }

    #[test]
    fn scroll_regions_alt_screen_and_resizing() {
        let mut e = emu(5, 4, "1\r\n2\r\n3\r\n4\x1b[2;3r\x1b[3;1H\nX");
        assert_eq!((0..4).map(|y| e.term.row_text(y)).collect::<Vec<_>>(), vec!["1", "3", "X", "4"]);
        e.feed(b"\x1b[r\x1b[?1049h\x1b[Halt");
        assert_eq!(e.term.row_text(0), "alt");
        e.feed(b"\x1b[?1049l");
        assert_eq!(e.term.row_text(0), "1");
        let mut e = emu(10, 4, "a\r\nb\r\nc\r\nd");
        e.term.resize(6, 2);
        assert_eq!((e.term.row_text(0).as_str(), e.term.row_text(1).as_str()), ("c", "d"));
        assert_eq!(e.term.scrollback.len(), 2);
        e.term.resize(12, 5);
        assert_eq!(e.term.rows, 5);
        assert_eq!(e.term.text_between((0, 0), (3, 5)), "a\nb\nc\nd");
    }
}
