//! The input box: a multi-line editor with history, word movement, paste,
//! and a menu of completions (slash commands, file paths).

use super::style::{pal, Style};
use super::text::{self, char_width};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Submit(String),
    /// Ctrl-C on an empty line.
    Interrupt,
    /// Ctrl-D on an empty line.
    Eof,
    ClearScreen,
    /// Esc on an empty line.
    Escape,
}

/// A completion: what replaces the word being typed, and a description.
#[derive(Clone, Debug)]
pub struct Completion {
    pub insert: String,
    pub label: String,
    pub detail: String,
}

#[derive(Default)]
pub struct Editor {
    pub buf: String,
    cursor: usize,
    history: Vec<String>,
    hist_pos: Option<usize>,
    stash: String,
    pub menu: Vec<Completion>,
    pub menu_sel: usize,
    menu_dismissed: bool,
}

fn prev_boundary(s: &str, i: usize) -> usize {
    s[..i].char_indices().next_back().map_or(0, |(j, _)| j)
}

fn next_boundary(s: &str, i: usize) -> usize {
    s[i..].chars().next().map_or(i, |c| i + c.len_utf8())
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

impl Editor {
    pub fn new(history: Vec<String>) -> Editor {
        Editor { history, ..Default::default() }
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn set(&mut self, text: &str) {
        self.buf = text.to_string();
        self.cursor = self.buf.len();
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn insert(&mut self, s: &str) {
        self.buf.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.hist_pos = None;
        self.menu_dismissed = false;
    }

    pub fn paste(&mut self, s: &str) {
        let clean = s.replace("\r\n", "\n").replace('\r', "\n");
        self.insert(&clean);
    }

    fn line_start(&self) -> usize {
        self.buf[..self.cursor].rfind('\n').map_or(0, |i| i + 1)
    }

    fn line_end(&self) -> usize {
        self.buf[self.cursor..].find('\n').map_or(self.buf.len(), |i| self.cursor + i)
    }

    fn word_left(&self) -> usize {
        let chars: Vec<(usize, char)> = self.buf[..self.cursor].char_indices().collect();
        let mut i = chars.len();
        while i > 0 && !is_word(chars[i - 1].1) {
            i -= 1;
        }
        while i > 0 && is_word(chars[i - 1].1) {
            i -= 1;
        }
        chars.get(i).map_or(0, |c| c.0).min(self.cursor)
    }

    fn word_right(&self) -> usize {
        let mut i = self.cursor;
        let rest: Vec<char> = self.buf[self.cursor..].chars().collect();
        let mut k = 0;
        while k < rest.len() && !is_word(rest[k]) {
            i += rest[k].len_utf8();
            k += 1;
        }
        while k < rest.len() && is_word(rest[k]) {
            i += rest[k].len_utf8();
            k += 1;
        }
        i
    }

    /// Move up or down a line in the text; false at the first/last line.
    fn vertical(&mut self, down: bool) -> bool {
        let start = self.line_start();
        let col = self.buf[start..self.cursor].chars().count();
        if down {
            let end = self.line_end();
            if end >= self.buf.len() {
                return false;
            }
            let next_start = end + 1;
            let next_end = self.buf[next_start..].find('\n').map_or(self.buf.len(), |i| next_start + i);
            self.cursor = self.buf[next_start..next_end].char_indices().nth(col).map_or(next_end, |(i, _)| next_start + i);
        } else {
            if start == 0 {
                return false;
            }
            let prev_start = self.buf[..start - 1].rfind('\n').map_or(0, |i| i + 1);
            let prev_end = start - 1;
            self.cursor = self.buf[prev_start..prev_end].char_indices().nth(col).map_or(prev_end, |(i, _)| prev_start + i);
        }
        true
    }

    fn history_move(&mut self, older: bool) {
        if self.history.is_empty() {
            return;
        }
        let pos = match (self.hist_pos, older) {
            (None, true) => {
                self.stash = self.buf.clone();
                self.history.len() - 1
            }
            (None, false) => return,
            (Some(0), true) => 0,
            (Some(p), true) => p - 1,
            (Some(p), false) if p + 1 >= self.history.len() => {
                self.hist_pos = None;
                self.buf = std::mem::take(&mut self.stash);
                self.cursor = self.buf.len();
                return;
            }
            (Some(p), false) => p + 1,
        };
        self.hist_pos = Some(pos);
        self.buf = self.history[pos].clone();
        self.cursor = self.buf.len();
    }

    /// The word the cursor is at the end of (for completion).
    pub fn current_word(&self) -> &str {
        let start = self.buf[..self.cursor].rfind(|c: char| c.is_whitespace()).map_or(0, |i| i + 1);
        &self.buf[start..self.cursor]
    }

    fn accept_completion(&mut self) -> bool {
        let Some(c) = self.menu.get(self.menu_sel).cloned() else { return false };
        let word = self.current_word().len();
        self.buf.replace_range(self.cursor - word..self.cursor, &c.insert);
        self.cursor = self.cursor - word + c.insert.len();
        self.menu.clear();
        self.menu_dismissed = c.insert.ends_with('/');
        if !c.insert.ends_with('/') {
            self.insert(" ");
        }
        true
    }

    /// Give the editor its completions for the current word (called after
    /// every change).
    pub fn set_menu(&mut self, items: Vec<Completion>) {
        if self.menu_dismissed {
            self.menu.clear();
            return;
        }
        self.menu = items;
        if self.menu_sel >= self.menu.len() {
            self.menu_sel = 0;
        }
    }

    pub fn key(&mut self, k: KeyEvent) -> Action {
        if k.kind == KeyEventKind::Release {
            return Action::None;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        let shift = k.modifiers.contains(KeyModifiers::SHIFT);
        let menu_open = !self.menu.is_empty();
        match k.code {
            KeyCode::Enter if shift || alt => self.insert("\n"),
            KeyCode::Char('j') if ctrl => self.insert("\n"),
            KeyCode::Enter if menu_open && self.menu.get(self.menu_sel).is_some_and(|c| c.insert.trim() != self.current_word()) => {
                self.accept_completion();
            }
            KeyCode::Enter => {
                if self.buf[..self.cursor].ends_with('\\') {
                    self.buf.remove(self.cursor - 1);
                    self.cursor -= 1;
                    self.insert("\n");
                    return Action::None;
                }
                let text = std::mem::take(&mut self.buf);
                self.cursor = 0;
                self.hist_pos = None;
                self.menu.clear();
                self.menu_dismissed = false;
                if !text.trim().is_empty() && self.history.last() != Some(&text) {
                    self.history.push(text.clone());
                }
                return Action::Submit(text);
            }
            KeyCode::Tab if menu_open => {
                self.accept_completion();
            }
            KeyCode::Up if menu_open => self.menu_sel = (self.menu_sel + self.menu.len() - 1) % self.menu.len(),
            KeyCode::Down if menu_open => self.menu_sel = (self.menu_sel + 1) % self.menu.len(),
            KeyCode::Esc if menu_open => {
                self.menu.clear();
                self.menu_dismissed = true;
            }
            KeyCode::Esc => return Action::Escape,
            KeyCode::Char('c') if ctrl => {
                if self.buf.is_empty() {
                    return Action::Interrupt;
                }
                self.buf.clear();
                self.cursor = 0;
                self.menu.clear();
            }
            KeyCode::Char('d') if ctrl => {
                if self.buf.is_empty() {
                    return Action::Eof;
                }
                if self.cursor < self.buf.len() {
                    let n = next_boundary(&self.buf, self.cursor);
                    self.buf.drain(self.cursor..n);
                }
            }
            KeyCode::Char('l') if ctrl => return Action::ClearScreen,
            KeyCode::Char('a') if ctrl => self.cursor = self.line_start(),
            KeyCode::Char('e') if ctrl => self.cursor = self.line_end(),
            KeyCode::Char('u') if ctrl => {
                let s = self.line_start();
                self.buf.drain(s..self.cursor);
                self.cursor = s;
            }
            KeyCode::Char('k') if ctrl => {
                let e = self.line_end();
                self.buf.drain(self.cursor..e);
            }
            KeyCode::Char('w') if ctrl => {
                let s = self.word_left();
                self.buf.drain(s..self.cursor);
                self.cursor = s;
            }
            KeyCode::Backspace if alt || ctrl => {
                let s = self.word_left();
                self.buf.drain(s..self.cursor);
                self.cursor = s;
            }
            KeyCode::Char('b') if alt => self.cursor = self.word_left(),
            KeyCode::Char('f') if alt => self.cursor = self.word_right(),
            KeyCode::Left if ctrl || alt => self.cursor = self.word_left(),
            KeyCode::Right if ctrl || alt => self.cursor = self.word_right(),
            KeyCode::Char(c) if !ctrl => self.insert(&c.to_string()),
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let p = prev_boundary(&self.buf, self.cursor);
                    self.buf.drain(p..self.cursor);
                    self.cursor = p;
                    self.menu_dismissed = false;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.buf.len() {
                    let n = next_boundary(&self.buf, self.cursor);
                    self.buf.drain(self.cursor..n);
                }
            }
            KeyCode::Left => self.cursor = prev_boundary(&self.buf, self.cursor),
            KeyCode::Right => self.cursor = next_boundary(&self.buf, self.cursor),
            KeyCode::Home => self.cursor = self.line_start(),
            KeyCode::End => self.cursor = self.line_end(),
            KeyCode::Up => {
                if !self.vertical(false) {
                    self.history_move(true);
                }
            }
            KeyCode::Down => {
                if !self.vertical(true) {
                    self.history_move(false);
                }
            }
            _ => {}
        }
        Action::None
    }

    /// The box with the text in it, and where the cursor goes (row, column).
    pub fn render(&self, width: usize, placeholder: &str, accent: bool) -> (Vec<String>, (usize, usize)) {
        let w = width.saturating_sub(1).max(20);
        let inner = w - 4;
        let border = Style::new().fg(if accent { pal::VIOLET.shade(0.75) } else { pal::BORDER });
        let prompt = Style::new().fg(pal::VIOLET).bold().paint("❯");
        let mut rows: Vec<String> = vec![border.paint(&format!("╭{}╮", "─".repeat(w - 2)))];
        let mut cursor = (1, 4);
        if self.buf.is_empty() {
            let ph = Style::new().fg(pal::FAINT).italic().paint(&text::truncate(placeholder, inner - 2));
            rows.push(format!("{} {prompt} {} {}", border.paint("│"), text::pad(&ph, inner - 2), border.paint("│")));
        } else {
            let text_w = inner - 2;
            let mut lines: Vec<String> = vec![String::new()];
            let mut col = 0;
            let mut pos = 0;
            let mut found = false;
            for c in self.buf.chars() {
                if pos == self.cursor {
                    cursor = (lines.len(), 4 + col);
                    found = true;
                }
                if c == '\n' {
                    lines.push(String::new());
                    col = 0;
                } else {
                    let cw = char_width(c);
                    if col + cw > text_w {
                        lines.push(String::new());
                        col = 0;
                    }
                    lines.last_mut().expect("a line").push(c);
                    col += cw;
                }
                pos += c.len_utf8();
            }
            if !found {
                if col >= text_w {
                    lines.push(String::new());
                    col = 0;
                }
                cursor = (lines.len(), 4 + col);
            }
            for (i, l) in lines.iter().enumerate() {
                let lead = if i == 0 { prompt.clone() } else { " ".into() };
                rows.push(format!("{} {lead} {} {}", border.paint("│"), text::pad(l, text_w), border.paint("│")));
            }
        }
        rows.push(border.paint(&format!("╰{}╯", "─".repeat(w - 2))));
        (rows, cursor)
    }

    /// The completion menu, under the box.
    pub fn render_menu(&self, width: usize) -> Vec<String> {
        let label_w = self.menu.iter().map(|c| text::width(&c.label)).max().unwrap_or(0).min(28);
        self.menu
            .iter()
            .enumerate()
            .take(8)
            .map(|(i, c)| {
                let sel = i == self.menu_sel;
                let label = text::pad(&c.label, label_w);
                let line = if sel {
                    format!("  {} {}  {}", Style::new().fg(pal::VIOLET).bold().paint("▸"), Style::new().fg(pal::VIOLET).bold().paint(&label),
                            Style::new().fg(pal::DIM).paint(&c.detail))
                } else {
                    format!("    {}  {}", Style::new().fg(pal::SKY).paint(&label), Style::new().fg(pal::FAINT).paint(&c.detail))
                };
                text::truncate(&line, width.saturating_sub(1))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }
    fn typed(e: &mut Editor, s: &str) {
        for c in s.chars() {
            e.key(key(KeyCode::Char(c)));
        }
    }

    #[test]
    fn editing_words_lines_and_history() {
        let mut e = Editor::new(vec!["old one".into()]);
        typed(&mut e, "hello brave world");
        e.key(ctrl('w'));
        assert_eq!(e.buf, "hello brave ");
        e.key(KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL));
        assert_eq!(e.current_word(), "");
        e.key(ctrl('a'));
        typed(&mut e, ">");
        assert_eq!(e.buf, ">hello brave ");
        assert_eq!(e.key(key(KeyCode::Enter)), Action::Submit(">hello brave ".into()));
        e.key(key(KeyCode::Up));
        assert_eq!(e.buf, ">hello brave ");
        e.key(key(KeyCode::Up));
        assert_eq!(e.buf, "old one");
        e.key(key(KeyCode::Down));
        e.key(key(KeyCode::Down));
        assert_eq!(e.buf, "");
        typed(&mut e, "line one\\");
        e.key(key(KeyCode::Enter));
        typed(&mut e, "two");
        assert_eq!(e.buf, "line one\ntwo");
        e.key(key(KeyCode::Up));
        typed(&mut e, "!");
        assert_eq!(e.buf, "lin!e one\ntwo");
        e.key(ctrl('c'));
        assert!(e.buf.is_empty());
        assert_eq!(e.key(ctrl('c')), Action::Interrupt);
        assert_eq!(e.key(ctrl('d')), Action::Eof);
        e.paste("a\r\nb");
        assert_eq!(e.buf, "a\nb");
    }

    #[test]
    fn completions_replace_the_word() {
        let mut e = Editor::new(vec![]);
        typed(&mut e, "/wr");
        e.set_menu(vec![Completion { insert: "/write".into(), label: "/write".into(), detail: "new file".into() }]);
        e.key(key(KeyCode::Tab));
        assert_eq!(e.buf, "/write ");
        assert!(e.menu.is_empty());
    }

    #[test]
    fn the_box_wraps_and_places_the_cursor() {
        let mut e = Editor::new(vec![]);
        let (rows, cur) = e.render(30, "type here", false);
        assert_eq!(rows.len(), 3);
        assert_eq!(cur, (1, 4));
        typed(&mut e, &"x".repeat(30));
        let (rows, cur) = e.render(30, "", false);
        assert_eq!(rows.len(), 4); // 23 columns per line: two lines of text
        assert!(rows.iter().all(|r| text::width(r) == 29));
        assert_eq!(cur, (2, 4 + 7));
    }
}
