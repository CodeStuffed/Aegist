//! Measuring and fitting styled text: display width (wide characters count
//! twice, escape sequences count nothing), cutting to a width, padding, and
//! word wrapping.

use super::style::{Style, RESET};
use unicode_width::UnicodeWidthChar;

/// Iterate over a string's characters, skipping ANSI escape sequences,
/// yielding (byte offset, char).
fn visible_chars(s: &str) -> impl Iterator<Item = (usize, char)> + '_ {
    let mut it = s.char_indices().peekable();
    std::iter::from_fn(move || {
        while let Some((i, c)) = it.next() {
            if c == '\x1b' {
                if it.peek().map(|p| p.1) == Some('[') {
                    it.next();
                    for (_, c2) in it.by_ref() {
                        if ('@'..='~').contains(&c2) {
                            break;
                        }
                    }
                } else if it.peek().map(|p| p.1) == Some(']') {
                    // OSC: until BEL or ST
                    let mut prev = ' ';
                    for (_, c2) in it.by_ref() {
                        if c2 == '\x07' || (prev == '\x1b' && c2 == '\\') {
                            break;
                        }
                        prev = c2;
                    }
                }
                continue;
            }
            return Some((i, c));
        }
        None
    })
}

pub fn char_width(c: char) -> usize {
    if c == '\t' {
        4
    } else {
        c.width().unwrap_or(0)
    }
}

/// Display width of `s`, ignoring escape sequences.
pub fn width(s: &str) -> usize {
    visible_chars(s).map(|(_, c)| char_width(c)).sum()
}

pub fn strip_ansi(s: &str) -> String {
    visible_chars(s).map(|(_, c)| c).collect()
}

/// Cut `s` to at most `max` columns, ending with "…" if anything was cut.
/// Escape sequences are kept, and styling is reset at the end.
pub fn truncate(s: &str, max: usize) -> String {
    if width(s) <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut w = 0;
    let mut last = 0;
    for (i, c) in visible_chars(s) {
        out.push_str(&s[last..i]); // escape sequences in between
        let cw = char_width(c);
        if w + cw > max - 1 {
            out.push('…');
            if s.contains('\x1b') {
                out.push_str(RESET);
            }
            return out;
        }
        out.push(c);
        w += cw;
        last = i + c.len_utf8();
    }
    out
}

/// `s` padded with spaces to `w` columns (unchanged if already wider).
pub fn pad(s: &str, w: usize) -> String {
    let have = width(s);
    if have >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - have))
    }
}

/// Word-wrap plain text to `w` columns. Existing line breaks are kept;
/// words longer than a line are split.
pub fn wrap(text: &str, w: usize) -> Vec<String> {
    let w = w.max(8);
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        let mut lw = 0;
        let indent: String = para.chars().take_while(|c| *c == ' ').collect();
        for word in para.split(' ').filter(|s| !s.is_empty()) {
            let ww = width(word);
            if lw > 0 && lw + 1 + ww > w {
                out.push(std::mem::take(&mut line));
                line.push_str(&indent);
                lw = indent.len();
            }
            if ww > w {
                for c in word.chars() {
                    let cw = char_width(c);
                    if lw + cw > w {
                        out.push(std::mem::take(&mut line));
                        lw = 0;
                    }
                    line.push(c);
                    lw += cw;
                }
                continue;
            }
            if lw > 0 && !(lw == indent.len() && line == indent) {
                line.push(' ');
                lw += 1;
            } else if lw == 0 {
                line.push_str(&indent);
                lw = indent.len();
            }
            line.push_str(word);
            lw += ww;
        }
        out.push(line);
    }
    out
}

/// A run of styled text.
pub type Span = (Style, String);

pub fn render(spans: &[Span]) -> String {
    spans.iter().map(|(s, t)| s.paint(t)).collect()
}

/// Break styled spans into lines of at most `w` columns (by character).
pub fn wrap_spans(spans: &[Span], w: usize) -> Vec<Vec<Span>> {
    let w = w.max(4);
    let mut lines: Vec<Vec<Span>> = vec![Vec::new()];
    let mut lw = 0;
    for (style, text) in spans {
        let mut cur = String::new();
        for c in text.chars() {
            let cw = char_width(c);
            if lw + cw > w {
                if !cur.is_empty() {
                    lines.last_mut().expect("a line").push((*style, std::mem::take(&mut cur)));
                }
                lines.push(Vec::new());
                lw = 0;
            }
            cur.push(c);
            lw += cw;
        }
        if !cur.is_empty() {
            lines.last_mut().expect("a line").push((*style, cur));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_ignore_escapes_and_count_wide_chars() {
        assert_eq!(width("\x1b[1;38;2;1;2;3mhi\x1b[0m"), 2);
        assert_eq!(width("日本"), 4);
        assert_eq!(width("\x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\"), 4);
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m!"), "red!");
    }

    #[test]
    fn truncating_and_padding() {
        assert_eq!(strip_ansi(&truncate("\x1b[31mhello world\x1b[0m", 6)), "hello…");
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(pad("ab", 4), "ab  ");
        assert_eq!(width(&truncate("日本語テキスト", 5)), 5);
    }

    #[test]
    fn wrapping_keeps_words_and_indent() {
        assert_eq!(wrap("the quick brown fox jumps", 10), vec!["the quick", "brown fox", "jumps"]);
        assert_eq!(wrap("  indented text that wraps", 12), vec!["  indented", "  text that", "  wraps"]);
        assert_eq!(wrap("a\n\nb", 10), vec!["a", "", "b"]);
        let spans = vec![(Style::new(), "abcdef".to_string()), (Style::new().bold(), "gh".to_string())];
        let lines = wrap_spans(&spans, 4);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].iter().map(|s| s.1.as_str()).collect::<String>(), "efgh");
    }
}
