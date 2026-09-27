//! Keys pressed in the window, turned into the bytes a terminal program
//! reads (the same ones any terminal sends).

#[derive(Clone, Debug, PartialEq)]
pub enum Key {
    /// Text the key types.
    Text(String),
    Enter,
    Backspace,
    Tab,
    Escape,
    Space,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    Insert,
    Delete,
    F(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mods {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

impl Mods {
    /// The xterm modifier number (1 + shift + 2·alt + 4·ctrl).
    fn code(self) -> u8 {
        1 + self.shift as u8 + 2 * self.alt as u8 + 4 * self.ctrl as u8
    }
    fn any(self) -> bool {
        self.shift || self.ctrl || self.alt
    }
}

/// The bytes for `key`, or None if it types nothing.
pub fn encode(key: &Key, m: Mods, app_cursor: bool) -> Option<Vec<u8>> {
    let esc = |s: &str| Some(s.as_bytes().to_vec());
    let cursor = |c: char| {
        if m.any() {
            Some(format!("\x1b[1;{}{c}", m.code()).into_bytes())
        } else if app_cursor {
            Some(format!("\x1bO{c}").into_bytes())
        } else {
            Some(format!("\x1b[{c}").into_bytes())
        }
    };
    let tilde = |n: u8| {
        if m.any() {
            Some(format!("\x1b[{n};{}~", m.code()).into_bytes())
        } else {
            Some(format!("\x1b[{n}~").into_bytes())
        }
    };
    match key {
        // shift/alt+enter: a new line in Aegist's input box
        Key::Enter if m.shift || m.alt => esc("\x1b\r"),
        Key::Enter => esc("\r"),
        Key::Backspace if m.ctrl => esc("\x08"),
        Key::Backspace if m.alt => esc("\x1b\x7f"),
        Key::Backspace => esc("\x7f"),
        Key::Tab if m.shift => esc("\x1b[Z"),
        Key::Tab => esc("\t"),
        Key::Escape => esc("\x1b"),
        Key::Space if m.ctrl => Some(vec![0]),
        Key::Space if m.alt => esc("\x1b "),
        Key::Space => esc(" "),
        Key::Up => cursor('A'),
        Key::Down => cursor('B'),
        Key::Right => cursor('C'),
        Key::Left => cursor('D'),
        Key::Home => cursor('H'),
        Key::End => cursor('F'),
        Key::Insert => tilde(2),
        Key::Delete => tilde(3),
        Key::PageUp => tilde(5),
        Key::PageDown => tilde(6),
        Key::F(n @ 1..=4) if !m.any() => Some(format!("\x1bO{}", (b'P' + n - 1) as char).into_bytes()),
        Key::F(n @ 1..=4) => Some(format!("\x1b[1;{}{}", m.code(), (b'P' + n - 1) as char).into_bytes()),
        Key::F(n) => {
            let code = match n {
                5 => 15,
                6 => 17,
                7 => 18,
                8 => 19,
                9 => 20,
                10 => 21,
                11 => 23,
                12 => 24,
                _ => return None,
            };
            tilde(code)
        }
        Key::Text(t) if t.is_empty() => None,
        Key::Text(t) => {
            let mut chars = t.chars();
            let (c, single) = (chars.next()?, chars.next().is_none());
            let mut out = Vec::new();
            if m.ctrl && single {
                // ctrl+letter and friends: the control characters
                let ctl = match c.to_ascii_lowercase() {
                    l @ 'a'..='z' => Some(l as u8 - b'a' + 1),
                    '@' | '2' => Some(0),
                    '[' | '3' => Some(0x1b),
                    '\\' | '4' => Some(0x1c),
                    ']' | '5' => Some(0x1d),
                    '^' | '6' => Some(0x1e),
                    '_' | '-' | '7' => Some(0x1f),
                    '/' => Some(0x1f),
                    _ => None,
                };
                if let Some(b) = ctl {
                    if m.alt {
                        out.push(0x1b);
                    }
                    out.push(b);
                    return Some(out);
                }
            }
            if m.alt {
                out.push(0x1b);
            }
            out.extend_from_slice(t.as_bytes());
            Some(out)
        }
    }
}

/// Pasted text as a program expects it: marked as a paste when it asked
/// (so line breaks don't send), and never able to end that marking early.
pub fn paste(text: &str, bracketed: bool) -> Vec<u8> {
    let clean = text.replace("\x1b[201~", "").replace("\r\n", "\r").replace('\n', "\r");
    if bracketed {
        format!("\x1b[200~{clean}\x1b[201~").into_bytes()
    } else {
        clean.into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(shift: bool, ctrl: bool, alt: bool) -> Mods {
        Mods { shift, ctrl, alt }
    }

    #[test]
    fn keys_become_terminal_bytes() {
        let none = Mods::default();
        assert_eq!(encode(&Key::Enter, none, false).unwrap(), b"\r");
        assert_eq!(encode(&Key::Enter, m(true, false, false), false).unwrap(), b"\x1b\r");
        assert_eq!(encode(&Key::Backspace, none, false).unwrap(), b"\x7f");
        assert_eq!(encode(&Key::Up, none, false).unwrap(), b"\x1b[A");
        assert_eq!(encode(&Key::Up, none, true).unwrap(), b"\x1bOA");
        assert_eq!(encode(&Key::Left, m(false, true, false), false).unwrap(), b"\x1b[1;5D");
        assert_eq!(encode(&Key::Delete, none, false).unwrap(), b"\x1b[3~");
        assert_eq!(encode(&Key::F(1), none, false).unwrap(), b"\x1bOP");
        assert_eq!(encode(&Key::F(5), none, false).unwrap(), b"\x1b[15~");
        assert_eq!(encode(&Key::Text("c".into()), m(false, true, false), false).unwrap(), vec![3]);
        assert_eq!(encode(&Key::Text("L".into()), m(true, true, false), false).unwrap(), vec![12]);
        assert_eq!(encode(&Key::Text("x".into()), m(false, false, true), false).unwrap(), b"\x1bx");
        assert_eq!(encode(&Key::Text("é".into()), none, false).unwrap(), "é".as_bytes());
        assert_eq!(encode(&Key::Tab, m(true, false, false), false).unwrap(), b"\x1b[Z");
        assert!(encode(&Key::Text(String::new()), none, false).is_none());
    }

    #[test]
    fn pastes_are_marked_and_safe() {
        assert_eq!(paste("a\nb", true), b"\x1b[200~a\rb\x1b[201~");
        assert_eq!(paste("x\x1b[201~rm -rf /\n", true), b"\x1b[200~xrm -rf /\r\x1b[201~");
        assert_eq!(paste("a\r\nb", false), b"a\rb");
    }
}
