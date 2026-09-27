//! Syntax highlighting in the aurora palette (syntect's grammars, our colors).

use super::style::{Rgb, Style};
use super::text::Span;
use std::str::FromStr;
use std::sync::OnceLock;
use syntect::highlighting::{Color, FontStyle, HighlightState, Highlighter, RangedHighlightIterator, ScopeSelectors, StyleModifier, Theme,
                            ThemeItem, ThemeSettings};
use syntect::parsing::{ParseState, ScopeStack, SyntaxReference, SyntaxSet};

/// Stands for "the terminal's own text color" in the theme.
const DEFAULT_FG: Color = Color { r: 1, g: 2, b: 3, a: 255 };

fn syntaxes() -> &'static SyntaxSet {
    static SET: OnceLock<SyntaxSet> = OnceLock::new();
    SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

/// Load the grammars in the background, so the first highlight is instant.
pub fn warm_up() {
    std::thread::spawn(|| {
        syntaxes();
        theme();
    });
}

fn theme() -> &'static Theme {
    static THEME: OnceLock<Theme> = OnceLock::new();
    THEME.get_or_init(|| {
        let item = |scopes: &str, hex: u32, font: FontStyle| ThemeItem {
            scope: ScopeSelectors::from_str(scopes).expect("valid scope selector"),
            style: StyleModifier {
                foreground: Some(Color { r: (hex >> 16) as u8, g: (hex >> 8) as u8, b: hex as u8, a: 255 }),
                background: None,
                font_style: Some(font),
            },
        };
        let n = FontStyle::empty();
        Theme {
            name: Some("aurora".into()),
            author: None,
            settings: ThemeSettings { foreground: Some(DEFAULT_FG), ..Default::default() },
            scopes: vec![
                item("comment, punctuation.definition.comment", 0x6B7394, FontStyle::ITALIC),
                item("string, string.quoted, punctuation.definition.string", 0x9ECE6A, n),
                item("constant.character.escape, string.regexp", 0x89DDFF, n),
                item("constant.numeric, constant.language, constant.character, support.constant, constant.other", 0xFF9E64, n),
                item("keyword, storage, storage.modifier, keyword.control, keyword.other", 0xBB9AF7, n),
                item("keyword.operator, punctuation.accessor, punctuation.separator.namespace", 0x89DDFF, n),
                item("storage.type", 0xBB9AF7, FontStyle::ITALIC),
                item("entity.name.function, support.function, variable.function, meta.function-call.generic", 0x7AA2F7, n),
                item("entity.name.type, entity.name.class, entity.name.struct, entity.name.enum, entity.name.trait, support.type, \
                      support.class, entity.other.inherited-class, storage.type.class", 0x2AC3DE, n),
                item("variable.parameter", 0xE0AF68, n),
                item("variable.language, variable.other.readwrite.instance", 0xF7768E, FontStyle::ITALIC),
                item("entity.name.tag", 0xF7768E, n),
                item("entity.other.attribute-name", 0xE0AF68, FontStyle::ITALIC),
                item("meta.preprocessor, keyword.control.import, keyword.control.directive", 0x7DCFFF, n),
                item("entity.name.namespace, entity.name.module", 0x2AC3DE, n),
                item("markup.heading", 0x7AA2F7, FontStyle::BOLD),
                item("markup.bold", 0xE0AF68, FontStyle::BOLD),
                item("markup.italic", 0xE0AF68, FontStyle::ITALIC),
                item("markup.inserted", 0x9ECE6A, n),
                item("markup.deleted", 0xF7768E, n),
                item("invalid", 0xF87171, FontStyle::UNDERLINE),
            ],
        }
    })
}

fn syntax_for(lang_id: &str) -> Option<&'static SyntaxReference> {
    let ss = syntaxes();
    let alias = match lang_id {
        "typescript" | "javascript" => "js",
        "cpp" => "cpp",
        "bash" | "shell" => "sh",
        "cs" => "cs",
        "toml" | "ini" => "yaml",
        "dockerfile" => "sh",
        "kotlin" | "swift" | "dart" | "scala" => "java",
        "zig" | "glsl" | "wgsl" | "cuda" => "c",
        "vue" | "svelte" => "html",
        other => other,
    };
    ss.find_syntax_by_token(alias).or_else(|| ss.find_syntax_by_extension(alias))
}

/// A highlighter that keeps its state from line to line.
pub struct Lines {
    syntax: Option<&'static SyntaxReference>,
    parse: Option<ParseState>,
    hl: Option<HighlightState>,
}

impl Lines {
    pub fn new(lang_id: Option<&str>) -> Lines {
        let syntax = lang_id.and_then(syntax_for);
        let parse = syntax.map(ParseState::new);
        let hl = syntax.map(|_| HighlightState::new(&Highlighter::new(theme()), ScopeStack::new()));
        Lines { syntax, parse, hl }
    }

    /// One line of code (without its newline) as styled spans.
    pub fn line(&mut self, line: &str) -> Vec<Span> {
        let (Some(_), Some(parse), Some(hl)) = (self.syntax, self.parse.as_mut(), self.hl.as_mut()) else {
            return vec![(Style::new(), line.to_string())];
        };
        let with_nl = format!("{line}\n");
        let Ok(ops) = parse.parse_line(&with_nl, syntaxes()) else {
            return vec![(Style::new(), line.to_string())];
        };
        let highlighter = Highlighter::new(theme());
        let mut out: Vec<Span> = Vec::new();
        for (st, text, _) in RangedHighlightIterator::new(hl, &ops, &with_nl, &highlighter) {
            let text = text.trim_end_matches('\n');
            if text.is_empty() {
                continue;
            }
            let mut s = Style::new();
            if st.foreground != DEFAULT_FG {
                s = s.fg(Rgb(st.foreground.r, st.foreground.g, st.foreground.b));
            }
            s.italic = st.font_style.contains(FontStyle::ITALIC);
            s.bold = st.font_style.contains(FontStyle::BOLD);
            s.underline = st.font_style.contains(FontStyle::UNDERLINE);
            match out.last_mut() {
                Some(last) if last.0 == s => last.1.push_str(text),
                _ => out.push((s, text.to_string())),
            }
        }
        out
    }
}

/// Highlight a whole piece of code, line by line.
pub fn code(text: &str, lang_id: Option<&str>) -> Vec<Vec<Span>> {
    let mut h = Lines::new(lang_id);
    text.lines().map(|l| h.line(l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlights_known_languages_and_passes_others_through() {
        let lines = code("def f(x):\n    return 'hi'  # note\n", Some("python"));
        assert_eq!(lines.len(), 2);
        let joined: String = lines[1].iter().map(|s| s.1.as_str()).collect();
        assert_eq!(joined, "    return 'hi'  # note");
        assert!(lines[1].iter().any(|(s, t)| t.contains("return") && s.fg == Some(Rgb::hex(0xBB9AF7))));
        assert!(lines[1].iter().any(|(s, t)| t.contains("note") && s.italic));
        let plain = code("anything at all", Some("no-such-language"));
        assert_eq!(plain[0], vec![(Style::new(), "anything at all".to_string())]);
    }
}
