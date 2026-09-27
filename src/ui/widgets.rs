//! The pieces the interface is built from: the banner, boxes, code panels
//! (highlighted, with the model's unsure tokens marked), diffs, bars and
//! the verification report.

use super::highlight;
use super::style::{self, pal, Rgb, Style, RESET};
use super::text::{self, Span};
use crate::verify::{Report, Status, Verdict};

const LOGO: [[&str; 6]; 6] = [
    [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    [" ██████╗ ", "██╔════╝ ", "██║  ███╗", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
];

/// The AEGIST logo in block letters, lit by the aurora gradient running
/// diagonally across it; the letters' shadows in a deeper shade.
pub fn banner(width: usize) -> Vec<String> {
    let rows: Vec<String> = (0..6).map(|r| LOGO.iter().map(|letter| letter[r]).collect::<String>()).collect();
    let w = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
    if width < w + 4 {
        return vec![format!("  {}", style::gradient_styled("◆ A E G I S T", 0.0, true))];
    }
    rows.iter()
        .enumerate()
        .map(|(r, row)| {
            let mut line = String::from("  ");
            for (c, ch) in row.chars().enumerate() {
                if ch == ' ' {
                    line.push(' ');
                    continue;
                }
                let t = (c as f32 + r as f32 * 1.6) / (w as f32 + 9.0);
                let color = pal::aurora(t);
                let s = if ch == '█' { Style::new().fg(color) } else { Style::new().fg(color.shade(0.5)) };
                line.push_str(&s.prefix());
                line.push(ch);
            }
            line.push_str(RESET);
            line
        })
        .collect()
}

/// A section heading, the techy way: `▍ C O M M A N D S ━━━━━━━━──────`,
/// the capitals in the gradient and the rule fading out.
pub fn heading(title: &str, phase: f32, width: usize) -> String {
    let label = style::gradient_styled(&style::spaced(title), phase, true);
    let used = text::width(&style::spaced(title)) + 4;
    let rest = width.saturating_sub(used + 2).min(48);
    let mut rule = String::new();
    for i in 0..rest {
        let t = i as f32 / rest.max(1) as f32;
        let c = pal::aurora(phase + t * 0.3).lerp(pal::BORDER, t);
        rule.push_str(&Style::new().fg(c).paint(if i < rest / 3 { "━" } else { "─" }));
    }
    format!("{} {label} {rule}", Style::new().fg(pal::VIOLET).bold().paint("▍"))
}

/// A boot-screen style status tag: `[ OK ]`, `[WAIT]`, `[FAIL]`.
pub fn tag(state: &str, color: Rgb) -> String {
    format!("{}{}{}", style::faint("["), Style::new().fg(color).bold().paint(&format!("{state:^4}")), style::faint("]"))
}

/// `lines` in a box with rounded corners and an optional title in the top border.
pub fn boxed(title: Option<&str>, lines: &[String], width: usize, border: Rgb) -> Vec<String> {
    let inner = width.saturating_sub(4).max(10);
    let b = Style::new().fg(border);
    let mut out = Vec::new();
    let top = match title {
        Some(t) => {
            let t = text::truncate(t, inner.saturating_sub(2));
            let used = text::width(&t) + 3;
            format!("{}{t}{}", b.paint("╭─ "), b.paint(&format!(" {}╮", "─".repeat((inner + 2).saturating_sub(used)))))
        }
        None => b.paint(&format!("╭{}╮", "─".repeat(inner + 2))),
    };
    out.push(top);
    for l in lines {
        let l = text::truncate(l, inner);
        out.push(format!("{} {} {}", b.paint("│"), text::pad(&l, inner), b.paint("│")));
    }
    out.push(b.paint(&format!("╰{}╯", "─".repeat(inner + 2))));
    out
}

/// A horizontal rule with an optional label.
pub fn rule(label: &str, width: usize) -> String {
    let b = Style::new().fg(pal::BORDER);
    if label.is_empty() {
        return b.paint(&"─".repeat(width));
    }
    let used = text::width(label) + 4;
    format!("{}{label}{}", b.paint("── "), b.paint(&format!(" {}", "─".repeat(width.saturating_sub(used)))))
}

/// A smooth bar, `frac` full, `width` columns, in the gradient.
pub fn bar(frac: f32, width: usize) -> String {
    const EIGHTHS: [&str; 8] = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
    let frac = frac.clamp(0.0, 1.0);
    let cells = frac * width as f32;
    let full = cells.floor() as usize;
    let mut out = String::new();
    for i in 0..width {
        let color = pal::aurora(i as f32 / width.max(2) as f32);
        let glyph = if i < full {
            "█"
        } else if i == full && cells - full as f32 > 0.06 {
            EIGHTHS[((cells - full as f32) * 8.0).floor().clamp(0.0, 7.0) as usize]
        } else {
            out.push_str(&Style::new().fg(pal::BORDER).paint("·"));
            continue;
        };
        out.push_str(&Style::new().fg(color).paint(glyph));
    }
    out
}

/// A sparkline of `values` (newest last), at most `width` points.
pub fn sparkline(values: &[f32], width: usize) -> String {
    const TICKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let vals: Vec<f32> = values.iter().rev().take(width).rev().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return String::new();
    }
    let (lo, hi) = vals.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let n = vals.len();
    vals.iter()
        .enumerate()
        .map(|(i, v)| {
            let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
            Style::new().fg(pal::aurora(i as f32 / n.max(2) as f32)).paint(&TICKS[(t * 7.0).round() as usize].to_string())
        })
        .collect()
}

/// Mark the byte ranges in `unsure` (relative to the whole code) on the
/// spans of one line starting at byte `line_start`.
fn overlay_unsure(spans: Vec<Span>, line_start: usize, unsure: &[(usize, usize)]) -> Vec<Span> {
    if unsure.is_empty() {
        return spans;
    }
    let mut out = Vec::new();
    let mut pos = line_start;
    for (style, t) in spans {
        let mut piece = String::new();
        let mut piece_unsure = None;
        for c in t.chars() {
            let u = unsure.iter().any(|&(a, b)| pos >= a && pos < b);
            if piece_unsure.is_some_and(|p| p != u) {
                out.push((mark(style, piece_unsure.unwrap_or(false)), std::mem::take(&mut piece)));
            }
            piece_unsure = Some(u);
            piece.push(c);
            pos += c.len_utf8();
        }
        if !piece.is_empty() {
            out.push((mark(style, piece_unsure.unwrap_or(false)), piece));
        }
    }
    out
}

fn mark(mut s: Style, unsure: bool) -> Style {
    if unsure {
        s.underline = true;
        s.bg = Some(pal::UNSURE_BG);
    }
    s
}

/// Code with line numbers, highlighted; `unsure` byte ranges are
/// underlined on a warm background. Long lines wrap under themselves.
pub fn code_panel(code: &str, lang_id: Option<&str>, first_line: usize, width: usize, unsure: &[(usize, usize)]) -> Vec<String> {
    let last = first_line + code.lines().count().max(1);
    let num_w = last.to_string().len().max(2);
    let gutter = Style::new().fg(pal::FAINT);
    let bar_s = Style::new().fg(pal::BORDER);
    let avail = width.saturating_sub(num_w + 5).max(20);
    let mut hl = highlight::Lines::new(lang_id);
    let mut out = Vec::new();
    let mut offset = 0;
    for (i, line) in code.split('\n').enumerate() {
        if i + 1 == code.split('\n').count() && line.is_empty() && i > 0 {
            break;
        }
        let spans = overlay_unsure(hl.line(line), offset, unsure);
        offset += line.len() + 1;
        for (k, part) in text::wrap_spans(&spans, avail).into_iter().enumerate() {
            let num = if k == 0 { format!("{:>num_w$}", first_line + i) } else { " ".repeat(num_w) };
            out.push(format!("  {} {} {}", gutter.paint(&num), bar_s.paint(if k == 0 { "│" } else { "┆" }), text::render(&part)));
        }
    }
    out
}

/// A unified diff from `old` to `new`, highlighted, changed lines on
/// tinted backgrounds, `context` unchanged lines around each change.
pub fn diff(old: &str, new: &str, lang_id: Option<&str>, width: usize, context: usize) -> Vec<String> {
    let d = similar::TextDiff::from_lines(old, new);
    let old_hl = highlight::code(old, lang_id);
    let new_hl = highlight::code(new, lang_id);
    let lines_max = old.lines().count().max(new.lines().count()).max(1);
    let num_w = lines_max.to_string().len().max(2);
    let avail = width.saturating_sub(num_w + 7).max(20);
    let mut out = Vec::new();
    for (g, group) in d.grouped_ops(context).iter().enumerate() {
        if g > 0 {
            out.push(format!("  {}", Style::new().fg(pal::FAINT).paint(&format!("{:>num_w$}", "⋮"))));
        }
        for op in group {
            for change in d.iter_changes(op) {
                let (sign, num, spans, bg, sign_fg) = match change.tag() {
                    similar::ChangeTag::Equal => {
                        let i = change.old_index().unwrap_or(0);
                        (" ", change.new_index().unwrap_or(0) + 1, old_hl.get(i).cloned().unwrap_or_default(), None, pal::FAINT)
                    }
                    similar::ChangeTag::Delete => {
                        let i = change.old_index().unwrap_or(0);
                        ("-", i + 1, old_hl.get(i).cloned().unwrap_or_default(), Some(pal::DEL_BG), pal::DEL_FG)
                    }
                    similar::ChangeTag::Insert => {
                        let i = change.new_index().unwrap_or(0);
                        ("+", i + 1, new_hl.get(i).cloned().unwrap_or_default(), Some(pal::ADD_BG), pal::ADD_FG)
                    }
                };
                let spans: Vec<Span> = spans.into_iter().map(|(mut s, t)| {
                    if bg.is_some() {
                        s.bg = bg;
                    } else {
                        s.dim = true;
                    }
                    (s, t)
                }).collect();
                for (k, part) in text::wrap_spans(&spans, avail).into_iter().enumerate() {
                    let body = text::render(&part);
                    let fill = avail.saturating_sub(text::width(&body));
                    let fill = bg.map_or(String::new(), |c| Style::new().bg(c).paint(&" ".repeat(fill)));
                    let n = if k == 0 { format!("{num:>num_w$}") } else { " ".repeat(num_w) };
                    let lead = match bg {
                        Some(c) => Style::new().fg(sign_fg).bg(c).bold().paint(&format!(" {} ", if k == 0 { sign } else { " " })),
                        None => "   ".to_string(),
                    };
                    out.push(format!("  {} {lead}{body}{fill}", Style::new().fg(pal::FAINT).paint(&n)));
                }
            }
        }
    }
    out
}

/// "3 additions, 1 removal"
pub fn change_summary(old: &str, new: &str) -> String {
    let d = similar::TextDiff::from_lines(old, new);
    let (mut add, mut del) = (0, 0);
    for c in d.iter_all_changes() {
        match c.tag() {
            similar::ChangeTag::Insert => add += 1,
            similar::ChangeTag::Delete => del += 1,
            similar::ChangeTag::Equal => {}
        }
    }
    let plural = |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    format!("{}, {}", plural(add, "addition", "additions"), plural(del, "removal", "removals"))
}

fn status_row(label: &str, status: &Status, pass: &str) -> String {
    let (icon, color, msg) = match status {
        Status::Pass => ("✓", pal::GREEN, pass.to_string()),
        Status::Fail(e) => {
            // the line that names the problem, not the traceback around it
            let lines: Vec<&str> = e.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
            let key = lines.iter().rev().find(|l| l.contains("Error") || l.contains("error")).or(lines.first()).copied().unwrap_or("failed");
            ("✗", pal::RED, key.to_string())
        }
        Status::Skipped(why) => ("·", pal::FAINT, why.clone()),
    };
    format!("    {} {} {}", style::fg(color, icon), Style::new().bold().paint(&text::pad(label, 11)), style::dim(&msg))
}

/// The verification report: one line per check, then what Aegist will
/// and won't say about the code.
pub fn report(r: &Report, lang_name: &str, width: usize) -> Vec<String> {
    let mut out = vec![String::new()];
    out.push(status_row("syntax", &r.syntax, &format!("{lang_name}'s own tools accept it")));
    let names = if r.invented.is_empty() {
        Status::Pass
    } else {
        Status::Fail(format!("made up? {}", r.invented.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")))
    };
    out.push(status_row("names", &names, "every name it uses exists somewhere I know"));
    out.push(status_row("tests", &r.tests, "they pass"));
    let score = r.confidence.score();
    let color = if score >= 0.7 { pal::GREEN } else if score >= 0.45 { pal::YELLOW } else { pal::RED };
    let icon = if score >= 0.7 { "●" } else if score >= 0.45 { "◐" } else { "○" };
    let unsure = match r.confidence.uncertain {
        0 => "no unsure spots".to_string(),
        1 => "1 unsure spot, underlined".to_string(),
        n => format!("{n} unsure spots, underlined"),
    };
    out.push(format!("    {} {} {} {}  {}", style::fg(color, icon), Style::new().bold().paint(&text::pad("certainty", 11)),
        Style::new().fg(color).bold().paint(&format!("{:>3.0}%", score * 100.0)), bar(score, 12), style::dim(&unsure)));
    out.push(String::new());
    let wrap_w = width.saturating_sub(8).max(30);
    let (head, color, tail) = match r.verdict {
        Verdict::Verified => ("◆ Verified.", pal::GREEN, "It passes every check, and its tests ran and passed.".to_string()),
        Verdict::Checked => ("◆ Checked.", pal::CYAN, "It passes every check I can run. Nothing has run it yet, so test it before relying on it.".to_string()),
        Verdict::Unverified => ("◇ Unverified.", pal::YELLOW, format!("I couldn't check it fully: {}.", r.reasons.join("; "))),
        Verdict::Refused => ("✗ No.", pal::RED, "I can't stand behind this code:".to_string()),
    };
    let first = format!("{} {}", Style::new().fg(color).bold().paint(head), tail);
    for (i, l) in text::wrap(&text::strip_ansi(&first), wrap_w).into_iter().enumerate() {
        let l = if i == 0 { l.replacen(head, &Style::new().fg(color).bold().paint(head), 1) } else { l };
        out.push(format!("    {l}"));
    }
    if r.verdict == Verdict::Refused {
        for reason in &r.reasons {
            for (i, l) in text::wrap(reason, wrap_w - 2).into_iter().enumerate() {
                out.push(format!("      {} {}", if i == 0 { style::fg(pal::RED, "•") } else { " ".into() }, style::dim(&l)));
            }
        }
    }
    out
}

/// The working line: a spinner, a scanner bar sweeping back and forth,
/// and the shimmering status text.
pub fn spinner(tick: u64, label: &str) -> String {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let phase = (tick % 60) as f32 / 60.0;
    let glyph = Style::new().fg(pal::aurora(phase)).bold().paint(FRAMES[(tick % 10) as usize]);
    format!("{glyph} {} {}", scanner(tick, 10), shimmer(label, tick))
}

/// A little bar with a light running back and forth: `▱▱▰▰▰▱▱▱▱▱`.
pub fn scanner(tick: u64, width: usize) -> String {
    if !style::enabled() {
        return String::new();
    }
    let span = (width.max(2) - 1) as u64 * 2;
    let p = (tick / 2) % span;
    let pos = if p < width as u64 { p } else { span - p } as f32;
    let mut out = String::new();
    for i in 0..width {
        let d = (i as f32 - pos).abs();
        let lit = d < 1.6;
        let c = if lit { pal::aurora(i as f32 / width as f32).lerp(Rgb(255, 255, 255), (1.0 - d / 1.6) * 0.35) } else { pal::BORDER };
        out.push_str(&Style::new().fg(c).paint(if lit { "▰" } else { "▱" }));
    }
    out
}

/// Text with a soft highlight sweeping across it.
pub fn shimmer(label: &str, tick: u64) -> String {
    if !style::enabled() {
        return label.to_string();
    }
    let n = label.chars().count();
    let pos = (tick as f32 * 0.8) % (n as f32 + 12.0) - 6.0;
    let mut out = String::new();
    for (i, ch) in label.chars().enumerate() {
        let d = (i as f32 - pos).abs();
        let glow = (1.0 - d / 4.0).clamp(0.0, 1.0);
        let base = pal::aurora(i as f32 / n.max(2) as f32 * 0.6 + 0.2);
        let c = base.lerp(Rgb(255, 255, 255), glow * 0.7);
        out.push_str(&Style::new().fg(c).prefix());
        out.push(ch);
    }
    out.push_str(RESET);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::text::{strip_ansi, width};

    #[test]
    fn banner_boxes_and_bars_have_the_right_shape() {
        let b = banner(100);
        assert_eq!(b.len(), 6);
        assert!(b.iter().all(|l| width(l) == width(&b[0])));
        assert_eq!(banner(20).len(), 1);
        let bx = boxed(Some("title"), &["hello".into(), "a much longer line that will be cut".into()], 24, pal::BORDER);
        assert!(bx.iter().all(|l| width(l) == 24), "{:?}", bx.iter().map(|l| width(l)).collect::<Vec<_>>());
        assert!(strip_ansi(&bx[0]).starts_with("╭─ title ─"));
        assert_eq!(width(&bar(0.5, 10)), 10);
        assert_eq!(width(&sparkline(&[3.0, 2.0, 1.0], 10)), 3);
    }

    #[test]
    fn code_and_diffs_render_every_line() {
        let code = "def f(x):\n    return x + 1\n";
        let panel = code_panel(code, Some("python"), 1, 60, &[(14, 20)]);
        assert_eq!(panel.len(), 2);
        assert!(strip_ansi(&panel[1]).contains("2 │     return x + 1"));
        let d = diff("a\nb\nc\n", "a\nB\nc\nd\n", Some("python"), 60, 3);
        let plain: Vec<String> = d.iter().map(|l| strip_ansi(l)).collect();
        assert!(plain.iter().any(|l| l.contains(" - b")) && plain.iter().any(|l| l.contains(" + B")) && plain.iter().any(|l| l.contains(" + d")));
        assert_eq!(change_summary("a\nb\n", "a\nc\nd\n"), "2 additions, 1 removal");
    }
}
