//! Doing several things in a row on its own: "open notepad, type hello and
//! press enter", "fix the tests then run main.py", "open paint then draw a
//! red circle in paint".
//!
//! The whole request is planned before anything happens, and every step has
//! to be something Aegist knows how to do. If one step isn't understood, or
//! is something it won't do, nothing runs at all: it says which step and
//! why, instead of doing the parts it understood and guessing at the rest.
//! You see the plan and say go; each step then runs and is checked, and the
//! first one that fails stops the rest.

use crate::actions::{self, Intent, COMMANDS};
use crate::computer::{self, Act, Pilot};
use crate::session::Ctx;
use crate::ui::style::{self, pal, Style};
use crate::ui::{text, widgets};
use anyhow::Result;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Something on the screen.
    Screen(Act),
    /// A slash command, as typed.
    Command { name: String, args: String },
    /// A request for the session in plain words (write, fix, run, test, ...).
    Say(String),
    /// A goal in an app Aegist has learned (draw, fill, erase, ...).
    App { app: Option<String>, goal: String },
}

impl Step {
    pub fn describe(&self) -> String {
        match self {
            Step::Screen(a) => a.describe(),
            Step::Command { name, args } => format!("/{name} {args}").trim().to_string(),
            Step::Say(s) => s.clone(),
            Step::App { app: Some(app), goal } => format!("{goal} in {app}"),
            Step::App { app: None, goal } => goal.clone(),
        }
    }

    fn icon(&self) -> String {
        match self {
            Step::Screen(Act::Type(_) | Act::Key(..)) => style::fg(pal::PINK, "⌨"),
            Step::Screen(Act::Look | Act::Where | Act::Windows) => style::fg(pal::PINK, "◳"),
            Step::Screen(Act::Launch(_) | Act::Focus(_)) => style::fg(pal::PINK, "▣"),
            Step::Screen(Act::Wait(_)) => style::fg(pal::PINK, "◷"),
            Step::Screen(_) => style::fg(pal::PINK, "◉"),
            Step::Command { .. } => style::fg(pal::VIOLET, "/"),
            Step::Say(_) => style::fg(pal::CYAN, "⏵"),
            Step::App { .. } => style::fg(pal::ORANGE, "✎"),
        }
    }
}

/// Words a step can start with. Commas and "and" only split a request
/// where what follows starts with one of these, so "click 10, 20" and
/// "a function that sorts and prints" stay whole.
const VERBS: &[&str] = &[
    "click", "double", "right", "middle", "left", "tap", "move", "hover", "point", "drag", "scroll", "type", "press", "hit", "open", "launch",
    "switch", "focus", "go", "bring", "wait", "pause", "take", "look", "screenshot", "write", "create", "make", "build", "complete", "finish",
    "fix", "run", "execute", "test", "find", "search", "grep", "draw", "fill", "erase", "learn", "close", "copy", "paste", "cut", "select",
    "refresh", "reload", "zoom", "start", "stop", "show", "list", "visit", "sketch", "enter", "hold",
];

const STRONG: &[&str] = &["\n", ";", ", and then ", ", then ", " and then ", " then ", ", after that ", " after that ", " afterwards "];
const WEAK: &[&str] = &[", and ", ", ", " and ", " & "];

fn starts_with_verb(rest: &str) -> bool {
    let r = rest.trim_start().trim_start_matches("please ").trim_start_matches("also ").trim_start_matches("then ");
    if r.starts_with('/') {
        return true;
    }
    let first: String = r.chars().take_while(|c| c.is_alphanumeric() || *c == '-').collect::<String>().to_lowercase();
    let first = first.split('-').next().unwrap_or("");
    VERBS.contains(&first)
}

/// The steps in a request, as said. Nothing inside quotes is split.
pub fn split(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut i = 0;
    let lower = text.to_lowercase();
    // (lowercasing can change byte lengths outside ASCII; then only split on the obvious)
    let same_len = lower.len() == text.len();
    while i < text.len() {
        let ch = text[i..].chars().next().expect("in bounds");
        if let Some(q) = quote {
            cur.push(ch);
            if ch == q || (q == '“' && ch == '”') {
                quote = None;
            }
            i += ch.len_utf8();
            continue;
        }
        if matches!(ch, '"' | '`' | '“') || (ch == '\'' && (cur.is_empty() || cur.ends_with(' '))) {
            quote = Some(ch);
            cur.push(ch);
            i += ch.len_utf8();
            continue;
        }
        let here = if same_len { &lower[i..] } else { &text[i..] };
        if let Some(sep) = STRONG.iter().find(|s| here.starts_with(**s)) {
            if !cur.trim().is_empty() {
                pieces.push(cur.trim().to_string());
            }
            cur.clear();
            i += sep.len();
            continue;
        }
        if let Some(sep) = WEAK.iter().find(|s| here.starts_with(**s)) {
            if !cur.trim().is_empty() && starts_with_verb(&text[i + sep.len()..]) {
                pieces.push(cur.trim().to_string());
                cur.clear();
                i += sep.len();
                continue;
            }
        }
        cur.push(ch);
        i += ch.len_utf8();
    }
    if !cur.trim().is_empty() {
        pieces.push(cur.trim().to_string());
    }
    pieces
}

/// Commands a plan may not contain.
const NOT_IN_PLANS: &[&str] = &["auto", "do", "exit", "quit", "q", "clear", "cls", "train", "export", "cd"];

/// Plan `text`: every step understood, or the first one that isn't (and why).
pub fn plan(text: &str) -> Result<Vec<(String, Step)>, String> {
    let pieces = split(text);
    if pieces.is_empty() {
        return Err("do what? Say the steps, like: open notepad, type hello and press enter".into());
    }
    let mut steps: Vec<(String, Step)> = Vec::new();
    for (n, piece) in pieces.iter().enumerate() {
        let n = n + 1;
        let on_screen = steps.iter().any(|(_, s)| matches!(s, Step::Screen(Act::Launch(_) | Act::Focus(_) | Act::Click { .. })));
        let step = if let Some(rest) = piece.strip_prefix('/') {
            let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
            let name = name.to_lowercase();
            if NOT_IN_PLANS.contains(&name.as_str()) {
                return Err(format!("step {n}: /{name} can't be part of a plan"));
            }
            if !COMMANDS.iter().any(|c| c.name == name) {
                return Err(format!("step {n}: there's no /{name} command"));
            }
            Step::Command { name, args: args.trim().to_string() }
        } else if let Some(text) = typed_into_app(piece, on_screen) {
            Step::Screen(Act::Type(text))
        } else {
            match computer::parse(piece) {
                Some(Ok(act)) => Step::Screen(act),
                Some(Err(why)) => return Err(format!("step {n} ({piece:?}): {why}")),
                None => match crate::abilities::app_request(piece) {
                    Some((app, goal)) => {
                        if let Some(why) = crate::agent::risk::refuse_goal(&goal) {
                            return Err(format!("No. Step {n} ({piece:?}): {why}."));
                        }
                        Step::App { app, goal }
                    }
                    None => match actions::interpret(piece) {
                        Intent::Unknown | Intent::Question | Intent::Greeting => {
                            return Err(format!("step {n}: I don't know how to {piece:?}, so I haven't done any of it"));
                        }
                        _ => Step::Say(piece.clone()),
                    },
                },
            }
        };
        if let Step::Screen(a) = &step {
            if let Some(why) = a.refusal() {
                return Err(format!("No. Step {n} ({piece:?}): {why}."));
            }
        }
        steps.push((piece.clone(), step));
    }
    Ok(steps)
}

/// After an app is opened, "write hello" means typing it there - unless it
/// names a file or a language (then it's code).
fn typed_into_app(piece: &str, on_screen: bool) -> Option<String> {
    if !on_screen {
        return None;
    }
    let lower = piece.to_lowercase();
    let rest = ["write ", "enter "].iter().find_map(|p| lower.starts_with(p).then(|| piece[p.len()..].trim()))?;
    let is_code = rest.split_whitespace().any(|w| w.contains('.') && crate::lang::is_code(std::path::Path::new(w.trim_matches(['"', '\'']))))
        || actions::language_in(&lower).is_some()
        || ["function", "program", "script", "class", "code"].iter().any(|w| lower.split_whitespace().any(|x| x == *w));
    if is_code {
        return None;
    }
    let text = rest.trim_start_matches("the text ").trim();
    let text = text.strip_suffix(" there").or_else(|| text.strip_suffix(" in it")).unwrap_or(text);
    let unquoted = ['"', '\'', '`'].iter().find_map(|q| text.strip_prefix(*q).and_then(|t| t.strip_suffix(*q)));
    Some(unquoted.unwrap_or(text).to_string()).filter(|t| !t.is_empty())
}

/// What a plain-words request is, as far as doing things goes.
pub enum Route {
    /// Several steps, all understood.
    Plan,
    /// One thing on the screen.
    One(Act),
    /// Meant for the screen, but it can't be done as said.
    CantDo(String),
    /// Not for this: coding requests, small talk, ...
    NotMine,
}

pub fn route(text: &str) -> Route {
    let pieces = split(text);
    // a coding request's description can say "then" and "and" all it likes
    let first_is_code = pieces.first().is_some_and(|p| matches!(actions::interpret(p), Intent::Write { .. } | Intent::Complete(_))
        && computer::parse(p).is_none());
    if pieces.len() >= 2 && !first_is_code {
        return match plan(text) {
            Ok(_) => Route::Plan,
            Err(why) if pieces.iter().any(|p| computer::parse(p).is_some()) => Route::CantDo(why),
            Err(_) => Route::NotMine,
        };
    }
    // bare words that mean something else in the session ("copy" the code)
    if matches!(text.trim().to_lowercase().as_str(), "copy" | "copy it" | "copy that" | "paste" | "cut" | "back" | "forward" | "refresh" | "reload") {
        return Route::NotMine;
    }
    match computer::parse(text) {
        Some(Ok(a)) => Route::One(a),
        Some(Err(why)) => Route::CantDo(why),
        None => Route::NotMine,
    }
}

/// "I don't know", said plainly, with the reason.
pub fn dont_know(ctx: &Ctx, why: &str) {
    let w = ctx.width().saturating_sub(6);
    let (head, body) = if why.starts_with("No.") { ("No.", why.trim_start_matches("No.").trim()) } else { ("I don't know how to do that.", why) };
    let color = if head == "No." { pal::RED } else { pal::YELLOW };
    let mut lines = vec![String::new(), format!("  {}", Style::new().fg(color).bold().paint(head))];
    for l in text::wrap(body, w) {
        lines.push(format!("  {}", style::dim(&l)));
    }
    ctx.print(lines);
}

/// Plan `text`, show the plan, and - once you say go - do it, step by step.
pub fn run(ctx: &Ctx, text: &str) -> Result<()> {
    let steps = match plan(text) {
        Ok(s) => s,
        Err(why) => {
            dont_know(ctx, &why);
            return Ok(());
        }
    };
    let w = ctx.width();
    let n = steps.len();
    let mut card = Vec::new();
    let mut asks = Vec::new();
    for (i, (_, step)) in steps.iter().enumerate() {
        let mut line = format!("{} {} {}", style::gradient_styled(&format!("{:>2}", i + 1), i as f32 / n.max(2) as f32, true), step.icon(),
                               text::truncate(&step.describe(), w.saturating_sub(20)));
        if let Step::Screen(a) = step {
            if let Some(why) = a.needs_ok() {
                line.push_str(&format!("  {}", style::fg(pal::YELLOW, "⚠")));
                asks.push(format!("step {}: {why}", i + 1));
            }
        }
        card.push(line);
    }
    for a in &asks {
        card.push(format!("   {} {}", style::fg(pal::YELLOW, "⚠"), style::dim(a)));
    }
    let uses_screen = steps.iter().any(|(_, s)| matches!(s, Step::Screen(a) if !matches!(a, Act::Wait(_))) || matches!(s, Step::App { .. }));
    if uses_screen {
        card.push(style::faint("move the mouse or press esc to take over at any time"));
    }
    let mut lines = vec![String::new()];
    let title = format!("{} {}", style::gradient_styled(&style::spaced("plan"), 0.1, true), style::faint(&format!("// {n} step{}", if n == 1 { "" } else { "s" })));
    lines.extend(widgets::boxed(Some(&title), &card, w.saturating_sub(4).min(96), pal::BORDER).into_iter().map(|l| format!("  {l}")));
    ctx.print(lines);

    let first_time = uses_screen && !ctx.shared.lock().expect("lock").desktop_ok;
    let must_ask = ctx.settings.agent.confirm || !asks.is_empty() || first_time;
    if must_ask {
        let q = if first_time { format!("Run these {n} steps? (they use your mouse and keyboard)") } else { format!("Run these {n} steps?") };
        if ctx.ask(&q, &[('y', "go"), ('n', "no")]) != 'y' {
            ctx.line(format!("  {}", style::dim("Okay - nothing was done.")));
            return Ok(());
        }
    }
    if uses_screen {
        ctx.shared.lock().expect("lock").desktop_ok = true;
    }

    let started = Instant::now();
    let mut pilot: Option<Pilot> = None;
    for (i, (said, step)) in steps.iter().enumerate() {
        if ctx.cancelled() {
            stopped(ctx, i, n, "you asked me to stop");
            return Ok(());
        }
        ctx.print(vec![String::new(), format!("  {} {} {}", style::gradient_styled(&format!("▸ {}/{n}", i + 1), i as f32 / n.max(2) as f32, true),
                                              step.icon(), Style::new().bold().paint(&step.describe()))]);
        ctx.status(format!("step {}/{n}: {}", i + 1, step.describe()));
        let outcome: Result<bool> = (|| match step {
            Step::Screen(act) => {
                if pilot.is_none() {
                    pilot = Some(computer::open_pilot(ctx, &act.describe())?);
                }
                let done = pilot.as_mut().expect("just opened").perform(act)?;
                computer::show(ctx, &done);
                Ok(done.ok())
            }
            Step::Command { name, args } => {
                ctx.shared.lock().expect("lock").last_exit_ok = None;
                crate::app::dispatch(ctx, name, args)?;
                Ok(ctx.shared.lock().expect("lock").last_exit_ok != Some(false))
            }
            Step::Say(s) => {
                ctx.shared.lock().expect("lock").last_exit_ok = None;
                crate::app::dispatch(ctx, "", s)?;
                Ok(ctx.shared.lock().expect("lock").last_exit_ok != Some(false))
            }
            Step::App { app, goal } => {
                let app = match app {
                    Some(a) => a.clone(),
                    None => match computer_front_app(&steps[..i]) {
                        Some(a) => a,
                        None => anyhow::bail!("which app? Say it with the app, like \"draw a red circle in paint\""),
                    },
                };
                // the paint agent opens its own window by name
                pilot = None;
                crate::abilities::run_goal(ctx, &app, goal, true)?;
                Ok(true)
            }
        })();
        match outcome {
            Ok(true) => {}
            Ok(false) => {
                stopped(ctx, i, n, &format!("step {} ({said:?}) didn't work", i + 1));
                return Ok(());
            }
            Err(e) => {
                ctx.line(format!("    {} {}", style::fg(pal::RED, "✗"), e));
                stopped(ctx, i, n, &format!("step {} ({said:?}) failed", i + 1));
                return Ok(());
            }
        }
    }
    let actions = pilot.as_ref().map_or(0, |p| p.actions);
    ctx.print(vec![String::new(), format!("  {} {}", style::fg(pal::GREEN, "✓"),
        style::dim(&format!("all {n} step{} done in {:.1}s{}", if n == 1 { "" } else { "s" }, started.elapsed().as_secs_f64(),
                            if actions > 0 { format!(" · {actions} mouse and keyboard actions") } else { String::new() })))]);
    Ok(())
}

/// The app an earlier step opened, switched to or drew in, for "draw ..."
/// without "in <app>".
fn computer_front_app(before: &[(String, Step)]) -> Option<String> {
    before.iter().rev().find_map(|(_, s)| match s {
        Step::Screen(Act::Launch(a) | Act::Focus(a)) => Some(a.clone()),
        Step::App { app: Some(a), .. } => Some(a.clone()),
        _ => None,
    })
}

fn stopped(ctx: &Ctx, at: usize, n: usize, why: &str) {
    let rest = if at + 1 < n { format!(" · step{} {}{} not run", if n - at - 1 == 1 { "" } else { "s" }, at + 2,
                                       if at + 2 < n { format!("-{n}") } else { String::new() }) } else { String::new() };
    ctx.print(vec![String::new(), format!("  {} {}{}", style::fg(pal::YELLOW, "■"), Style::new().bold().paint(&format!("Stopped: {why}")),
                                          style::faint(&rest))]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computer::{Button, Spot};

    #[test]
    fn requests_split_into_steps_only_where_a_step_starts() {
        assert_eq!(split("open notepad, type hello and press enter"), vec!["open notepad", "type hello", "press enter"]);
        assert_eq!(split("click 10, 20 then type \"a, b and then c\""), vec!["click 10, 20", "type \"a, b and then c\""]);
        assert_eq!(split("fix the tests; run main.py"), vec!["fix the tests", "run main.py"]);
        assert_eq!(split("write a function that sorts and prints a list"), vec!["write a function that sorts and prints a list"]);
        assert_eq!(split("open firefox\ngo to github.com\n\nscroll down"), vec!["open firefox", "go to github.com", "scroll down"]);
    }

    #[test]
    fn plans_are_all_or_nothing() {
        let p = plan("open notepad, write hello world and press enter").unwrap();
        assert_eq!(p.iter().map(|s| s.1.clone()).collect::<Vec<_>>(), vec![
            Step::Screen(Act::Launch("notepad".into())),
            Step::Screen(Act::Type("hello world".into())),
            Step::Screen(Act::Key("enter".into(), 1)),
        ]);
        let p = plan("click 640 400 then run main.py then /test").unwrap();
        assert_eq!(p[0].1, Step::Screen(Act::Click { at: Some(Spot::Px(640, 400)), button: Button::Left, times: 1 }));
        assert_eq!(p[1].1, Step::Say("run main.py".into()));
        assert_eq!(p[2].1, Step::Command { name: "test".into(), args: String::new() });
        let p = plan("open paint then draw a red circle in paint").unwrap();
        assert!(matches!(&p[1].1, Step::App { app: Some(a), .. } if a == "paint"));
        // one step it can't do: none of it runs
        let e = plan("open notepad, then do my taxes, then press enter").unwrap_err();
        assert!(e.contains("step 2") && e.contains("do my taxes"), "{e}");
        let e = plan("open notepad then click the save button").unwrap_err();
        assert!(e.contains("step 2") && e.contains("I don't know where"), "{e}");
        assert!(plan("open my bank then type 1234").unwrap_err().starts_with("No."));
        assert!(plan("open notepad then /exit").unwrap_err().contains("/exit"));
        assert!(plan("press ctrl+alt+delete").unwrap_err().starts_with("No."));
        // after opening an app, "write" is typing; a file or a language means code
        let p = plan("switch to notepad and then write a python function in utils.py").unwrap();
        assert!(matches!(&p[1].1, Step::Say(_)), "{:?}", p[1].1);
    }

    #[test]
    fn plain_requests_are_routed() {
        assert!(matches!(route("open notepad and type hi"), Route::Plan));
        assert!(matches!(route("click 5 5"), Route::One(_)));
        assert!(matches!(route("click the OK button"), Route::CantDo(_)));
        assert!(matches!(route("write a function that reads a file, then sorts the lines"), Route::NotMine));
        assert!(matches!(route("fix `pytest -q`"), Route::NotMine));
        assert!(matches!(route("hello"), Route::NotMine));
        assert!(matches!(route("fix the tests and run main.py"), Route::Plan));
    }
}
