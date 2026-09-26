//! Aegist as a mouse and keyboard on your screen.
//!
//! Say what to do in plain words - "click 640 400", "right click", "scroll
//! down 3", "type hello", "press ctrl+c", "open notepad", "switch to
//! firefox", "take a screenshot" - and it does exactly that. It never picks
//! a place to click on its own: every action is one you asked for.
//!
//! It can't read the text on your screen, so "click the OK button" gets an
//! honest "I don't know where that is" and a way to tell it (coordinates,
//! which `/screen` shows), never a click somewhere that looks likely.
//!
//! Safety, on every action:
//!   - it asks once per session before touching your desktop at all;
//!   - keys that close, save, print or delete ask first, and a few it
//!     never presses; launching things that involve passwords, money or
//!     deleting is refused (`agent::risk`);
//!   - moving the mouse yourself takes it back at once, esc stops it, and
//!     a run stops after `agent.max_actions` actions.

use crate::agent::desktop::{self, Desktop, Window};
use crate::agent::risk::{self, Stopped};
use crate::agent::vision::Image;
use crate::config::AgentSettings;
use crate::session::Ctx;
use crate::ui::style::{self, pal, Style};
use crate::ui::text;
use anyhow::{bail, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Left,
    Middle,
    Right,
}

impl Button {
    fn code(self) -> u8 {
        match self {
            Button::Left => 1,
            Button::Middle => 2,
            Button::Right => 3,
        }
    }
}

/// A place on the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spot {
    /// Pixels from the screen's top-left corner.
    Px(i32, i32),
    /// Fractions of the screen's width and height (0-1).
    Frac(f32, f32),
    /// The middle of the window in front.
    WindowCenter,
}

impl Spot {
    fn describe(&self) -> String {
        match self {
            Spot::Px(x, y) => format!("({x}, {y})"),
            Spot::Frac(x, y) => format!("({:.0}%, {:.0}%) of the screen", x * 100.0, y * 100.0),
            Spot::WindowCenter => "the middle of the window in front".into(),
        }
    }
}

/// One thing to do on the screen.
#[derive(Clone, Debug, PartialEq)]
pub enum Act {
    Move(Spot),
    /// No spot: wherever the pointer is.
    Click { at: Option<Spot>, button: Button, times: u8 },
    Drag { from: Spot, to: Spot },
    /// Positive: up.
    Scroll(i32),
    Type(String),
    /// A key or combination, pressed this many times.
    Key(String, u8),
    /// Bring the window whose title matches to the front.
    Focus(String),
    /// Start an app, or open a web address.
    Launch(String),
    Wait(u64),
    /// Take a screenshot and show it.
    Look,
    /// Where the pointer is, and which window is in front.
    Where,
    Windows,
}

impl Act {
    pub fn describe(&self) -> String {
        match self {
            Act::Move(s) => format!("move the mouse to {}", s.describe()),
            Act::Click { at, button, times } => {
                let kind = match (button, times) {
                    (Button::Left, 1) => "click".to_string(),
                    (Button::Left, 2) => "double-click".to_string(),
                    (Button::Left, n) => format!("click {n} times"),
                    (Button::Right, _) => "right-click".to_string(),
                    (Button::Middle, _) => "middle-click".to_string(),
                };
                match at {
                    Some(s) => format!("{kind} at {}", s.describe()),
                    None => format!("{kind} where the pointer is"),
                }
            }
            Act::Drag { from, to } => format!("drag from {} to {}", from.describe(), to.describe()),
            Act::Scroll(n) => format!("scroll {} {} notch{}", if *n > 0 { "up" } else { "down" }, n.abs(), if n.abs() == 1 { "" } else { "es" }),
            Act::Type(t) => format!("type {:?}", text::truncate(t, 60)),
            Act::Key(k, 1) => format!("press {k}"),
            Act::Key(k, n) => format!("press {k} {n} times"),
            Act::Focus(w) => format!("switch to the {w:?} window"),
            Act::Launch(t) if is_url(t) => format!("open {} in the browser", url_of(t)),
            Act::Launch(t) => format!("open {t}"),
            Act::Wait(ms) if ms % 1000 == 0 => format!("wait {} second{}", ms / 1000, if *ms == 1000 { "" } else { "s" }),
            Act::Wait(ms) => format!("wait {ms} ms"),
            Act::Look => "take a screenshot".into(),
            Act::Where => "say where the pointer is".into(),
            Act::Windows => "list the open windows".into(),
        }
    }

    /// Only looks: nothing on the screen changes.
    pub fn only_looks(&self) -> bool {
        matches!(self, Act::Look | Act::Where | Act::Windows | Act::Wait(_))
    }

    /// Why this is never done, if it isn't.
    pub fn refusal(&self) -> Option<String> {
        match self {
            Act::Key(k, _) => match key_rule(k) {
                KeyRule::Refuse(why) => Some(format!("{k} {why}")),
                _ => None,
            },
            Act::Launch(t) | Act::Focus(t) => risk::refuse_goal(t),
            _ => None,
        }
    }

    /// What to ask you about before doing this, if anything.
    pub fn needs_ok(&self) -> Option<String> {
        match self {
            Act::Key(k, _) => match key_rule(k) {
                KeyRule::Confirm(why) => Some(format!("{k} {why}")),
                _ => None,
            },
            Act::Type(t) if t.contains('\n') => Some("the text has line breaks, which press enter (it may send or run something)".into()),
            _ => None,
        }
    }
}

/// How a key you asked for is treated.
#[derive(Debug, PartialEq)]
pub enum KeyRule {
    Fine,
    Confirm(String),
    Refuse(String),
}

pub fn key_rule(combo: &str) -> KeyRule {
    let c = combo.to_lowercase().replace(' ', "");
    if matches!(c.as_str(), "ctrl+alt+delete" | "ctrl+alt+del") {
        return KeyRule::Refuse("is the system's security screen".into());
    }
    const ASK: &[(&str, &str)] = &[
        ("alt+f4", "closes the app in front"),
        ("ctrl+w", "closes a window or tab"),
        ("ctrl+q", "quits the app"),
        ("ctrl+s", "saves over a file"),
        ("ctrl+shift+s", "saves a file"),
        ("ctrl+p", "prints"),
        ("delete", "deletes what's selected"),
        ("del", "deletes what's selected"),
        ("shift+delete", "deletes for good, skipping the bin"),
        ("ctrl+shift+delete", "clears browsing data"),
        ("ctrl+shift+esc", "opens the task manager"),
        ("super+l", "locks the computer"),
        ("win+l", "locks the computer"),
    ];
    match ASK.iter().find(|(k, _)| *k == c) {
        Some((_, why)) => KeyRule::Confirm(why.to_string()),
        None => KeyRule::Fine,
    }
}

// ---------------------------------------------------------------- words

/// `t` without `prefix` (any case), if it starts with it at a word boundary.
fn eat<'a>(t: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    for p in prefixes {
        if t.len() >= p.len() && t.is_char_boundary(p.len()) && t[..p.len()].eq_ignore_ascii_case(p) {
            let rest = &t[p.len()..];
            if rest.is_empty() || rest.starts_with(|c: char| !c.is_alphanumeric()) || p.ends_with(' ') {
                return Some(rest.trim_start());
            }
        }
    }
    None
}

/// Leading words that don't change what's asked.
fn strip_filler(mut t: &str) -> &str {
    const FILLER: &[&str] = &["please ", "now ", "then ", "and ", "also ", "can you ", "could you ", "would you ", "aegist, ", "aegist ",
                              "go ahead and ", "just ", "next ", "finally ", "first ", "after that ", "afterwards "];
    loop {
        let t2 = t.trim_start();
        match eat(t2, FILLER) {
            Some(rest) => t = rest,
            None => return t2.trim_end_matches(['.', '!']).trim_end_matches(" please").trim(),
        }
    }
}

pub fn is_url(t: &str) -> bool {
    let t = t.trim();
    if t.starts_with("http://") || t.starts_with("https://") {
        return true;
    }
    let host = t.trim_start_matches("www.").split('/').next().unwrap_or("");
    let re = regex::Regex::new(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.(com|org|net|io|dev|ai|app|edu|gov|co|uk|de|fr|jp|me|tv|gg|info|xyz|ca|au|us|in|ly)$")
        .expect("valid regex");
    !t.contains(' ') && re.is_match(host)
}

fn url_of(t: &str) -> String {
    let t = t.trim();
    if t.starts_with("http://") || t.starts_with("https://") { t.to_string() } else { format!("https://{t}") }
}

/// Every place named in `text`: number pairs ("640 400", "640,400",
/// "50% 20%"), or one named place ("the center", "top left").
pub fn spots(text: &str) -> Vec<Spot> {
    let re = regex::Regex::new(r"(-?\d+(?:\.\d+)?)\s*(%?)\s*(?:,|x|×|\s)\s*(-?\d+(?:\.\d+)?)\s*(%?)").expect("valid regex");
    let found: Vec<Spot> = re
        .captures_iter(text)
        .filter_map(|c| {
            let (a, b): (f32, f32) = (c[1].parse().ok()?, c[3].parse().ok()?);
            Some(if !c[2].is_empty() || !c[4].is_empty() { Spot::Frac(a / 100.0, b / 100.0) } else { Spot::Px(a.round() as i32, b.round() as i32) })
        })
        .collect();
    if !found.is_empty() {
        return found;
    }
    let t = format!(" {} ", text.to_lowercase().replace(['.', ','], " "));
    let has = |w: &str| t.contains(&format!(" {w} "));
    if has("window") && (has("middle") || has("center") || has("centre") || t.trim() == "window" || has("the window")) {
        return vec![Spot::WindowCenter];
    }
    let (top, bottom, left, right) = (has("top"), has("bottom"), has("left"), has("right"));
    let x = if left { Some(0.05) } else if right { Some(0.95) } else { None };
    let y = if top { Some(0.05) } else if bottom { Some(0.95) } else { None };
    match (x, y) {
        (None, None) if has("middle") || has("center") || has("centre") => vec![Spot::Frac(0.5, 0.5)],
        (None, None) => Vec::new(),
        (x, y) => vec![Spot::Frac(x.unwrap_or(0.5), y.unwrap_or(0.5))],
    }
}

/// "ctrl c", "Control-Shift-T", "the enter key" -> "ctrl+c", "ctrl+shift+t", "enter".
pub fn normalize_combo(text: &str) -> String {
    let t = text.to_lowercase();
    let t = t.replace("page down", "pagedown").replace("page up", "pageup").replace("caps lock", "capslock");
    let parts: Vec<String> = t
        .split(|c: char| c.is_whitespace() || c == '+' || c == '-')
        .filter(|w| !w.is_empty() && !matches!(*w, "the" | "key" | "keys" | "button" | "and" | "on" | "keyboard"))
        .map(|w| match w {
            "control" | "ctl" | "strg" => "ctrl".to_string(),
            "return" => "enter".to_string(),
            "esc" => "escape".to_string(),
            "windows" | "win" | "cmd" | "command" | "meta" => "super".to_string(),
            "option" => "alt".to_string(),
            "spacebar" => "space".to_string(),
            "arrow" => String::new(),
            "del" => "delete".to_string(),
            "pgdn" => "pagedown".to_string(),
            "pgup" => "pageup".to_string(),
            other => other.to_string(),
        })
        .filter(|w| !w.is_empty())
        .collect();
    parts.join("+")
}

/// Is `text` something to do on the screen? None: no (it's for something
/// else). Some(Err): yes, but it can't be done as said (the message says why).
pub fn parse(text: &str) -> Option<Result<Act, String>> {
    let t = strip_filler(text);
    let lower = t.to_lowercase();
    if lower.is_empty() {
        return None;
    }
    // everyday shortcuts, said as words
    const SHORTCUTS: &[(&[&str], &str)] = &[
        (&["open a new tab", "new tab", "open new tab"], "ctrl+t"),
        (&["open a new window", "new window", "open new window"], "ctrl+n"),
        (&["close the tab", "close tab", "close this tab"], "ctrl+w"),
        (&["close the window", "close window", "close this window", "close the app", "close it", "close this app"], "alt+f4"),
        (&["go back", "back"], "alt+left"),
        (&["go forward", "forward"], "alt+right"),
        (&["refresh", "reload", "refresh the page", "reload the page"], "f5"),
        (&["copy", "copy it", "copy that", "copy this"], "ctrl+c"),
        (&["paste", "paste it", "paste that"], "ctrl+v"),
        (&["cut", "cut it", "cut that"], "ctrl+x"),
        (&["select all", "select everything"], "ctrl+a"),
        (&["zoom in"], "ctrl+plus"),
        (&["zoom out"], "ctrl+minus"),
        (&["switch windows", "switch window", "alt tab"], "alt+tab"),
    ];
    if let Some((_, k)) = SHORTCUTS.iter().find(|(said, _)| said.contains(&lower.as_str())) {
        return Some(Ok(Act::Key(k.to_string(), 1)));
    }
    const LOOK: &[&str] = &["screenshot", "take a screenshot", "take screenshot", "look at the screen", "look at my screen", "show me the screen",
                            "show the screen", "show my screen", "show me my screen", "what's on the screen", "what's on my screen",
                            "what is on the screen", "what is on my screen", "see the screen", "look", "look around"];
    if LOOK.contains(&lower.as_str()) {
        return Some(Ok(Act::Look));
    }
    const WHERE: &[&str] = &["where is the mouse", "where's the mouse", "where is the cursor", "where's the cursor", "where is the pointer",
                             "where's the pointer", "mouse position", "cursor position", "where am i pointing"];
    if WHERE.contains(&lower.as_str()) {
        return Some(Ok(Act::Where));
    }
    const WINDOWS: &[&str] = &["list windows", "list the windows", "list all windows", "list open windows", "show windows", "show the windows",
                               "show open windows", "what windows are open", "which windows are open", "open windows", "windows"];
    if WINDOWS.contains(&lower.as_str()) {
        return Some(Ok(Act::Windows));
    }

    // clicks
    let click_leads: &[(&[&str], Button, u8)] = &[
        (&["double click", "double-click", "doubleclick", "dbl click", "double tap"], Button::Left, 2),
        (&["triple click", "triple-click"], Button::Left, 3),
        (&["right click", "right-click", "rightclick"], Button::Right, 1),
        (&["middle click", "middle-click"], Button::Middle, 1),
        (&["left click", "left-click", "click", "tap"], Button::Left, 1),
    ];
    for (leads, button, times) in click_leads {
        if let Some(rest) = eat(t, leads) {
            let (mut button, mut times, mut rest) = (*button, *times, rest);
            // "/click right 10 20", "click twice"
            if let Some(r) = eat(rest, &["right"]).filter(|r| r.is_empty() || r.starts_with(|c: char| c.is_ascii_digit() || c == '-')) {
                button = Button::Right;
                rest = r;
            } else if let Some(r) = eat(rest, &["middle"]).filter(|r| r.is_empty() || r.starts_with(|c: char| c.is_ascii_digit())) {
                button = Button::Middle;
                rest = r;
            } else if let Some(r) = eat(rest, &["double", "twice"]) {
                times = 2;
                rest = r;
            }
            let rest = rest.trim_end_matches(" twice");
            if lower.ends_with(" twice") {
                times = 2;
            }
            let place = eat(rest, &["on the ", "on ", "at the ", "at ", "in the ", "the "]).unwrap_or(rest).trim();
            if place.is_empty() || matches!(place, "here" | "there" | "it" | "where the mouse is" | "where the pointer is" | "where the cursor is") {
                return Some(Ok(Act::Click { at: None, button, times }));
            }
            return Some(match spots(place).first() {
                Some(&s) => Ok(Act::Click { at: Some(s), button, times }),
                None => Err(cant_find(place)),
            });
        }
    }

    // the pointer
    let mouse_word = ["mouse", "cursor", "pointer"].iter().any(|w| lower.contains(w));
    if let Some(rest) = eat(t, &["move", "hover", "point", "put", "place"]) {
        let rest = rest.to_lowercase();
        let place = rest.trim_start_matches("the ").trim_start_matches("mouse").trim_start_matches("cursor").trim_start_matches("pointer").trim();
        let place = place.trim_start_matches("over ").trim_start_matches("at ").trim_start_matches("to ").trim_start_matches("on ").trim();
        match spots(place).first() {
            Some(&s) if mouse_word || t.to_lowercase().starts_with("move to") || t.to_lowercase().starts_with("hover") => return Some(Ok(Act::Move(s))),
            None if mouse_word || t.to_lowercase().starts_with("hover") => return Some(Err(cant_find(place))),
            _ => {}
        }
    }
    if let Some(rest) = eat(t, &["mouse to", "cursor to", "pointer to"]) {
        return Some(match spots(rest).first() {
            Some(&s) => Ok(Act::Move(s)),
            None => Err(cant_find(rest)),
        });
    }
    if let Some(rest) = eat(t, &["drag"]) {
        let found: Vec<Spot> = spots(rest).into_iter().filter(|s| matches!(s, Spot::Px(..) | Spot::Frac(..))).collect();
        return Some(match found.as_slice() {
            [from, to, ..] if rest.chars().filter(|c| c.is_ascii_digit()).count() >= 4 => Ok(Act::Drag { from: *from, to: *to }),
            _ => Err("drag from where to where? Give both places: drag 100 200 to 400 300".into()),
        });
    }
    if let Some(rest) = eat(t, &["scroll"]) {
        let r = rest.to_lowercase();
        let n: i32 = r.split_whitespace().find_map(|w| w.parse().ok()).unwrap_or(3).clamp(1, 100);
        let up = r.contains("up") || r.contains("top");
        let n = if r.contains("top") || r.contains("bottom") || r.contains("all the way") { 50 } else { n };
        return Some(Ok(Act::Scroll(if up { n } else { -n })));
    }

    // the keyboard
    if let Some(rest) = eat(t, &["type out", "type in", "type", "enter the text", "write out"]) {
        let text = quoted(rest).unwrap_or_else(|| rest.to_string());
        return Some(if text.is_empty() { Err("type what? Say it after \"type\": type hello there".into()) } else { Ok(Act::Type(text)) });
    }
    if let Some(rest) = eat(t, &["press", "hit", "hold"]) {
        let r = rest.to_lowercase();
        let (keys, times) = repeats(&r);
        let combo = normalize_combo(keys);
        return Some(if combo.is_empty() { Err("press which key? For example: press enter, press ctrl+c".into()) } else { Ok(Act::Key(combo, times)) });
    }

    // apps and windows
    if let Some(rest) = eat(t, &["switch to", "focus on", "focus", "bring up", "activate", "go to the", "go to"]) {
        let name = rest.trim_start_matches("the ").trim_end_matches(" window").trim_end_matches(" app").trim();
        if name.is_empty() {
            return Some(Err("switch to which window? /windows lists them".into()));
        }
        return Some(Ok(if is_url(name) { Act::Launch(name.to_string()) } else { Act::Focus(name.to_string()) }));
    }
    if let Some(rest) = eat(t, &["launch", "open up", "open", "start up", "fire up", "run the app", "visit", "browse to"]) {
        let name = rest.trim_start_matches("the ").trim_start_matches("a ").trim();
        let name = name.trim_end_matches(" app").trim_end_matches(" application").trim_end_matches(" program").trim();
        let starts_code_file = crate::lang::is_code(std::path::Path::new(name)) && name.contains('.') && !is_url(name);
        let first = t.split_whitespace().next().unwrap_or("").to_lowercase();
        if name.is_empty() || starts_code_file || (first == "open" && std::path::Path::new(name).exists()) {
            return None; // a file: that's /open's job
        }
        return Some(Ok(Act::Launch(name.to_string())));
    }
    if let Some(rest) = eat(t, &["wait", "pause", "sleep", "hold on"]) {
        let r = rest.to_lowercase();
        let n: f64 = r.split_whitespace().find_map(|w| w.trim_end_matches(['s', 'm']).parse().ok()).unwrap_or(1.0);
        let ms = if r.contains("ms") || r.contains("milli") { n } else if r.contains("min") { n * 60_000.0 } else { n * 1000.0 };
        return Some(Ok(Act::Wait((ms as u64).clamp(1, 120_000))));
    }
    None
}

fn cant_find(what: &str) -> String {
    format!("I don't know where \"{}\" is: I can't read what's on your screen, so I won't guess and click somewhere that looks likely. \
             Tell me where instead, like \"click 640 400\" or \"click 50% 20%\" - /screen shows the screen with its coordinates.", what.trim())
}

/// Text in quotes, if there is some.
fn quoted(t: &str) -> Option<String> {
    for q in ['"', '\'', '`', '“'] {
        let close = if q == '“' { '”' } else { q };
        if let Some(a) = t.find(q) {
            if let Some(b) = t[a + q.len_utf8()..].find(close) {
                return Some(t[a + q.len_utf8()..a + q.len_utf8() + b].to_string());
            }
        }
    }
    None
}

/// "enter twice" -> ("enter", 2); "tab 3 times" -> ("tab", 3).
fn repeats(t: &str) -> (&str, u8) {
    if let Some(k) = t.strip_suffix(" twice") {
        return (k, 2);
    }
    if let Some(k) = t.strip_suffix(" times").or_else(|| t.strip_suffix(" time")) {
        if let Some((keys, n)) = k.rsplit_once(' ') {
            if let Ok(n) = n.parse::<u8>() {
                return (keys, n.clamp(1, 50));
            }
        }
    }
    (t, 1)
}

// ---------------------------------------------------------------- apps

/// Apps by everyday name: (names, Windows, macOS, Linux programs to try).
const APPS: &[(&[&str], &str, &str, &[&str])] = &[
    (&["paint", "ms paint", "mspaint", "drawing app"], "mspaint", "Preview", &["pinta", "kolourpaint", "mtpaint", "gimp", "drawing"]),
    (&["calculator", "calc"], "calc", "Calculator", &["gnome-calculator", "kcalc", "galculator", "mate-calc", "qalculate-gtk", "xcalc"]),
    (&["notepad", "text editor", "editor", "notes"], "notepad", "TextEdit",
     &["gedit", "gnome-text-editor", "mousepad", "kate", "xed", "pluma", "leafpad", "featherpad", "kwrite"]),
    (&["files", "file explorer", "explorer", "file manager", "finder", "my files"], "explorer", "Finder",
     &["nautilus", "dolphin", "thunar", "pcmanfm", "nemo", "caja"]),
    (&["terminal", "command prompt", "cmd", "console", "shell"], "cmd", "Terminal",
     &["gnome-terminal", "konsole", "xfce4-terminal", "mate-terminal", "alacritty", "kitty", "xterm"]),
    (&["browser", "web browser", "the internet", "internet"], "msedge", "Safari", &["firefox", "chromium", "google-chrome", "chromium-browser", "brave-browser"]),
    (&["chrome", "google chrome"], "chrome", "Google Chrome", &["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]),
    (&["firefox"], "firefox", "Firefox", &["firefox"]),
    (&["edge", "microsoft edge"], "msedge", "Microsoft Edge", &["microsoft-edge", "microsoft-edge-stable"]),
    (&["vs code", "vscode", "visual studio code", "code editor"], "code", "Visual Studio Code", &["code", "codium"]),
    (&["settings", "control panel", "system settings"], "ms-settings:", "System Settings", &["gnome-control-center", "systemsettings", "xfce4-settings-manager"]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Windows,
    Mac,
    Linux,
}

impl Os {
    pub fn this() -> Os {
        if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "macos") {
            Os::Mac
        } else {
            Os::Linux
        }
    }
}

/// The program and arguments that open `target` (an app's everyday name, a
/// program, or a web address). `installed` says whether a program exists.
pub fn launch_argv(target: &str, os: Os, installed: &dyn Fn(&str) -> bool) -> Result<Vec<String>, String> {
    let t = target.trim();
    let lower = t.to_lowercase();
    let known = APPS.iter().find(|(names, ..)| names.contains(&lower.as_str()));
    let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    if is_url(t) {
        let url = url_of(t);
        return Ok(match os {
            Os::Windows => v(&["cmd", "/C", "start", "", &url]),
            Os::Mac => v(&["open", &url]),
            Os::Linux if installed("xdg-open") => v(&["xdg-open", &url]),
            Os::Linux => return Err("nothing here opens web pages (xdg-open isn't installed)".into()),
        });
    }
    match os {
        Os::Windows => Ok(v(&["cmd", "/C", "start", "", known.map_or(t, |k| k.1)])),
        Os::Mac => Ok(v(&["open", "-a", known.map_or(t, |k| k.2)])),
        Os::Linux => {
            let candidates: Vec<&str> = match known {
                Some(k) => k.3.to_vec(),
                None => vec![t],
            };
            match candidates.iter().find(|c| installed(c)) {
                Some(p) => Ok(v(&[p])),
                None if known.is_some() => Err(format!("I couldn't find a {lower} app on this computer (I looked for {})", candidates.join(", "))),
                None => Err(format!("I couldn't find an app called {t:?} on this computer")),
            }
        }
    }
}

/// Start `target` and let it run on its own.
pub fn system_launch(target: &str) -> Result<()> {
    let argv = launch_argv(target, Os::this(), &|p| crate::proc::which(p).is_some()).map_err(|e| anyhow::anyhow!(e))?;
    let mut cmd = std::process::Command::new(&argv[0]);
    cmd.args(&argv[1..]).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flashes up
    }
    let mut child = cmd.spawn().map_err(|e| anyhow::anyhow!("couldn't start {}: {e}", argv[0]))?;
    // reap it when it ends, so it doesn't linger
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// The window that best matches `name`: its title has every word in it,
/// or it belongs to that program ("xterm", "chrome"). Title matches first,
/// then the shortest title (the most specific).
pub fn best_window<'a>(windows: &'a [Window], name: &str) -> Option<&'a Window> {
    let q = name.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    let squashed: String = q.chars().filter(|c| c.is_alphanumeric()).collect();
    windows
        .iter()
        .filter_map(|w| {
            let title = w.title.to_lowercase();
            if words.iter().all(|x| title.contains(x)) {
                return Some((0, w));
            }
            let app = w.app.as_str();
            let by_app = !app.is_empty() && app.len() >= 2 && (app.contains(&squashed) || squashed.contains(app) || words.contains(&app));
            by_app.then_some((1, w))
        })
        .min_by_key(|(rank, w)| (*rank, w.title.len()))
        .map(|(_, w)| w)
}

// ---------------------------------------------------------------- hands

/// What an action did.
#[derive(Debug)]
pub enum Done {
    Did(String),
    /// It was done, but didn't have the effect asked for.
    Failed(String),
    Shot { image: Image, cursor: Option<(i32, i32)>, focused: Option<Window> },
    Windows { list: Vec<Window>, focused: Option<Window> },
}

impl Done {
    pub fn ok(&self) -> bool {
        !matches!(self, Done::Failed(_))
    }
}

/// The only way Aegist touches the real desktop from a request.
pub struct Pilot {
    pub desk: Box<dyn Desktop>,
    max_actions: usize,
    delay_ms: u64,
    failsafe_px: i32,
    pub actions: usize,
    last_put: Option<(i32, i32)>,
    cancel: Arc<AtomicBool>,
    size: Option<(i32, i32)>,
    pub launcher: fn(&str) -> Result<()>,
    /// Glide the pointer instead of jumping (so you can see it move).
    pub glide: bool,
}

impl Pilot {
    pub fn new(desk: Box<dyn Desktop>, settings: &AgentSettings, cancel: Arc<AtomicBool>) -> Pilot {
        Pilot { desk, max_actions: settings.max_actions, delay_ms: settings.action_delay_ms, failsafe_px: settings.failsafe_px, actions: 0,
                last_put: None, cancel, size: None, launcher: system_launch, glide: true }
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Before each action: not stopped, not over budget, and you haven't taken the mouse back.
    fn check(&mut self) -> Result<()> {
        if self.cancelled() {
            bail!(Stopped("stopped: you asked me to".into()));
        }
        if self.actions >= self.max_actions {
            bail!(Stopped(format!("stopped: used all {} actions one run allows (agent.max_actions)", self.max_actions)));
        }
        if let (Some(put), Some(now)) = (self.last_put, self.desk.cursor()) {
            if (put.0 - now.0).abs().max((put.1 - now.1).abs()) > self.failsafe_px {
                bail!(Stopped("stopped: you moved the mouse, so I let go".into()));
            }
        }
        self.actions += 1;
        Ok(())
    }

    fn screen(&mut self) -> Result<(i32, i32)> {
        if let Some(s) = self.size {
            return Ok(s);
        }
        let s = self.desk.screen_size()?;
        self.size = Some(s);
        Ok(s)
    }

    /// Screen pixels for `spot`, refusing places off the screen.
    pub fn resolve(&mut self, spot: Spot) -> Result<(i32, i32)> {
        let (w, h) = self.screen()?;
        let (x, y) = match spot {
            Spot::Px(x, y) => (x, y),
            Spot::Frac(fx, fy) => ((fx * w as f32).round() as i32, (fy * h as f32).round() as i32),
            Spot::WindowCenter => {
                let win = self.desk.focused().ok_or_else(|| anyhow::anyhow!("no window is in front"))?;
                (win.rect.x + win.rect.w / 2, win.rect.y + win.rect.h / 2)
            }
        };
        let (x, y) = if matches!(spot, Spot::Frac(..)) { (x.clamp(0, w - 1), y.clamp(0, h - 1)) } else { (x, y) };
        if x < 0 || y < 0 || x >= w || y >= h {
            bail!(Stopped(format!("refused: ({x}, {y}) is off the screen, which is {w}×{h}")));
        }
        Ok((x, y))
    }

    /// Move the pointer, gliding there so you can see it go.
    fn put(&mut self, x: i32, y: i32, steps: i32, step_ms: u64) -> Result<()> {
        let from = if self.glide { self.last_put.or_else(|| self.desk.cursor()) } else { None };
        if let Some((fx, fy)) = from {
            let n = steps.max(1);
            for i in 1..n {
                let ease = |t: f32| t * t * (3.0 - 2.0 * t);
                let t = ease(i as f32 / n as f32);
                let (px, py) = (fx + ((x - fx) as f32 * t) as i32, fy + ((y - fy) as f32 * t) as i32);
                self.desk.move_to(px, py)?;
                self.last_put = Some((px, py));
                self.desk.wait(step_ms);
            }
        }
        self.desk.move_to(x, y)?;
        self.last_put = Some((x, y));
        Ok(())
    }

    fn settle(&mut self) {
        let ms = self.delay_ms;
        self.desk.wait(ms);
    }

    pub fn perform(&mut self, act: &Act) -> Result<Done> {
        if let Some(why) = act.refusal() {
            bail!(Stopped(format!("No. I won't {}: {why}.", act.describe())));
        }
        match act {
            Act::Move(s) => {
                let (x, y) = self.resolve(*s)?;
                self.check()?;
                self.put(x, y, 14, 8)?;
                self.settle();
                Ok(Done::Did(format!("the pointer is at ({x}, {y})")))
            }
            Act::Click { at, button, times } => {
                let place = match at {
                    Some(s) => {
                        let (x, y) = self.resolve(*s)?;
                        self.check()?;
                        self.put(x, y, 14, 8)?;
                        self.desk.wait(30);
                        (x, y)
                    }
                    None => {
                        self.check()?;
                        let here = self.desk.cursor().unwrap_or((0, 0));
                        self.last_put = Some(here);
                        here
                    }
                };
                for i in 0..*times {
                    self.desk.mouse_button(button.code(), true)?;
                    self.desk.wait(25);
                    self.desk.mouse_button(button.code(), false)?;
                    if i + 1 < *times {
                        self.desk.wait(70);
                    }
                }
                self.settle();
                Ok(Done::Did(format!("clicked at ({}, {})", place.0, place.1)))
            }
            Act::Drag { from, to } => {
                let (a, b) = (self.resolve(*from)?, self.resolve(*to)?);
                self.check()?;
                self.put(a.0, a.1, 14, 8)?;
                self.desk.wait(30);
                self.desk.mouse_button(1, true)?;
                let dist = (b.0 - a.0).abs().max((b.1 - a.1).abs());
                let steps = (dist / 8).clamp(4, 200);
                let saved = self.glide;
                self.glide = true;
                let r = self.put(b.0, b.1, steps, 4);
                self.glide = saved;
                self.desk.wait(30);
                self.desk.mouse_button(1, false)?;
                r?;
                self.settle();
                Ok(Done::Did(format!("dragged from ({}, {}) to ({}, {})", a.0, a.1, b.0, b.1)))
            }
            Act::Scroll(n) => {
                self.check()?;
                self.desk.scroll(*n)?;
                self.settle();
                Ok(Done::Did(format!("scrolled {} {}", if *n > 0 { "up" } else { "down" }, n.abs())))
            }
            Act::Type(t) => {
                // a little at a time, so esc and the mouse failsafe work mid-way
                let chars: Vec<char> = t.chars().collect();
                for chunk in chars.chunks(24) {
                    self.check()?;
                    self.desk.type_text(&chunk.iter().collect::<String>())?;
                    self.desk.wait(10);
                }
                self.settle();
                Ok(Done::Did(format!("typed {} character{}", chars.len(), if chars.len() == 1 { "" } else { "s" })))
            }
            Act::Key(k, n) => {
                for _ in 0..*n {
                    self.check()?;
                    self.desk.key(k)?;
                    self.desk.wait(40);
                }
                self.settle();
                Ok(Done::Did(format!("pressed {k}{}", if *n > 1 { format!(" ×{n}") } else { String::new() })))
            }
            Act::Focus(name) => {
                self.check()?;
                let windows = self.desk.windows();
                let Some(w) = best_window(&windows, name).cloned() else {
                    let titles: Vec<String> = windows.iter().take(8).map(|w| format!("{:?}", text::truncate(&w.title, 40))).collect();
                    return Ok(Done::Failed(format!("no window's title has {name:?} in it. Open windows: {}", titles.join(", "))));
                };
                self.desk.activate(&w)?;
                self.desk.wait(250);
                match self.desk.focused() {
                    Some(f) if f.id == w.id || f.title == w.title => Ok(Done::Did(format!("{:?} is in front", w.title))),
                    _ => Ok(Done::Failed(format!("I asked for {:?} to come to the front, but the system kept another window there", w.title))),
                }
            }
            Act::Launch(target) => {
                self.check()?;
                let before: Vec<u64> = self.desk.windows().iter().map(|w| w.id).collect();
                (self.launcher)(target)?;
                let wanted = APPS.iter().find(|(names, ..)| names.contains(&target.to_lowercase().as_str())).map_or(target.to_lowercase(), |k| k.1.to_string());
                // look for its window for up to 12 seconds
                for poll in 0..48 {
                    if self.cancelled() {
                        bail!(Stopped("stopped: you asked me to".into()));
                    }
                    self.desk.wait(250);
                    let now = self.desk.windows();
                    if let Some(w) = now.iter().find(|w| !before.contains(&w.id) && !w.title.trim().is_empty()) {
                        let w = w.clone();
                        let _ = self.desk.activate(&w);
                        self.desk.wait(300);
                        return Ok(Done::Did(format!("opened it: {:?} is in front", w.title)));
                    }
                    // no new window after 3 seconds: it was already open and may just have come forward
                    if poll >= 12 {
                        let short = target.trim().to_lowercase();
                        if let Some(w) = best_window(&now, &short).or_else(|| best_window(&now, &wanted)).cloned() {
                            let _ = self.desk.activate(&w);
                            self.desk.wait(300);
                            return Ok(Done::Did(format!("{:?} is in front", w.title)));
                        }
                    }
                }
                Ok(Done::Failed(format!("I started {target}, but no window for it appeared within 12 seconds")))
            }
            Act::Wait(ms) => {
                let until = Instant::now() + Duration::from_millis(*ms);
                while Instant::now() < until {
                    if self.cancelled() {
                        bail!(Stopped("stopped: you asked me to".into()));
                    }
                    std::thread::sleep(Duration::from_millis(50).min(until.saturating_duration_since(Instant::now())));
                }
                Ok(Done::Did(format!("waited {:.1}s", *ms as f64 / 1000.0)))
            }
            Act::Look => {
                let image = self.desk.capture()?;
                Ok(Done::Shot { image, cursor: self.desk.cursor(), focused: self.desk.focused() })
            }
            Act::Where => {
                let at = self.desk.cursor();
                let front = self.desk.focused();
                Ok(Done::Did(format!("the pointer is at {}{}", at.map_or("an unknown place".into(), |(x, y)| format!("({x}, {y})")),
                                     front.map_or(String::new(), |w| format!(" · {:?} is in front", w.title)))))
            }
            Act::Windows => Ok(Done::Windows { list: self.desk.windows(), focused: self.desk.focused() }),
        }
    }
}

// ---------------------------------------------------------------- session

/// The desktop, once you've said Aegist may use it this session.
pub fn open_pilot(ctx: &Ctx, what: &str) -> Result<Pilot> {
    let desk = desktop::open()?;
    let asked_before = ctx.shared.lock().expect("lock").desktop_ok;
    if ctx.settings.agent.confirm && !asked_before {
        let q = format!("Let Aegist use your mouse and keyboard this session? (first: {what}) - it only does what you ask; move the mouse or press esc to stop it");
        if ctx.ask(&q, &[('y', "yes"), ('n', "no")]) != 'y' {
            bail!("Okay - I won't touch your mouse or keyboard.");
        }
        ctx.shared.lock().expect("lock").desktop_ok = true;
    }
    Ok(Pilot::new(desk, &ctx.settings.agent, ctx.cancel.clone()))
}

fn head(act: &Act) -> String {
    let icon = match act {
        Act::Type(_) | Act::Key(..) => "⌨",
        Act::Look | Act::Where | Act::Windows => "◳",
        Act::Launch(_) | Act::Focus(_) => "▣",
        Act::Wait(_) => "◷",
        _ => "◉",
    };
    format!("  {} {}", style::fg(pal::PINK, icon), Style::new().bold().paint(&act.describe()))
}

/// Show what an action did.
pub fn show(ctx: &Ctx, done: &Done) {
    let w = ctx.width();
    match done {
        Done::Did(s) => ctx.line(format!("    {} {}", style::fg(pal::GREEN, "✓"), style::dim(s))),
        Done::Failed(s) => {
            for (i, l) in text::wrap(s, w.saturating_sub(8)).into_iter().enumerate() {
                ctx.line(if i == 0 { format!("    {} {l}", style::fg(pal::YELLOW, "✗")) } else { format!("      {}", style::dim(&l)) });
            }
        }
        Done::Shot { image, cursor, focused } => {
            let saved = save_png(ctx, image).ok();
            let mut lines = draw_screen(image, *cursor, w.saturating_sub(12).clamp(20, 150));
            lines.push(format!("    {}", style::faint(&format!("{}×{} · pointer at {} · in front: {}", image.w, image.h,
                cursor.map_or("?".into(), |(x, y)| format!("({x}, {y})")), focused.as_ref().map_or("nothing".into(), |f| format!("{:?}", f.title))))));
            if let Some(p) = saved {
                lines.push(format!("    {}", style::faint(&format!("full size: {}", p.display()))));
            }
            ctx.print(lines);
        }
        Done::Windows { list, focused } => {
            let mut lines = Vec::new();
            if list.is_empty() {
                lines.push(format!("    {}", style::dim("no windows with titles")));
            }
            for win in list.iter().take(40) {
                let front = focused.as_ref().is_some_and(|f| f.id == win.id);
                let mark = if front { style::fg(pal::GREEN, "●") } else { style::faint("○") };
                let r = &win.rect;
                lines.push(text::truncate(&format!("    {mark} {}  {}", Style::new().bold().paint(&text::pad(&text::truncate(&win.title, 48), 48)),
                    style::faint(&format!("{}{}×{} at ({}, {})", if win.app.is_empty() { String::new() } else { format!("{} · ", win.app) }, r.w, r.h, r.x, r.y))), w));
            }
            ctx.print(lines);
        }
    }
}

/// Do one action you asked for: refuse it, ask about it, or do it and show what happened.
pub fn run_one(ctx: &Ctx, act: Act) -> Result<()> {
    ctx.line(String::new());
    ctx.line(head(&act));
    if let Some(why) = act.refusal() {
        bail!("No. I won't {}: {why}.", act.describe());
    }
    let mut pilot = open_pilot(ctx, &act.describe())?;
    if let Some(why) = act.needs_ok() {
        if ctx.ask(&format!("{} - {why}. Go ahead?", act.describe()), &[('y', "yes"), ('n', "no")]) != 'y' {
            ctx.line(format!("    {}", style::dim("left it alone")));
            return Ok(());
        }
    }
    ctx.status(format!("{}…", act.describe()));
    let done = pilot.perform(&act)?;
    show(ctx, &done);
    Ok(())
}

/// `/click`, `/type`, `/screen`, ... as one action.
pub fn command(ctx: &Ctx, name: &str, args: &str) -> Result<()> {
    let a = args.trim();
    let need = |what: &str| -> Result<()> {
        if a.is_empty() {
            bail!("{what}");
        }
        Ok(())
    };
    let parsed = |phrase: String| -> Result<Act> {
        match parse(&phrase) {
            Some(Ok(act)) => Ok(act),
            Some(Err(why)) => bail!("{why}"),
            None => bail!("I don't know how to do {phrase:?} on the screen"),
        }
    };
    let act = match name {
        "screen" | "screenshot" | "shot" => Act::Look,
        "windows" | "wins" => Act::Windows,
        "where" | "cursor" | "pointer" => Act::Where,
        "focus" | "switch" => {
            need("/focus <part of a window's title> - /windows lists them")?;
            Act::Focus(a.to_string())
        }
        "launch" | "app-open" | "browse" => {
            need("/launch <app or web address> - e.g. /launch calculator, /launch github.com")?;
            Act::Launch(a.to_string())
        }
        "type" => {
            need("/type <text>")?;
            Act::Type(quoted(a).filter(|q| q.len() + 2 == a.len()).unwrap_or_else(|| a.to_string()))
        }
        "key" | "press" | "keys" => {
            need("/key <key or combo> - e.g. /key enter, /key ctrl+c, /key tab 3 times")?;
            parsed(format!("press {a}"))?
        }
        "mouse" | "move" => {
            if a.is_empty() {
                Act::Where
            } else {
                parsed(format!("move the mouse to {a}"))?
            }
        }
        "click" => parsed(format!("click {a}"))?,
        "rclick" | "right-click" | "rightclick" => parsed(format!("right click {a}"))?,
        "dclick" | "double-click" | "doubleclick" => parsed(format!("double click {a}"))?,
        "drag" => parsed(format!("drag {a}"))?,
        "scroll" => parsed(format!("scroll {a}"))?,
        "wait" => parsed(format!("wait {}", if a.is_empty() { "1" } else { a }))?,
        other => bail!("/{other} isn't a screen command"),
    };
    run_one(ctx, act)
}

// ---------------------------------------------------------------- pictures

fn marked_pixels(img: &Image, cursor: Option<(i32, i32)>, mark_r: i32) -> Vec<style::Rgb> {
    let mut px: Vec<style::Rgb> = img.px.chunks(3).map(|p| style::Rgb(p[0], p[1], p[2])).collect();
    if let Some((cx, cy)) = cursor {
        for dy in -mark_r..=mark_r {
            for dx in -mark_r..=mark_r {
                let d2 = dx * dx + dy * dy;
                let (x, y) = (cx + dx, cy + dy);
                if d2 > mark_r * mark_r || x < 0 || y < 0 || x as usize >= img.w || y as usize >= img.h {
                    continue;
                }
                let inner = d2 * 2 < mark_r * mark_r;
                px[y as usize * img.w + x as usize] = if inner { pal::PINK } else { style::Rgb(20, 16, 30) };
            }
        }
    }
    px
}

/// The screen drawn in the terminal, with pixel coordinates along the top
/// and left (so you can say where to click) and the pointer marked.
pub fn draw_screen(img: &Image, cursor: Option<(i32, i32)>, cols: usize) -> Vec<String> {
    let cols = cols.min(img.w).max(8);
    let scale = img.w as f32 / cols as f32;
    let px = marked_pixels(img, cursor, (scale * 1.6).ceil() as i32 + 1);
    let rows = crate::ide::render_pixels(img.w, img.h, &px, cols, 70);
    let margin = 6;
    let mut ruler = vec![' '; cols];
    let mut col = 0;
    while col < cols {
        let label = ((col as f32 * scale) as i32).to_string();
        if col + label.len() <= cols {
            for (i, ch) in label.chars().enumerate() {
                ruler[col + i] = ch;
            }
        }
        col += 12;
    }
    let mut out = vec![format!("  {}{}", " ".repeat(margin), style::faint(&ruler.into_iter().collect::<String>()))];
    for (r, row) in rows.into_iter().enumerate() {
        let label = if r % 4 == 0 { format!("{:>w$} ", (r as f32 * 2.0 * scale) as i32, w = margin - 1) } else { " ".repeat(margin) };
        out.push(format!("  {}{row}", style::faint(&label)));
    }
    out
}

fn save_png(ctx: &Ctx, img: &Image) -> Result<std::path::PathBuf> {
    let path = ctx.settings.data_path("screens").join("last.png");
    let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let mut enc = png::Encoder::new(file, img.w as u32, img.h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&img.px)?;
    Ok(path)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::agent::vision::Rect;
    use std::sync::Mutex;

    /// A screen that writes down what's done to it.
    pub struct Fake {
        pub log: Arc<Mutex<Vec<String>>>,
        pub at: (i32, i32),
        pub windows: Vec<Window>,
        pub front: Option<u64>,
        /// Someone grabs the mouse after this many moves.
        pub grab_after: Option<usize>,
        moves: usize,
    }

    impl Fake {
        pub fn new() -> (Fake, Arc<Mutex<Vec<String>>>) {
            let log = Arc::new(Mutex::new(Vec::new()));
            let win = |id, t: &str, app: &str| Window { id, title: t.into(), rect: Rect::new(100, 100, 400, 300), app: app.into() };
            (Fake { log: log.clone(), at: (0, 0), windows: vec![win(1, "Untitled - Notepad", "notepad"), win(2, "Mozilla Firefox", "firefox"),
                                                                 win(3, "root@vm: ~", "xterm")], front: Some(1),
                    grab_after: None, moves: 0 }, log)
        }
        fn note(&self, s: String) {
            self.log.lock().unwrap().push(s);
        }
    }

    impl Desktop for Fake {
        fn name(&self) -> String {
            "fake".into()
        }
        fn capture(&mut self) -> Result<Image> {
            Ok(Image::new(1000, 800, [30, 30, 40]))
        }
        fn move_to(&mut self, x: i32, y: i32) -> Result<()> {
            self.moves += 1;
            self.at = (x, y);
            if self.grab_after.is_some_and(|n| self.moves > n) {
                self.at = (x + 300, y); // a hand on the mouse
            }
            Ok(())
        }
        fn button(&mut self, down: bool) -> Result<()> {
            self.mouse_button(1, down)
        }
        fn mouse_button(&mut self, b: u8, down: bool) -> Result<()> {
            self.note(format!("button {b} {} at {:?}", if down { "down" } else { "up" }, self.at));
            Ok(())
        }
        fn scroll(&mut self, n: i32) -> Result<()> {
            self.note(format!("scroll {n}"));
            Ok(())
        }
        fn key(&mut self, combo: &str) -> Result<()> {
            self.note(format!("key {combo}"));
            Ok(())
        }
        fn type_text(&mut self, text: &str) -> Result<()> {
            self.note(format!("type {text}"));
            Ok(())
        }
        fn cursor(&mut self) -> Option<(i32, i32)> {
            Some(self.at)
        }
        fn focused(&mut self) -> Option<Window> {
            self.windows.iter().find(|w| Some(w.id) == self.front).cloned()
        }
        fn windows(&mut self) -> Vec<Window> {
            self.windows.clone()
        }
        fn activate(&mut self, w: &Window) -> Result<()> {
            self.front = Some(w.id);
            self.note(format!("activate {}", w.title));
            Ok(())
        }
        fn wait(&mut self, _ms: u64) {}
    }

    fn pilot(fake: Fake) -> Pilot {
        let mut p = Pilot::new(Box::new(fake), &AgentSettings::default(), Arc::new(AtomicBool::new(false)));
        p.launcher = |_| Ok(());
        p
    }

    fn act(s: &str) -> Act {
        parse(s).unwrap_or_else(|| panic!("{s:?} not understood")).unwrap_or_else(|e| panic!("{s:?}: {e}"))
    }

    #[test]
    fn plain_words_become_actions() {
        assert_eq!(act("click 640 400"), Act::Click { at: Some(Spot::Px(640, 400)), button: Button::Left, times: 1 });
        assert_eq!(act("please double click at 10, 20"), Act::Click { at: Some(Spot::Px(10, 20)), button: Button::Left, times: 2 });
        assert_eq!(act("right-click"), Act::Click { at: None, button: Button::Right, times: 1 });
        assert_eq!(act("click right 5 6"), Act::Click { at: Some(Spot::Px(5, 6)), button: Button::Right, times: 1 });
        assert_eq!(act("click the center"), Act::Click { at: Some(Spot::Frac(0.5, 0.5)), button: Button::Left, times: 1 });
        assert_eq!(act("click 50% 20%"), Act::Click { at: Some(Spot::Frac(0.5, 0.2)), button: Button::Left, times: 1 });
        assert_eq!(act("click the top right"), Act::Click { at: Some(Spot::Frac(0.95, 0.05)), button: Button::Left, times: 1 });
        assert_eq!(act("move the mouse to 100 200"), Act::Move(Spot::Px(100, 200)));
        assert_eq!(act("hover over the middle of the window"), Act::Move(Spot::WindowCenter));
        assert_eq!(act("drag from 10 20 to 300 400"), Act::Drag { from: Spot::Px(10, 20), to: Spot::Px(300, 400) });
        assert_eq!(act("scroll down 5"), Act::Scroll(-5));
        assert_eq!(act("scroll up"), Act::Scroll(3));
        assert_eq!(act("type \"Hello, World!\""), Act::Type("Hello, World!".into()));
        assert_eq!(act("type hello there"), Act::Type("hello there".into()));
        assert_eq!(act("press the enter key"), Act::Key("enter".into(), 1));
        assert_eq!(act("press Control-Shift-T"), Act::Key("ctrl+shift+t".into(), 1));
        assert_eq!(act("press tab 3 times"), Act::Key("tab".into(), 3));
        assert_eq!(act("hit enter twice"), Act::Key("enter".into(), 2));
        assert_eq!(act("open a new tab"), Act::Key("ctrl+t".into(), 1));
        assert_eq!(act("open notepad"), Act::Launch("notepad".into()));
        assert_eq!(act("launch the calculator app"), Act::Launch("calculator".into()));
        assert_eq!(act("go to github.com"), Act::Launch("github.com".into()));
        assert_eq!(act("switch to firefox"), Act::Focus("firefox".into()));
        assert_eq!(act("wait 2 seconds"), Act::Wait(2000));
        assert_eq!(act("wait 300ms"), Act::Wait(300));
        assert_eq!(act("take a screenshot"), Act::Look);
        assert_eq!(act("where's the mouse"), Act::Where);
        assert_eq!(act("what windows are open"), Act::Windows);
        // not for the screen: files, code, and other commands
        assert!(parse("open main.py").is_none());
        assert!(parse("write a function that adds numbers").is_none());
        assert!(parse("run the tests").is_none());
        assert!(parse("move the file to the folder").is_none());
    }

    #[test]
    fn it_wont_guess_where_things_are() {
        let e = parse("click the OK button").unwrap().unwrap_err();
        assert!(e.contains("I don't know where") && e.contains("OK button"), "{e}");
        assert!(parse("drag the file").unwrap().is_err());
        assert!(parse("type").unwrap().is_err());
    }

    #[test]
    fn risky_keys_ask_and_some_are_never_pressed() {
        assert_eq!(key_rule("ctrl+c"), KeyRule::Fine);
        assert!(matches!(key_rule("alt+f4"), KeyRule::Confirm(_)));
        assert!(matches!(key_rule("Ctrl+Alt+Delete"), KeyRule::Refuse(_)));
        assert!(act("close the window").needs_ok().is_some());
        assert!(Act::Type("rm -rf /\n".into()).needs_ok().is_some());
        assert!(Act::Launch("my bank account".into()).refusal().is_some());
        assert!(Act::Key("ctrl+alt+del".into(), 1).refusal().is_some());
    }

    #[test]
    fn apps_are_found_by_everyday_names() {
        let has = |p: &str| matches!(p, "gnome-calculator" | "xdg-open" | "firefox");
        assert_eq!(launch_argv("calculator", Os::Linux, &has).unwrap(), vec!["gnome-calculator"]);
        assert_eq!(launch_argv("github.com", Os::Linux, &has).unwrap(), vec!["xdg-open", "https://github.com"]);
        assert!(launch_argv("paint", Os::Linux, &has).unwrap_err().contains("couldn't find a paint app"));
        assert_eq!(launch_argv("paint", Os::Windows, &has).unwrap(), vec!["cmd", "/C", "start", "", "mspaint"]);
        assert_eq!(launch_argv("spotify", Os::Windows, &has).unwrap().last().unwrap(), "spotify");
        assert_eq!(launch_argv("notepad", Os::Mac, &has).unwrap(), vec!["open", "-a", "TextEdit"]);
        assert!(is_url("https://x.dev/a b") && is_url("www.example.org") && !is_url("notepad") && !is_url("main.py"));
    }

    #[test]
    fn hands_do_what_was_asked_and_let_go_when_you_take_the_mouse() {
        let (fake, log) = Fake::new();
        let mut p = pilot(fake);
        p.perform(&act("click 640 400")).unwrap();
        p.perform(&act("right click")).unwrap();
        p.perform(&act("type hi")).unwrap();
        p.perform(&act("press ctrl+c")).unwrap();
        p.perform(&act("scroll down 2")).unwrap();
        let got = log.lock().unwrap().clone();
        assert_eq!(got, vec!["button 1 down at (640, 400)", "button 1 up at (640, 400)", "button 3 down at (640, 400)", "button 3 up at (640, 400)",
                             "type hi", "key ctrl+c", "scroll -2"]);
        assert!(p.perform(&act("click 5000 10")).unwrap_err().to_string().contains("off the screen"));
        assert!(matches!(p.perform(&act("switch to firefox")).unwrap(), Done::Did(_)));
        assert!(matches!(p.perform(&act("switch to photoshop")).unwrap(), Done::Failed(_)));
        // by program, when the title doesn't say
        assert!(matches!(p.perform(&act("switch to xterm")).unwrap(), Done::Did(s) if s.contains("root@vm")));
        assert!(matches!(p.perform(&Act::Look).unwrap(), Done::Shot { .. }));
        // someone grabs the mouse mid-run: it lets go
        let (mut fake, log) = Fake::new();
        fake.grab_after = Some(3);
        let mut p = pilot(fake);
        p.glide = false;
        p.perform(&act("move the mouse to 10 10")).unwrap();
        p.perform(&act("move the mouse to 20 20")).unwrap();
        p.perform(&act("move the mouse to 30 30")).unwrap();
        p.perform(&act("move the mouse to 40 40")).unwrap();
        let err = p.perform(&act("click 50 50")).unwrap_err();
        assert!(err.to_string().contains("you moved the mouse"), "{err}");
        assert!(log.lock().unwrap().is_empty(), "no click after the grab");
        // esc stops it
        let (fake, _) = Fake::new();
        let mut p = pilot(fake);
        p.cancel.store(true, Ordering::Relaxed);
        assert!(p.perform(&act("type hello")).unwrap_err().to_string().contains("asked me to"));
        // refused before anything happens
        let (fake, log) = Fake::new();
        let mut p = pilot(fake);
        assert!(p.perform(&Act::Key("ctrl+alt+delete".into(), 1)).is_err());
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn launching_waits_for_the_new_window() {
        let (fake, log) = Fake::new();
        let mut p = pilot(fake);
        // the fake shows no new window, but notepad is already open: it comes forward
        let done = p.perform(&Act::Launch("notepad".into())).unwrap();
        assert!(matches!(&done, Done::Did(s) if s.contains("Notepad")), "{done:?}");
        assert!(log.lock().unwrap().iter().any(|l| l == "activate Untitled - Notepad"));
    }

    #[test]
    fn the_screen_is_drawn_with_coordinates() {
        let img = Image::new(800, 400, [10, 200, 30]);
        let lines = draw_screen(&img, Some((400, 200)), 80);
        assert!(text::strip_ansi(&lines[0]).contains("0") && text::strip_ansi(&lines[0]).contains("120"), "{}", text::strip_ansi(&lines[0]));
        assert!(lines.len() > 10 && text::strip_ansi(&lines[1]).trim_start().starts_with('0'));
    }
}
