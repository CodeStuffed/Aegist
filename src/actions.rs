//! Everything you can ask Aegist to do, as slash commands or in plain
//! words. Plain requests are understood by simple, predictable rules - not
//! by guessing: a request it can't place gets a straight "I don't know
//! what you want", and a question gets a straight "no, I can't answer
//! questions" rather than made-up English.

use crate::assist::{Answer, Assistant, FixOutcome, Progress};
use crate::checkpoint;
use crate::ide::{self, JobState};
use crate::lang::{self, Lang};
use crate::learn;
use crate::proc;
use crate::project::Project;
use crate::session::{Ctx, Msg};
use crate::trainer::{self, commas, human_duration, Budget, Size};
use crate::ui::style::{self, pal, Style};
use crate::ui::text;
use crate::ui::widgets;
use crate::verify::Verdict;
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

pub struct Command {
    pub name: &'static str,
    pub args: &'static str,
    pub about: &'static str,
    pub group: &'static str,
}

const fn cmd(group: &'static str, name: &'static str, args: &'static str, about: &'static str) -> Command {
    Command { name, args, about, group }
}

/// Command groups, in the order /help shows them.
pub const GROUPS: &[(&str, &str)] = &[
    ("code", "Code"),
    ("run", "Run things"),
    ("look", "Look around"),
    ("screen", "Your screen: be the mouse and keyboard"),
    ("auto", "Autopilot"),
    ("think", "Decide and predict"),
    ("model", "The model"),
    ("session", "Session"),
];

pub const COMMANDS: &[Command] = &[
    cmd("code", "write", "<file> <what it should do>", "Write a new file - checked before it's saved"),
    cmd("code", "complete", "<file>[:line]", "Fill in code at a line (a TODO, or the file's end)"),
    cmd("code", "fix", "[command]", "Make a failing command pass (default: the tests)"),
    cmd("code", "show", "", "The last code Aegist refused to stand behind"),
    cmd("code", "diff", "", "Everything Aegist changed this session"),
    cmd("code", "undo", "", "Undo Aegist's last change"),
    cmd("code", "copy", "[text]", "Copy the last code it wrote (or the last output) to the clipboard"),
    cmd("run", "run", "<command or file>", "Run a program and show its output"),
    cmd("run", "test", "", "Run the project's tests"),
    cmd("run", "start", "<command>", "Start a program in the background (servers, games)"),
    cmd("run", "jobs", "", "Background programs and how they're doing"),
    cmd("run", "logs", "<job>", "A background program's latest output"),
    cmd("run", "stop", "<job|all>", "Stop a background program"),
    cmd("run", "serve", "[folder]", "Serve a website folder on localhost"),
    cmd("run", "preview", "<url, file or job>", "Look at a web page, drawn right here"),
    cmd("run", "git", "[status|log|diff|any git command]", "Git, asking before anything that can't be undone"),
    cmd("look", "open", "<file>[:from-to]", "Show a file, highlighted"),
    cmd("look", "find", "<name pattern>", "Find files by name (* and ? work)"),
    cmd("look", "grep", "<text or regex>", "Search the project's code"),
    cmd("look", "tree", "[folder]", "The project's layout"),
    cmd("look", "cd", "<folder>", "Work in another folder"),
    cmd("screen", "screen", "", "Screenshot, drawn here with coordinates to click by"),
    cmd("screen", "windows", "", "The open windows (● is in front)"),
    cmd("screen", "focus", "<window title>", "Bring a window to the front"),
    cmd("screen", "launch", "<app or web address>", "Open an app (notepad, calculator, paint...) or a website"),
    cmd("screen", "click", "[right|double] [x y | 50% 20% | center]", "Click - where you say, or where the pointer is"),
    cmd("screen", "rclick", "[x y]", "Right-click"),
    cmd("screen", "dclick", "[x y]", "Double-click"),
    cmd("screen", "mouse", "[x y]", "Move the pointer there (or say where it is)"),
    cmd("screen", "drag", "<x1 y1> <x2 y2>", "Drag from one place to another"),
    cmd("screen", "scroll", "<up|down> [n]", "Turn the mouse wheel"),
    cmd("screen", "type", "<text>", "Type text into the window in front"),
    cmd("screen", "key", "<key or combo> [n times]", "Press keys: enter, ctrl+c, alt+tab, tab 3 times..."),
    cmd("screen", "wait", "<seconds>", "Wait (for an app to catch up)"),
    cmd("screen", "agent", "learn|do|practice [--app name | --sim]", "Learn an app on your screen by trying it, then use it"),
    cmd("auto", "auto", "<steps in plain words>", "Plan several steps, show the plan, then do them all"),
    cmd("auto", "do", "<steps in plain words>", "Same as /auto"),
    cmd("auto", "safety", "", "What Aegist will and won't do on your computer"),
    cmd("think", "decide", "<question>: a | b | c", "Pick an option with its probability - or say I don't know"),
    cmd("think", "answer", "<option>", "The right answer to the last decision - it learns from it"),
    cmd("think", "predict", "<numbers> [next n]", "The next values of a series, with a range - or I don't know"),
    cmd("think", "estimate", "<thing> [= number]", "Estimate from similar things you've told it (= teaches it)"),
    cmd("think", "good", "/ bad", "Tell Aegist whether its last action or decision was right"),
    cmd("model", "learn", "<folder | git URL | --pack name>", "Add code for the model to learn from"),
    cmd("model", "train", "[hours]", "Train the model (ctrl+c stops and saves)"),
    cmd("model", "model", "", "The model: size, training, and honest limits"),
    cmd("model", "config", "[setting [value]]", "See or change settings (certainty limits, agent, voice...)"),
    cmd("session", "status", "", "Everything about this session at a glance"),
    cmd("session", "history", "[n]", "What you've asked this session"),
    cmd("session", "export", "[file]", "Save this session's transcript to a file"),
    cmd("session", "voice", "[on|off]", "Speak answers aloud (the system's own speech)"),
    cmd("session", "install", "", "Add Aegist to your apps menu, to open like any app"),
    cmd("session", "clear", "", "Clear the screen"),
    cmd("session", "help", "[group]", "All of this"),
    cmd("session", "exit", "", "Leave (background programs are stopped)"),
];

/// What a plain-words request is asking for.
#[derive(Debug, PartialEq)]
pub enum Intent {
    Write { path: Option<String>, lang: Option<&'static Lang>, description: String },
    Complete(String),
    Fix(Option<String>),
    Run(String),
    Test,
    Grep(String),
    Open(String),
    Undo,
    Greeting,
    /// A question, or a request for English: the honest answer is no.
    Question,
    Unknown,
}

fn path_like(word: &str) -> Option<String> {
    let w = word.trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | ',' | ';' | '(' | ')'));
    let w = w.strip_suffix('.').filter(|s| s.contains('.')).unwrap_or(w);
    let file = w.split(':').next().unwrap_or(w);
    (file.contains('.') && !file.starts_with('.') && lang::is_code(Path::new(file)) && !file.contains("://")).then(|| w.to_string())
}

/// Understand a plain-words request by simple rules.
pub fn interpret(text: &str) -> Intent {
    let t = text.trim();
    let lower = t.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let first = words.first().copied().unwrap_or("");
    let has = |w: &str| words.contains(&w);
    let path = t.split_whitespace().find_map(path_like);
    let quoted = t.split('`').nth(1).map(str::to_string).filter(|s| !s.is_empty());
    if matches!(lower.as_str(), "hi" | "hello" | "hey" | "yo" | "hi!" | "hello!" | "hey!") {
        return Intent::Greeting;
    }
    if first == "undo" {
        return Intent::Undo;
    }
    const WRITE: &[&str] = &["write", "create", "make", "build", "generate", "code", "implement", "program", "scaffold", "add"];
    if WRITE.contains(&first) {
        let lang = language_in(&lower);
        return Intent::Write { path, lang, description: t.to_string() };
    }
    if matches!(first, "complete" | "finish" | "continue" | "fill") {
        return match path {
            Some(p) => Intent::Complete(p),
            None => Intent::Unknown,
        };
    }
    if matches!(first, "fix" | "debug" | "repair") {
        let cmd = quoted.or_else(|| {
            let rest = t.split_once(char::is_whitespace).map(|x| x.1.trim()).unwrap_or("");
            let head = rest.split_whitespace().next().unwrap_or("");
            (proc::which(head).is_some() && !matches!(head, "the" | "my" | "it" | "this")).then(|| rest.to_string())
        });
        return Intent::Fix(cmd);
    }
    if lower == "test" || lower == "tests" || lower.starts_with("run the test") || lower.starts_with("run tests") || lower.starts_with("run my test") {
        return Intent::Test;
    }
    if first == "run" || first == "execute" {
        let rest = t.split_once(char::is_whitespace).map(|x| x.1.trim()).unwrap_or("");
        return if rest.is_empty() { Intent::Unknown } else { Intent::Run(quoted.unwrap_or_else(|| rest.to_string())) };
    }
    if matches!(first, "show" | "open" | "cat" | "view" | "read") {
        if let Some(p) = path {
            return Intent::Open(p);
        }
    }
    if matches!(first, "find" | "search" | "grep" | "locate") || lower.starts_with("where is") || lower.starts_with("where's") {
        let term = quoted.unwrap_or_else(|| {
            words.iter().filter(|w| !matches!(**w, "find" | "search" | "grep" | "locate" | "where" | "is" | "where's" | "for" | "the" | "a"
                                                     | "defined" | "function" | "class" | "in" | "code" | "usages" | "of" | "uses"))
                .copied().collect::<Vec<_>>().join(" ")
        });
        return if term.is_empty() { Intent::Unknown } else { Intent::Grep(term) };
    }
    const ASK: &[&str] = &["what", "why", "how", "who", "when", "which", "explain", "tell", "describe", "is", "are", "does", "do", "should",
                           "could", "would", "can", "will", "summarize", "compare", "teach"];
    if t.ends_with('?') || ASK.contains(&first) || has("explain") {
        return Intent::Question;
    }
    Intent::Unknown
}

/// Words that name a language in a request. Only real names: plenty of
/// file extensions are also English words ("go", "less", "pm").
const LANG_WORDS: &[(&str, &str)] = &[
    ("python", "Python"), ("py", "Python"), ("python3", "Python"), ("javascript", "JavaScript"), ("js", "JavaScript"), ("node", "JavaScript"),
    ("nodejs", "JavaScript"), ("node.js", "JavaScript"), ("typescript", "TypeScript"), ("ts", "TypeScript"), ("rust", "Rust"),
    ("golang", "Go"), ("c++", "C++"), ("cpp", "C++"), ("c#", "C#"), ("csharp", "C#"), ("java", "Java"), ("kotlin", "Kotlin"),
    ("swift", "Swift"), ("ruby", "Ruby"), ("php", "PHP"), ("lua", "Lua"), ("perl", "Perl"), ("bash", "Shell"), ("shell", "Shell"),
    ("zsh", "Shell"), ("powershell", "PowerShell"), ("html", "HTML"), ("css", "CSS"), ("sql", "SQL"), ("haskell", "Haskell"),
    ("ocaml", "OCaml"), ("elixir", "Elixir"), ("erlang", "Erlang"), ("clojure", "Clojure"), ("julia", "Julia"), ("dart", "Dart"),
    ("zig", "Zig"), ("scala", "Scala"), ("assembly", "Assembly"), ("asm", "Assembly"), ("nasm", "Assembly"), ("cuda", "CUDA"),
    ("glsl", "GLSL"), ("wgsl", "WGSL"), ("verilog", "Verilog"), ("website", "HTML"), ("webpage", "HTML"), ("json", "JSON"),
    ("yaml", "YAML"), ("markdown", "Markdown"), ("dockerfile", "Dockerfile"), ("makefile", "Makefile"),
];

/// The language a request names ("... in rust", "a python script", "a website").
pub fn language_in(lower: &str) -> Option<&'static Lang> {
    let named = |name: &str| lang::LANGS.iter().find(|l| l.name == name);
    let words: Vec<&str> = lower.split(|c: char| c.is_whitespace() || c == ',').map(|w| w.trim_end_matches(['.', '!', ':'])).filter(|w| !w.is_empty()).collect();
    for w in &words {
        if let Some((_, name)) = LANG_WORDS.iter().find(|(word, _)| word == w) {
            return named(name);
        }
    }
    // one-letter and everyday-word names only in unmistakable phrases
    for (word, name) in [("go", "Go"), ("c", "C"), ("r", "R")] {
        for (i, w) in words.iter().enumerate() {
            let after = words.get(i + 1).copied().unwrap_or("");
            let before = if i > 0 { words[i - 1] } else { "" };
            if *w == word && (before == "in" || matches!(after, "program" | "code" | "function" | "script" | "file" | "module" | "library" | "server")) {
                return named(name);
            }
        }
    }
    if ["web page", "landing page", "home page", "homepage", "browser game", "web app", "web game", "html page"].iter().any(|p| lower.contains(p)) {
        return named("HTML");
    }
    None
}

// ------------------------------------------------------------------ help

pub fn help(ctx: &Ctx, args: &str) {
    let w = ctx.width();
    let only = args.trim().trim_start_matches('/').to_lowercase();
    let mut lines = vec![String::new()];
    let name_w = COMMANDS.iter().map(|c| c.name.len() + c.args.len() + 2).max().unwrap_or(20).min(34);
    for (i, (group, title)) in GROUPS.iter().enumerate() {
        if !only.is_empty() && !group.starts_with(&only) && !title.to_lowercase().contains(&only) {
            continue;
        }
        lines.push(format!("  {}", style::gradient_styled(title, i as f32 / GROUPS.len() as f32, true)));
        for c in COMMANDS.iter().filter(|c| c.group == *group) {
            let left = format!("{} {}", Style::new().fg(pal::VIOLET).bold().paint(&format!("/{}", c.name)), style::faint(c.args));
            lines.push(text::truncate(&format!("    {}  {}", text::pad(&text::truncate(&left, name_w + 1), name_w + 1), style::dim(c.about)), w));
        }
        lines.push(String::new());
    }
    if !only.is_empty() {
        ctx.print(lines);
        return;
    }
    lines.push(format!("  {}", style::gradient_styled("Or just say it", 0.3, true)));
    for ex in ["write a snake game as a web page", "complete src/app.py:42", "fix `pytest -q`", "run main.py",
               "open notepad, type hello world and press enter", "switch to firefox then scroll down 5", "take a screenshot",
               "click 640 400", "fix the tests then run main.py", "draw a red rectangle in paint", "good / bad"] {
        lines.push(format!("    {} {}", style::fg(pal::VIOLET, "❯"), ex));
    }
    lines.push(String::new());
    lines.push(format!("  {}", style::gradient_styled("Keys", 0.5, true)));
    for (k, what) in [("enter", "send"), ("shift/alt+enter  \\+enter", "new line"), ("tab", "complete a command or path"),
                      ("↑ ↓", "history"), ("esc / ctrl+c", "stop the work in progress (and take the mouse back)"), ("!command", "run a shell command"),
                      ("ctrl+l", "clear the screen"), ("ctrl+d", "leave")] {
        lines.push(format!("    {}  {}", Style::new().fg(pal::SKY).paint(&text::pad(k, 26)), style::dim(what)));
    }
    lines.push(format!("    {}", style::faint("/help <group> shows one group: /help screen, /help auto, /help code ...")));
    lines.push(String::new());
    ctx.print(lines);
}

/// Say no, plainly, and what can be done instead.
pub fn decline(ctx: &Ctx, intent: &Intent) {
    let w = ctx.width().saturating_sub(6);
    let (head, body) = match intent {
        Intent::Question => ("No.", "I can't answer questions or explain things in English. I'm a small model grown from scratch on code \
                                   alone, so any English I produced would be guesswork dressed up as an answer - and I won't do that."),
        Intent::Greeting => ("Hi.", "I write code, complete it, and fix it - and I check everything before I hand it over."),
        _ => ("I don't know what you want me to do, so I won't guess.", "Here's what I can do:"),
    };
    let mut lines = vec![String::new(), format!("  {} {}", Style::new().fg(if matches!(intent, Intent::Greeting) { pal::CYAN } else { pal::RED })
                                                    .bold().paint(head), "")];
    for l in text::wrap(body, w) {
        lines.push(format!("  {}", style::dim(&l)));
    }
    lines.push(String::new());
    for (ex, what) in [("write <what it should do>", "new code, checked before it's saved"), ("complete <file>[:line]", "fill in code in a file"),
                       ("fix [command]", "make a failing command or test pass"), ("run <command>", "run programs; /start keeps them running"),
                       ("/help", "everything else")] {
        lines.push(format!("    {} {}  {}", style::fg(pal::VIOLET, "❯"), text::pad(ex, 26), style::faint(what)));
    }
    lines.push(String::new());
    ctx.print(lines);
}

// ----------------------------------------------------------- code actions

fn resolve(ctx: &Ctx, p: &str) -> PathBuf {
    let path = PathBuf::from(p);
    if path.is_absolute() {
        path
    } else {
        ctx.cwd().join(path)
    }
}

fn shown(ctx: &Ctx, p: &Path) -> String {
    let s = ctx.shared.lock().expect("lock");
    match &s.project {
        Some(proj) => proj.rel(p),
        None => p.strip_prefix(&s.cwd).unwrap_or(p).to_string_lossy().to_string(),
    }
}

/// Run a workflow that writes code, streaming the careful candidate.
fn with_assistant<T>(ctx: &Ctx, title: &str, lang: Option<&'static Lang>, f: impl FnOnce(&Assistant, Option<&Project>, &(dyn Fn(Progress) + Sync)) -> Result<T>) -> Result<T> {
    let brain = ctx.brain()?;
    let settings = ctx.settings.clone();
    let names = ctx.names();
    let assistant = Assistant { brain: &brain, settings: &settings, names, cancel: &ctx.cancel };
    ctx.send(Msg::StreamStart { title: title.to_string(), lang: lang.map(|l| l.id) });
    let tx = ctx.clone();
    let streaming = std::sync::atomic::AtomicBool::new(true);
    let progress = move |p: Progress| match p {
        Progress::Reading { tokens, sources } => {
            let from = if sources.is_empty() { String::new() } else { format!(" + {} related snippets", sources.len()) };
            tx.status(format!("Reading {} tokens{from}…", commas(tokens as u64)));
        }
        Progress::Text(t, prob) => {
            if streaming.load(Ordering::Relaxed) {
                tx.send(Msg::Stream(t, prob));
            }
        }
        Progress::Checking { candidate, of, what } => {
            if streaming.swap(false, Ordering::Relaxed) {
                tx.send(Msg::StreamEnd);
            }
            tx.status(format!("Checking candidate {candidate} of {of}: {what}…"));
        }
        Progress::Running { command, attempt } => {
            tx.status(if attempt == 0 { format!("Running {command}…") } else { format!("Trying fix #{attempt}: running {command}…") })
        }
        Progress::Note(n) => tx.status(n),
    };
    // the project index is borrowed for the duration (the screen doesn't need it meanwhile)
    let project = ctx.shared.lock().expect("lock").project.take();
    let result = f(&assistant, project.as_ref(), &progress);
    if let Some(p) = project {
        let mut shared = ctx.shared.lock().expect("lock");
        if shared.project.is_none() {
            shared.project = Some(p);
        }
    }
    ctx.send(Msg::StreamEnd);
    result
}

fn unsure_ranges(c: &crate::assist::Candidate, below: f32) -> Vec<(usize, usize)> {
    c.generation.spans.iter().filter(|s| s.2 < below).map(|s| (s.0, s.1)).collect()
}

/// Show an answer's best candidate and its report; returns whether the user
/// wants it written.
fn present(ctx: &Ctx, answer: &Answer, what: &str) -> Option<usize> {
    let w = ctx.width();
    let settings = &ctx.settings;
    let Some(best) = answer.best() else {
        ctx.print(vec![String::new(), format!("  {} {}", Style::new().fg(pal::RED).bold().paint("No."), "I didn't write anything I could check.")]);
        return None;
    };
    let lang = lang::for_path(&answer.path);
    let lang_name = lang.map_or("its language", |l| l.name);
    let mut lines = vec![String::new()];
    let g = &best.generation;
    let meta = format!("{} tokens · {:.1}s · {:.0} tok/s{}", g.tokens(), g.seconds, g.tokens_per_second(),
                       if g.guessed > 0 { format!(" · {} guessed ahead", g.guessed) } else { String::new() });
    let icon = if best.report.verdict > Verdict::Refused { style::fg(pal::CYAN, "✦") } else { style::fg(pal::RED, "✦") };
    match &answer.original {
        Some(orig) if orig != &best.file => {
            lines.push(format!("  {icon} {} {} {}", Style::new().bold().paint(&answer.shown), style::dim(&format!("· {what} ·")),
                               style::faint(&widgets::change_summary(orig, &best.file))));
            lines.push(format!("    {}", style::faint(&meta)));
            lines.extend(widgets::diff(orig, &best.file, lang.map(|l| l.id), w, 3));
        }
        _ => {
            lines.push(format!("  {icon} {} {} {}", Style::new().bold().paint(&answer.shown), style::dim(&format!("· {what} ·")),
                               style::faint(&format!("{} lines", best.file.lines().count()))));
            lines.push(format!("    {}", style::faint(&meta)));
            let offset = best.file.len() - best.code.len();
            let unsure: Vec<(usize, usize)> = unsure_ranges(best, settings.honesty.uncertain_below).into_iter().map(|(a, b)| (a + offset, b + offset)).collect();
            lines.extend(widgets::code_panel(&best.file, lang.map(|l| l.id), 1, w, &unsure));
        }
    }
    if !answer.sources.is_empty() {
        lines.push(format!("    {} {}", style::faint("read from your project:"), style::faint(&answer.sources.join(", "))));
    }
    lines.extend(widgets::report(&best.report, lang_name, w));
    ctx.print(lines);
    if best.report.verdict > Verdict::Refused {
        ctx.shared.lock().expect("lock").last_code = Some(best.code.clone());
    }
    if best.report.verdict == Verdict::Refused {
        let others = answer.candidates.len().saturating_sub(1);
        ctx.print(vec![format!("    {}", style::dim(&format!(
            "Nothing was written. I wrote {} different candidate{}; none passed. /show shows the best attempt.",
            others + 1, if others == 0 { "" } else { "s" })))]);
        ctx.shared.lock().expect("lock").last_refused = Some(answer.clone());
        return None;
    }
    let verb = if answer.original.is_some() { "Apply this change to" } else { "Write" };
    let mut choices = vec![('y', "yes"), ('n', "no")];
    if answer.candidates.len() > 1 {
        choices.push(('a', "see alternatives"));
    }
    match ctx.ask(&format!("{verb} {}?", answer.shown), &choices) {
        'y' => Some(0),
        'a' => {
            for (i, c) in answer.candidates.iter().enumerate().skip(1) {
                let mut lines = vec![String::new(), format!("  {} {}", style::accent(&format!("Alternative {i}")),
                    style::dim(&format!("· {:?} · {:.0}% certainty", c.report.verdict, c.report.confidence.score() * 100.0)))];
                match &answer.original {
                    Some(orig) => lines.extend(widgets::diff(orig, &c.file, lang.map(|l| l.id), w, 2)),
                    None => lines.extend(widgets::code_panel(&c.file, lang.map(|l| l.id), 1, w, &[])),
                }
                ctx.print(lines);
            }
            let picks: Vec<(char, String)> = answer.candidates.iter().enumerate()
                .filter(|(_, c)| c.report.verdict > Verdict::Refused)
                .map(|(i, _)| (char::from_digit(i as u32 % 10, 10).unwrap_or('0'), if i == 0 { "0 (the first)".to_string() } else { i.to_string() }))
                .collect();
            let mut opts: Vec<(char, &str)> = picks.iter().map(|(k, l)| (*k, l.as_str())).collect();
            opts.push(('n', "none"));
            let k = ctx.ask("Which one?", &opts);
            k.to_digit(10).map(|d| d as usize).filter(|&d| d < answer.candidates.len())
        }
        _ => None,
    }
}

fn apply(ctx: &Ctx, answer: &Answer, pick: usize, label: &str) -> Result<()> {
    let c = &answer.candidates[pick];
    let mut s = ctx.shared.lock().expect("lock");
    match s.project.as_mut() {
        Some(p) => p.write_file(&answer.path, &c.file, label)?,
        None => std::fs::write(&answer.path, &c.file)?,
    }
    drop(s);
    ctx.print(vec![format!("  {} {} {}", style::fg(pal::GREEN, "✓"), if answer.original.is_some() { "Updated" } else { "Wrote" },
                           Style::new().bold().paint(&answer.shown)), format!("    {}", style::faint("/undo puts it back"))]);
    Ok(())
}

/// Default language for new code: the project's most common one.
fn project_language(ctx: &Ctx) -> Option<&'static Lang> {
    let s = ctx.shared.lock().expect("lock");
    let p = s.project.as_ref()?;
    let mut counts: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    for f in &p.files {
        if let Some(l) = lang::for_path(Path::new(f)).filter(|l| !matches!(l.id, "markdown" | "json" | "yaml" | "toml")) {
            *counts.entry(l.name).or_default() += 1;
        }
    }
    let best = counts.into_iter().max_by_key(|(_, n)| *n)?.0;
    lang::LANGS.iter().find(|l| l.name == best)
}

pub fn write(ctx: &Ctx, path: Option<&str>, lang_hint: Option<&'static Lang>, description: &str) -> Result<()> {
    let lang = path.and_then(|p| lang::for_path(Path::new(p))).or(lang_hint).or_else(|| project_language(ctx)).or_else(|| lang::by_name("python"));
    let lang = lang.expect("python is a known language");
    let rel = match path {
        Some(p) => p.to_string(),
        None => crate::prompt::suggest_path(description, lang),
    };
    let full = resolve(ctx, &rel);
    let shown_path = shown(ctx, &full);
    if let Ok(existing) = std::fs::read_to_string(&full) {
        return extend(ctx, &full, &shown_path, lang, &existing, description);
    }
    let answer = with_assistant(ctx, &shown_path, Some(lang), |a, project, progress| {
        a.write_new(project, &full, &shown_path, description, progress)
    })?;
    if ctx.cancelled() {
        ctx.line(format!("  {}", style::dim("Stopped.")));
        return Ok(());
    }
    if let Some(pick) = present(ctx, &answer, if answer.original.is_some() { "rewrite" } else { "new file" }) {
        apply(ctx, &answer, pick, "write")?;
    }
    Ok(())
}

/// Add to an existing file: a comment saying what's wanted goes at its
/// end, and the model writes what follows it.
fn extend(ctx: &Ctx, full: &Path, shown_path: &str, lang: &'static Lang, existing: &str, description: &str) -> Result<()> {
    let sep = if existing.is_empty() || existing.ends_with("\n\n") { "" } else if existing.ends_with('\n') { "\n" } else { "\n\n" };
    let text = format!("{existing}{sep}{}\n", lang.comment(description.trim()));
    let at = text.len();
    let mut answer = with_assistant(ctx, shown_path, Some(lang), |a, project, progress| a.complete(project, full, shown_path, &text, at, progress))?;
    answer.original = Some(existing.to_string());
    if ctx.cancelled() {
        ctx.line(format!("  {}", style::dim("Stopped.")));
        return Ok(());
    }
    if let Some(pick) = present(ctx, &answer, "addition") {
        apply(ctx, &answer, pick, "write")?;
    }
    Ok(())
}

/// Where to fill in: "file:line" (or just "file": a TODO line, else the end).
fn completion_point(text: &str, line: Option<usize>) -> (usize, usize) {
    let starts: Vec<usize> = std::iter::once(0).chain(text.match_indices('\n').map(|(i, _)| i + 1)).collect();
    let is_placeholder = |l: &str| {
        let t = l.trim();
        t.is_empty() || matches!(t, "..." | "pass" | "todo!()" | "unimplemented!()" | "TODO") || t.contains("TODO") || t.contains("FIXME")
    };
    let line_text = |n: usize| -> &str {
        let a = starts.get(n - 1).copied().unwrap_or(text.len());
        let b = starts.get(n).copied().unwrap_or(text.len());
        &text[a..b]
    };
    let target = line.or_else(|| (1..=starts.len()).find(|&n| {
        let l = line_text(n);
        l.contains("TODO") || l.contains("FIXME") || l.trim() == "todo!()"
    }));
    match target {
        Some(n) if n >= 1 && n <= starts.len() => {
            let a = starts[n - 1];
            let b = starts.get(n).copied().unwrap_or(text.len());
            if is_placeholder(line_text(n)) {
                (a, b) // replace the placeholder line
            } else {
                (b, b) // continue after the line
            }
        }
        _ => (text.len(), text.len()),
    }
}

pub fn complete(ctx: &Ctx, target: &str) -> Result<()> {
    let (file, line) = match target.rsplit_once(':') {
        Some((f, l)) if l.parse::<usize>().is_ok() => (f, l.parse().ok()),
        _ => (target, None),
    };
    let full = resolve(ctx, file);
    let text = std::fs::read_to_string(&full).map_err(|e| anyhow::anyhow!("can't read {file}: {e}"))?;
    let (at, cut) = completion_point(&text, line);
    let stripped = format!("{}{}", &text[..at], &text[cut..]);
    let shown_path = shown(ctx, &full);
    let lang = lang::for_path(&full);
    let mut answer = with_assistant(ctx, &format!("{shown_path}:{}", stripped[..at].matches('\n').count() + 1), lang, |a, project, progress| {
        a.complete(project, &full, &shown_path, &stripped, at, progress)
    })?;
    answer.original = Some(text);
    if ctx.cancelled() {
        ctx.line(format!("  {}", style::dim("Stopped.")));
        return Ok(());
    }
    if let Some(pick) = present(ctx, &answer, "completion") {
        apply(ctx, &answer, pick, "complete")?;
    }
    Ok(())
}

pub fn fix(ctx: &Ctx, command: Option<&str>) -> Result<()> {
    let root = {
        let s = ctx.shared.lock().expect("lock");
        s.project.as_ref().map(|p| p.root.clone()).unwrap_or_else(|| s.cwd.clone())
    };
    let command = match command {
        Some(c) => c.to_string(),
        None => match ide::test_command(&root) {
            Some(c) => c,
            None => bail!("I couldn't find this project's tests. Tell me what to run: /fix <command>"),
        },
    };
    ctx.print(vec![String::new(), format!("  {} Making {} pass", style::fg(pal::VIOLET, "✦"), Style::new().fg(pal::SKY).paint(&format!("`{command}`")))]);
    let brain = ctx.brain()?;
    let names = ctx.names();
    let settings = ctx.settings.clone();
    let assistant = Assistant { brain: &brain, settings: &settings, names, cancel: &ctx.cancel };
    let c2 = ctx.clone();
    let progress = move |p: Progress| match p {
        Progress::Running { command, attempt } => c2.status(if attempt == 0 { format!("Running {command}…") } else { format!("Testing fix #{attempt}…") }),
        Progress::Note(n) => c2.status(n),
        Progress::Checking { candidate, of, what } => c2.status(format!("Checking candidate {candidate}/{of}: {what}…")),
        _ => {}
    };
    let mut project = {
        let mut s = ctx.shared.lock().expect("lock");
        s.project.take().unwrap_or_else(|| Project::open(&root))
    };
    let outcome = assistant.fix(&mut project, &command, &progress);
    ctx.shared.lock().expect("lock").project = Some(project);
    let w = ctx.width();
    let tail = |out: &proc::Output| -> Vec<String> {
        let all = out.combined();
        let lines: Vec<&str> = all.lines().collect();
        lines[lines.len().saturating_sub(12)..].iter().map(|l| format!("    {} {}", style::faint("│"), style::dim(&text::truncate(l, w - 8)))).collect()
    };
    match outcome? {
        FixOutcome::AlreadyPassing(o) => ctx.print(vec![format!("  {} It already passes ({:.1}s). Nothing to fix.", style::fg(pal::GREEN, "✓"), o.seconds)]),
        FixOutcome::NoLocation(o) => {
            let mut lines = vec![format!("  {} {} It fails, but the error doesn't point at a line of your code, so I won't guess where to change it.",
                                         Style::new().fg(pal::RED).bold().paint("No."), "")];
            lines.extend(tail(&o));
            ctx.print(lines);
        }
        FixOutcome::Cancelled { tried } => ctx.line(format!("  {} after {tried} attempt(s); every file is as it was.", style::dim("Stopped"))),
        FixOutcome::NotFixed { tried, output, places } => {
            let mut lines = vec![format!("  {} I couldn't find a fix that makes it pass.", Style::new().fg(pal::RED).bold().paint("No."))];
            lines.push(format!("    {}", style::dim(&format!("I tried {tried} candidate change{} around {}; each was tested and put back. Your files are unchanged.",
                if tried == 1 { "" } else { "s" },
                places.iter().take(3).map(|(p, l)| format!("{}:{l}", shown(ctx, p))).collect::<Vec<_>>().join(", ")))));
            lines.push(format!("    {}", style::faint("The failure:")));
            lines.extend(tail(&output));
            ctx.print(lines);
        }
        FixOutcome::Fixed { path, shown: rel, before, candidate, output } => {
            let lang = lang::for_path(&path);
            let mut lines = vec![String::new(), format!("  {} {} now passes ({:.1}s) after this change to {}:", style::fg(pal::GREEN, "✓"),
                Style::new().fg(pal::SKY).paint(&format!("`{command}`")), output.seconds, Style::new().bold().paint(&rel))];
            lines.extend(widgets::diff(&before, &candidate.file, lang.map(|l| l.id), w, 3));
            lines.push(format!("    {}", style::dim("Tests passing means this change is consistent with them - not that it's what you meant. Review it; /undo puts it back.")));
            ctx.print(lines);
        }
    }
    Ok(())
}

pub fn show_refused(ctx: &Ctx) {
    let last = ctx.shared.lock().expect("lock").last_refused.clone();
    let Some(a) = last else {
        ctx.line(format!("  {}", style::dim("Nothing refused this session.")));
        return;
    };
    let Some(best) = a.best() else { return };
    let lang = lang::for_path(&a.path);
    let mut lines = vec![String::new(), format!("  {} {} {}", style::fg(pal::RED, "✦"), Style::new().bold().paint(&a.shown),
        Style::new().fg(pal::RED).bold().paint("· UNVERIFIED - I don't stand behind this"))];
    let offset = best.file.len() - best.code.len();
    let unsure: Vec<(usize, usize)> = unsure_ranges(best, ctx.settings.honesty.uncertain_below).into_iter().map(|(x, y)| (x + offset, y + offset)).collect();
    lines.extend(widgets::code_panel(&best.file, lang.map(|l| l.id), 1, ctx.width(), &unsure));
    lines.extend(widgets::report(&best.report, lang.map_or("its language", |l| l.name), ctx.width()));
    ctx.print(lines);
}

// -------------------------------------------------------------- workbench

/// Run a shell command in the project, streaming its output.
pub fn run(ctx: &Ctx, command: &str) -> Result<()> {
    let cwd = ctx.cwd();
    let as_file = resolve(ctx, command.trim());
    let command = if as_file.is_file() && !command.contains(' ') {
        match ide::run_file_command(&as_file) {
            Some(c) => c,
            None if as_file.extension().is_some_and(|e| e == "html" || e == "htm") => return preview(ctx, command),
            None => bail!("I don't know how to run {} (is its language's tool installed?)", command.trim()),
        }
    } else {
        command.to_string()
    };
    let w = ctx.width();
    ctx.print(vec![String::new(), format!("  {} {}", style::fg(pal::VIOLET, "⏵"), Style::new().fg(pal::SKY).bold().paint(&command))]);
    ctx.status(format!("Running {command}…"));
    let mut shown_lines = 0usize;
    let mut all: Vec<String> = Vec::new();
    let mut cmd = proc::shell(&command);
    cmd.current_dir(&cwd);
    let out = proc::run_streaming(cmd, None, Some(&ctx.cancel), &mut |stream, line| {
        all.push(line.to_string());
        if shown_lines < 400 {
            let body = text::truncate(&text::strip_ansi(line), w.saturating_sub(8));
            let body = if stream == proc::Stream::Err { Style::new().fg(pal::DEL_FG).paint(&body) } else { body };
            ctx.line(format!("    {} {body}", style::faint("│")));
            shown_lines += 1;
            if shown_lines == 400 {
                ctx.line(format!("    {}", style::faint("… more output hidden (it keeps running; esc stops it)")));
            }
        }
    })?;
    let status = if out.cancelled {
        style::fg(pal::YELLOW, "■ stopped")
    } else if out.ok() {
        style::fg(pal::GREEN, "✓ exit 0")
    } else {
        style::fg(pal::RED, &format!("✗ exit {}", out.code.map_or("?".into(), |c| c.to_string())))
    };
    ctx.print(vec![format!("    {status} {}", style::faint(&format!("· {:.2}s · {} lines", out.seconds, all.len())))]);
    {
        let mut s = ctx.shared.lock().expect("lock");
        s.last_output = all;
        s.last_exit_ok = Some(out.ok() && !out.cancelled);
    }
    Ok(())
}

pub fn test(ctx: &Ctx) -> Result<()> {
    let root = {
        let s = ctx.shared.lock().expect("lock");
        s.project.as_ref().map(|p| p.root.clone()).unwrap_or_else(|| s.cwd.clone())
    };
    match ide::test_command(&root) {
        Some(c) => run(ctx, &c),
        None => bail!("I couldn't find this project's tests (no Cargo.toml, package.json test script, go.mod, pytest setup or Makefile test target)"),
    }
}

pub fn start(ctx: &Ctx, command: &str) -> Result<()> {
    let cwd = ctx.cwd();
    let id = ctx.shared.lock().expect("lock").jobs.start(command, &cwd)?;
    ctx.line(format!("  {} job {} started: {}", style::fg(pal::GREEN, "▶"), Style::new().bold().paint(&id.to_string()), Style::new().fg(pal::SKY).paint(command)));
    // give a server a moment to say where it's listening
    for _ in 0..30 {
        std::thread::sleep(Duration::from_millis(100));
        let s = ctx.shared.lock().expect("lock");
        let Some(job) = s.jobs.get(id) else { break };
        if let Some(url) = job.url() {
            ctx.line(format!("    {} {}   {}", style::faint("listening at"), Style::new().fg(pal::CYAN).underline().paint(&url),
                             style::faint(&format!("/preview {id} to look at it"))));
            break;
        }
        if let JobState::Exited(code) = job.state() {
            ctx.line(format!("    {} it exited already (code {}). /logs {id} shows why.", style::fg(pal::YELLOW, "!"), code.map_or("?".into(), |c| c.to_string())));
            break;
        }
    }
    Ok(())
}

pub fn jobs(ctx: &Ctx) {
    let s = ctx.shared.lock().expect("lock");
    if s.jobs.list.is_empty() && s.servers.is_empty() {
        ctx.line(format!("  {}", style::dim("No background programs. /start <command> runs one; /serve serves a website folder.")));
        return;
    }
    let mut lines = vec![String::new()];
    for j in &s.jobs.list {
        let (icon, state) = match j.state() {
            JobState::Running => (style::fg(pal::GREEN, "●"), format!("running {}", human_duration(j.started.elapsed().as_secs_f64()))),
            JobState::Exited(Some(0)) => (style::fg(pal::FAINT, "○"), "finished".to_string()),
            JobState::Exited(c) => (style::fg(pal::RED, "○"), format!("exited ({})", c.map_or("killed".into(), |c| c.to_string()))),
        };
        lines.push(format!("  {icon} {}  {}  {}  {}", Style::new().bold().paint(&format!("{:>2}", j.id)), text::pad(&state, 22),
                           Style::new().fg(pal::SKY).paint(&j.command), j.url().map(|u| style::fg(pal::CYAN, &u)).unwrap_or_default()));
    }
    for srv in &s.servers {
        lines.push(format!("  {} {}  {}  {}", style::fg(pal::GREEN, "●"), Style::new().bold().paint(" ~"), text::pad("serving", 22),
                           style::fg(pal::CYAN, &format!("{} → {}", srv.url, srv.dir.display()))));
    }
    ctx.print(lines);
}

fn job_id(arg: &str) -> Result<usize> {
    arg.trim().trim_start_matches('#').parse().map_err(|_| anyhow::anyhow!("which job? (a number from /jobs)"))
}

pub fn logs(ctx: &Ctx, arg: &str) -> Result<()> {
    let id = job_id(arg)?;
    let s = ctx.shared.lock().expect("lock");
    let job = s.jobs.get(id).ok_or_else(|| anyhow::anyhow!("no job {id}"))?;
    let w = ctx.width();
    let mut lines = vec![format!("  {} {}", style::dim(&format!("job {id}:")), Style::new().fg(pal::SKY).paint(&job.command))];
    let tail = job.tail(40);
    if tail.is_empty() {
        lines.push(format!("    {}", style::faint("(no output yet)")));
    }
    lines.extend(tail.iter().map(|l| format!("    {} {}", style::faint("│"), text::truncate(l, w - 8))));
    ctx.print(lines);
    Ok(())
}

pub fn stop(ctx: &Ctx, arg: &str) -> Result<()> {
    let mut s = ctx.shared.lock().expect("lock");
    if arg.trim() == "all" {
        s.jobs.stop_all();
        s.servers.clear();
        ctx.line(format!("  {} stopped every background program", style::fg(pal::YELLOW, "■")));
        return Ok(());
    }
    let id = job_id(arg)?;
    s.jobs.get(id).ok_or_else(|| anyhow::anyhow!("no job {id}"))?.stop();
    ctx.line(format!("  {} stopped job {id}", style::fg(pal::YELLOW, "■")));
    Ok(())
}

pub fn serve(ctx: &Ctx, arg: &str) -> Result<()> {
    let dir = if arg.trim().is_empty() { ctx.cwd() } else { resolve(ctx, arg.trim()) };
    let server = ide::serve(&dir)?;
    ctx.print(vec![format!("  {} serving {} at {}", style::fg(pal::GREEN, "●"), dir.display(), Style::new().fg(pal::CYAN).underline().paint(&server.url)),
                   format!("    {}", style::faint(&format!("open it in your browser, or /preview {} to look at it here", server.url)))]);
    ctx.shared.lock().expect("lock").servers.push(server);
    Ok(())
}

pub fn preview(ctx: &Ctx, arg: &str) -> Result<()> {
    let arg = arg.trim();
    if arg.is_empty() {
        bail!("preview what? A URL, an .html file, or a job number");
    }
    let url = match arg.parse::<usize>() {
        Ok(id) => {
            let s = ctx.shared.lock().expect("lock");
            let job = s.jobs.get(id).ok_or_else(|| anyhow::anyhow!("no job {id}"))?;
            job.url().ok_or_else(|| anyhow::anyhow!("job {id} hasn't printed a local address to look at"))?
        }
        Err(_) => ide::to_url(arg, &ctx.cwd()),
    };
    ctx.status(format!("Opening {url} in a headless browser…"));
    let shot = ide::screenshot(&url, 1280, 800, 2000)?;
    let w = ctx.width().saturating_sub(4).min(120);
    let mut lines = vec![String::new(), format!("  {} {}", style::fg(pal::VIOLET, "◳"), Style::new().fg(pal::CYAN).underline().paint(&url))];
    lines.extend(ide::render_png(&shot.png, w, 40)?.into_iter().map(|l| format!("  {l}")));
    if shot.console.is_empty() {
        lines.push(format!("  {}", style::faint("console: quiet")));
    } else {
        lines.push(format!("  {}", style::dim("console:")));
        for l in &shot.console {
            let bad = l.contains("Uncaught") || l.contains("Error") || l.contains("error");
            lines.push(format!("    {} {}", if bad { style::fg(pal::RED, "✗") } else { style::faint("·") }, if bad { style::fg(pal::DEL_FG, l) } else { style::dim(l) }));
        }
    }
    lines.push(format!("  {}", style::faint(&format!("full-size screenshot: {}", shot.png.display()))));
    ctx.print(lines);
    Ok(())
}

pub fn open(ctx: &Ctx, arg: &str) -> Result<()> {
    let (file, range) = match arg.trim().rsplit_once(':') {
        Some((f, r)) if r.chars().next().is_some_and(|c| c.is_ascii_digit()) => (f, Some(r)),
        _ => (arg.trim(), None),
    };
    let full = resolve(ctx, file);
    let text = std::fs::read_to_string(&full).map_err(|e| anyhow::anyhow!("can't read {file}: {e}"))?;
    let total = text.lines().count();
    let (a, b) = match range.map(|r| r.split_once('-').unwrap_or((r, r))) {
        Some((a, b)) => (a.parse().unwrap_or(1usize).max(1), b.parse().unwrap_or(total).min(total)),
        None => (1, total.min(400)),
    };
    let slice: String = text.lines().skip(a - 1).take(b.saturating_sub(a) + 1).map(|l| format!("{l}\n")).collect();
    let lang = lang::for_path(&full);
    let mut lines = vec![String::new(), format!("  {} {} {}", style::fg(pal::VIOLET, "▤"), Style::new().bold().paint(&shown(ctx, &full)),
                                                style::faint(&format!("· lines {a}-{b} of {total}")))];
    lines.extend(widgets::code_panel(&slice, lang.map(|l| l.id), a, ctx.width(), &[]));
    ctx.print(lines);
    Ok(())
}

fn glob_regex(pattern: &str) -> regex::Regex {
    let mut re = String::from("(?i)");
    let anchored = pattern.contains('/');
    if !anchored {
        re.push_str("(^|/)");
    } else {
        re.push('^');
    }
    for c in pattern.chars() {
        match c {
            '*' => re.push_str("[^/]*"),
            '?' => re.push_str("[^/]"),
            c => re.push_str(&regex::escape(&c.to_string())),
        }
    }
    if !pattern.contains('*') && !pattern.contains('?') && !anchored {
        // a plain word finds files whose name contains it
        re = format!("(?i){}[^/]*$", regex::escape(pattern));
    } else {
        re.push('$');
    }
    regex::Regex::new(&re.replace("[^/]*[^/]*", ".*")).unwrap_or_else(|_| regex::Regex::new("$^").expect("valid"))
}

pub fn find(ctx: &Ctx, pattern: &str) -> Result<()> {
    let s = ctx.shared.lock().expect("lock");
    let p = s.project.as_ref().ok_or_else(|| anyhow::anyhow!("the project is still being read; try again in a moment"))?;
    let re = glob_regex(pattern.trim());
    let hits: Vec<&String> = p.files.iter().filter(|f| re.is_match(f)).collect();
    let mut lines: Vec<String> = hits.iter().take(200).map(|f| format!("    {}", Style::new().fg(pal::SKY).paint(f))).collect();
    lines.insert(0, format!("  {} {}", style::dim(&format!("{} file{} match", hits.len(), if hits.len() == 1 { "" } else { "s" })),
                            Style::new().bold().paint(pattern.trim())));
    ctx.print(lines);
    Ok(())
}

pub fn grep(ctx: &Ctx, pattern: &str) -> Result<()> {
    let s = ctx.shared.lock().expect("lock");
    let p = s.project.as_ref().ok_or_else(|| anyhow::anyhow!("the project is still being read; try again in a moment"))?;
    let re = regex::RegexBuilder::new(pattern.trim()).case_insensitive(!pattern.chars().any(|c| c.is_uppercase())).build()
        .or_else(|_| regex::Regex::new(&regex::escape(pattern.trim())))?;
    let w = ctx.width();
    let mut lines = Vec::new();
    let mut n = 0;
    for f in &p.files {
        let Some(text) = p.content(f) else { continue };
        let mut first = true;
        for (i, line) in text.lines().enumerate() {
            let Some(m) = re.find(line) else { continue };
            n += 1;
            if n > 200 {
                continue;
            }
            if first {
                lines.push(format!("  {}", Style::new().fg(pal::SKY).bold().paint(f)));
                first = false;
            }
            let hl = format!("{}{}{}", &line[..m.start()], Style::new().fg(pal::YELLOW).bold().underline().paint(m.as_str()), &line[m.end()..]);
            lines.push(text::truncate(&format!("    {} {}", style::faint(&format!("{:>5}", i + 1)), hl.trim_start()), w));
        }
    }
    lines.insert(0, format!("  {} {}", style::dim(&format!("{n} match{} for", if n == 1 { "" } else { "es" })), Style::new().bold().paint(pattern.trim())));
    if n > 200 {
        lines.push(format!("  {}", style::faint(&format!("… and {} more", n - 200))));
    }
    ctx.print(lines);
    Ok(())
}

pub fn tree(ctx: &Ctx, arg: &str) -> Result<()> {
    let s = ctx.shared.lock().expect("lock");
    let p = s.project.as_ref().ok_or_else(|| anyhow::anyhow!("the project is still being read; try again in a moment"))?;
    let prefix = arg.trim().trim_matches('/');
    let files: Vec<&String> = p.files.iter().filter(|f| prefix.is_empty() || f.starts_with(&format!("{prefix}/"))).collect();
    let mut lines = vec![format!("  {} {}", style::fg(pal::VIOLET, "▾"), Style::new().bold().paint(&format!("{}/{}",
        p.root.file_name().map_or(String::new(), |n| n.to_string_lossy().to_string()), prefix)))];
    let mut shown_dirs: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for f in &files {
        let rel = f.strip_prefix(&format!("{prefix}/")).unwrap_or(f);
        let parts: Vec<&str> = rel.split('/').collect();
        let depth = parts.len() - 1;
        if depth >= 3 {
            *shown_dirs.entry(parts[..3].join("/")).or_default() += 1;
            continue;
        }
        let dir = parts[..depth].join("/");
        if depth > 0 && !shown_dirs.contains_key(&dir) {
            for d in 1..=depth {
                let sub = parts[..d].join("/");
                if !shown_dirs.contains_key(&sub) {
                    shown_dirs.insert(sub.clone(), 0);
                    lines.push(format!("  {}{} {}", "  ".repeat(d), style::fg(pal::VIOLET, "▾"), Style::new().fg(pal::BLUE).bold().paint(parts[d - 1])));
                }
            }
        }
        lines.push(format!("  {}{} {}", "  ".repeat(depth + 1), style::faint("·"), parts[depth]));
        if lines.len() > 300 {
            lines.push(format!("  {}", style::faint(&format!("… {} files in all", files.len()))));
            break;
        }
    }
    ctx.print(lines);
    Ok(())
}

pub fn diff(ctx: &Ctx) {
    let s = ctx.shared.lock().expect("lock");
    let Some(p) = s.project.as_ref() else { return };
    if p.history().is_empty() {
        ctx.line(format!("  {}", style::dim("Aegist hasn't changed anything this session.")));
        return;
    }
    let mut firsts: Vec<(&Path, Option<&String>)> = Vec::new();
    for e in p.history() {
        if !firsts.iter().any(|(path, _)| *path == e.path.as_path()) {
            firsts.push((&e.path, e.before.as_ref()));
        }
    }
    let mut lines = Vec::new();
    for (path, before) in firsts {
        let now = std::fs::read_to_string(path).unwrap_or_default();
        let before = before.cloned().unwrap_or_default();
        let lang = lang::for_path(path);
        lines.push(String::new());
        lines.push(format!("  {} {} {}", style::fg(pal::VIOLET, "✦"), Style::new().bold().paint(&p.rel(path)), style::faint(&widgets::change_summary(&before, &now))));
        lines.extend(widgets::diff(&before, &now, lang.map(|l| l.id), ctx.width(), 3));
    }
    ctx.print(lines);
}

pub fn undo(ctx: &Ctx) -> Result<()> {
    let mut s = ctx.shared.lock().expect("lock");
    let p = s.project.as_mut().ok_or_else(|| anyhow::anyhow!("nothing to undo"))?;
    match p.undo()? {
        Some(e) => {
            let what = if e.before.is_none() { "Removed" } else { "Restored" };
            let rel = p.rel(&e.path);
            drop(s);
            ctx.line(format!("  {} {what} {} {}", style::fg(pal::YELLOW, "↶"), Style::new().bold().paint(&rel), style::faint(&format!("(undid: {})", e.label))));
        }
        None => ctx.line(format!("  {}", style::dim("Nothing to undo."))),
    }
    Ok(())
}

// ----------------------------------------------------------------- model

pub fn learn(ctx: &Ctx, arg: &str) -> Result<()> {
    let arg = arg.trim();
    let settings = ctx.settings.clone();
    let mut log = |m: String| ctx.status(m);
    let reports = if let Some(name) = arg.strip_prefix("--pack").map(str::trim) {
        let Some(pack) = learn::pack(name) else {
            let names: Vec<&str> = learn::PACKS.iter().map(|p| p.name).collect();
            bail!("no pack named {name:?}; there are: {}", names.join(", "));
        };
        let mut out = Vec::new();
        for repo in pack.repos {
            if ctx.cancelled() {
                break;
            }
            match learn::learn_git(&settings, repo, &mut log) {
                Ok(r) => out.push(r),
                Err(e) => ctx.line(format!("  {} {repo}: {e}", style::fg(pal::YELLOW, "!"))),
            }
        }
        out
    } else if arg.is_empty() {
        let mut lines = vec![String::new(), format!("  {}", style::gradient_styled("Packs of code to learn from", 0.2, true))];
        for p in learn::PACKS {
            lines.push(format!("    {}  {}", Style::new().fg(pal::VIOLET).bold().paint(&text::pad(p.name, 12)), style::dim(p.about)));
        }
        lines.push(format!("  {}", style::faint("/learn --pack <name> downloads one (with git). /learn <folder or git URL> adds your own code.")));
        let learned = learn::sources(&settings);
        if !learned.is_empty() {
            lines.push(String::new());
            lines.push(format!("  {}", style::dim("Already learned:")));
            for (src, files, bytes) in learned.iter().take(30) {
                lines.push(format!("    {}  {}", text::pad(src, 24), style::faint(&format!("{files} files · {:.1} MB", *bytes as f64 / 1e6))));
            }
        }
        ctx.print(lines);
        return Ok(());
    } else if learn::is_git_url(arg) {
        vec![learn::learn_git(&settings, arg, &mut log)?]
    } else {
        vec![learn::learn_path(&settings, &resolve(ctx, arg), None, &mut log)?]
    };
    let mut lines = Vec::new();
    for r in &reports {
        lines.push(format!("  {} {} {}", style::fg(pal::GREEN, "✓"), Style::new().bold().paint(&r.source),
            style::dim(&format!("{} files · {:.1} MB · {} names{}{}", r.files, r.bytes as f64 / 1e6, commas(r.names as u64),
                if r.skipped > 0 { format!(" · {} skipped (generated, minified or not code)", r.skipped) } else { String::new() },
                if r.duplicates > 0 { format!(" · {} duplicates", r.duplicates) } else { String::new() }))));
    }
    lines.push(format!("    {}", style::faint("The model learns it when it trains: /train <hours>")));
    ctx.print(lines);
    ctx.shared.lock().expect("lock").learned_names = std::sync::Arc::new(learn::known_names(&settings));
    Ok(())
}

pub fn train(ctx: &Ctx, arg: &str) -> Result<()> {
    let hours: f64 = if arg.trim().is_empty() { 1.0 } else { arg.trim().parse().map_err(|_| anyhow::anyhow!("/train <hours>, like /train 0.5"))? };
    let settings = ctx.settings.clone();
    ctx.print(vec![String::new(), format!("  {} Training for {} {}", style::fg(pal::VIOLET, "✦"), human_duration(hours * 3600.0),
                                          style::faint("· esc or ctrl+c stops and saves"))]);
    let losses = std::sync::Mutex::new(Vec::<f32>::new());
    let mut log = |m: String| {
        if m.starts_with("step") {
            if let Some(l) = m.split("loss ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|v| v.parse::<f32>().ok()) {
                losses.lock().expect("lock").push(l);
            }
            let spark = widgets::sparkline(&losses.lock().expect("lock"), 24);
            ctx.status(format!("{}  {spark}", m.replace(" | ", " · ")));
        } else {
            ctx.line(format!("    {}", style::dim(&m)));
        }
    };
    let report = trainer::train(&settings, Budget::Minutes(hours * 60.0), Size::ForHours(hours), &mut log, None, &ctx.cancel)?;
    let st = &report.stats;
    ctx.print(vec![format!("  {} {} {}", style::fg(pal::GREEN, "✓"), if report.interrupted { "Stopped and saved." } else { "Done." },
        style::dim(&format!("{} steps this session · {:.1}M tokens read in all · held-out loss {:.3} ({:.3} bits per byte)",
            commas(report.steps_this_session), st.tokens_seen as f64 / 1e6, st.val_loss.unwrap_or(f32::NAN), st.val_bpb.unwrap_or(f32::NAN))))]);
    crate::session::load_brain(settings, ctx.shared.clone());
    Ok(())
}

pub fn model(ctx: &Ctx) {
    let settings = &ctx.settings;
    let mut lines = vec![String::new()];
    let Some(meta) = checkpoint::load_meta(settings) else {
        lines.push(format!("  {} {}", style::fg(pal::YELLOW, "◇"), "No trained model yet."));
        lines.push(format!("    {}", style::dim("/learn --pack python (or your own code), then /train 1")));
        ctx.print(lines);
        return;
    };
    let c = &meta.config;
    let params = crate::model::Layout::new(c).total as f64;
    let st = &meta.stats;
    let row = |k: &str, v: String| format!("    {}  {v}", Style::new().fg(pal::VIOLET).paint(&text::pad(k, 12)));
    lines.push(format!("  {}", style::gradient_styled("The model", 0.1, true)));
    lines.push(row("shape", format!("{} layers × {} wide, {} heads · {:.1}M parameters", c.n_layer, c.d_model, c.n_head, params / 1e6)));
    lines.push(row("context", format!("{} tokens trained · reads up to {} ({}× stretched positions)", commas(c.block_size as u64),
        commas((c.block_size * settings.inference.context_scale) as u64), settings.inference.context_scale)));
    lines.push(row("vocabulary", format!("{} tokens, learned from its own corpus", commas(c.vocab_size as u64))));
    lines.push(row("weights", format!("{} for writing code", settings.inference.precision)));
    lines.push(row("trained", format!("{} steps · {:.1}M tokens read · {}", commas(st.steps), st.tokens_seen as f64 / 1e6, human_duration(st.train_seconds))));
    let target = trainer::TOKENS_PER_PARAM * params;
    lines.push(row("progress", format!("{} {:.0}% of \"trained well\" (~20 tokens per parameter)", widgets::bar((st.tokens_seen as f64 / target) as f32, 16),
        (st.tokens_seen as f64 / target * 100.0).min(999.0))));
    if let (Some(v), Some(b)) = (st.val_loss, st.val_bpb) {
        let hist: Vec<f32> = st.history.iter().map(|h| h.val_loss).collect();
        lines.push(row("held-out", format!("loss {v:.3} · {b:.3} bits per byte  {}", widgets::sparkline(&hist, 30))));
    }
    let learned = learn::sources(settings);
    let bytes: u64 = learned.iter().map(|s| s.2).sum();
    lines.push(row("learned", format!("{} sources · {:.1} MB of code", learned.len(), bytes as f64 / 1e6)));
    lines.push(String::new());
    lines.push(format!("  {}", style::gradient_styled("What it can and can't do", 0.4, true)));
    for l in text::wrap("It's a code model trained from scratch on this machine, so it's far smaller than the AIs big labs train on thousands of \
                         GPUs. It completes and writes code in the styles it has read, and gets better with more code and more training. \
                         It doesn't understand English requests the way a chatbot does, it can't explain or answer questions, and it \
                         makes mistakes - which is why every answer is checked (syntax, names, tests, certainty) and refused when the \
                         checks fail.", ctx.width() - 8) {
        lines.push(format!("    {}", style::dim(&l)));
    }
    ctx.print(lines);
}

/// The welcome card shown when a session starts.
pub fn welcome(ctx: &Ctx) -> Vec<String> {
    let settings = &ctx.settings;
    let w = ctx.width();
    let mut lines = vec![String::new()];
    lines.extend(widgets::banner(w));
    lines.push(String::new());
    lines.push(format!("  {}  {}", style::gradient_styled("a coding AI grown from scratch", 0.0, true), style::faint(&format!("v{}", env!("CARGO_PKG_VERSION")))));
    lines.push(format!("  {}", style::dim("its own transformer and tokenizer, trained on code alone · no borrowed brains")));
    lines.push(String::new());
    let mut card = Vec::new();
    let key = |k: &str| Style::new().fg(pal::VIOLET).bold().paint(&text::pad(k, 9));
    let dot = |ok: bool| if ok { style::fg(pal::CYAN, "◆") } else { style::fg(pal::YELLOW, "◇") };
    match checkpoint::load_meta(settings) {
        Some(meta) => {
            let c = &meta.config;
            let params = crate::model::Layout::new(c).total as f64;
            card.push(format!("{} {} {}×{} transformer · {:.1}M params · {} weights", dot(true), key("model"), c.n_layer, c.d_model, params / 1e6,
                              settings.inference.precision));
            card.push(format!("{} {} {} tokens · reads up to {}", dot(true), key("context"), commas(c.block_size as u64),
                              commas((c.block_size * settings.inference.context_scale) as u64)));
            let target = trainer::TOKENS_PER_PARAM * params;
            card.push(format!("{} {} {:.1}M tokens read · {} {:.0}%", dot(meta.stats.tokens_seen as f64 >= target), key("trained"),
                              meta.stats.tokens_seen as f64 / 1e6, widgets::bar((meta.stats.tokens_seen as f64 / target) as f32, 10),
                              (meta.stats.tokens_seen as f64 / target * 100.0).min(999.0)));
        }
        None => card.push(format!("{} {} {}", dot(false), key("model"), style::fg(pal::YELLOW, "not trained yet - see below"))),
    }
    let learned = learn::sources(settings);
    let bytes: u64 = learned.iter().map(|s| s.2).sum();
    card.push(format!("{} {} {}", dot(!learned.is_empty()), key("learned"),
        if learned.is_empty() { style::fg(pal::YELLOW, "no code yet") } else { format!("{} sources · {:.1} MB of code", learned.len(), bytes as f64 / 1e6) }));
    let cwd = ctx.cwd();
    let home = crate::config::home_dir();
    let place = match home.as_ref().and_then(|h| cwd.strip_prefix(h).ok()) {
        Some(rel) => format!("~/{}", rel.display()),
        None => cwd.display().to_string(),
    };
    card.push(format!("{} {} {place}", dot(true), key("folder")));
    lines.extend(widgets::boxed(Some(&style::gradient_styled("✦ aegist", 0.2, true)), &card, w.min(78), pal::BORDER).into_iter().map(|l| format!("  {l}")));
    if checkpoint::load_meta(settings).is_none() {
        lines.push(String::new());
        lines.push(format!("  {}", Style::new().bold().paint("Getting started")));
        for (n, cmd, what) in [("1", "/learn --pack python", "download well-known code to learn from (or /learn <your folder>)"),
                               ("2", "/train 1", "train for an hour - longer is smarter; ctrl+c stops and saves"),
                               ("3", "write a function that ...", "ask for code; everything is checked before you get it")] {
            lines.push(format!("    {}  {}  {}", style::fg(pal::VIOLET, n), Style::new().fg(pal::SKY).paint(&text::pad(cmd, 28)), style::dim(what)));
        }
    }
    lines.push(String::new());
    // what it can do, at a glance
    let chip = |icon: &str, color: style::Rgb, name: &str, ex: &str| {
        format!("{} {} {}", style::fg(color, icon), Style::new().fg(color).bold().paint(name), style::faint(ex))
    };
    let chips = [chip("⌨", pal::CYAN, "code", "write · complete · fix · run"), chip("◉", pal::PINK, "screen", "click · type · keys · apps"),
                 chip("✦", pal::VIOLET, "autopilot", "many steps, planned first")];
    if w >= 100 {
        lines.push(format!("  {}", chips.join(&style::faint("   │   "))));
    } else {
        lines.extend(chips.iter().map(|c| format!("  {c}")));
    }
    lines.push(String::new());
    let tries = ["write a snake game as a web page", "open notepad, type hello and press enter", "take a screenshot", "fix the tests then run main.py"];
    lines.push(format!("  {} {}", style::faint("try"), tries.iter().map(|t| Style::new().fg(pal::SKY).paint(t)).collect::<Vec<_>>()
        .join(&style::faint(" · "))));
    lines.push(format!("  {}", style::faint("/help for everything · /help screen · /safety for what it will and won't do on your computer")));
    lines.push(String::new());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_requests_are_understood_by_rules() {
        match interpret("write a snake game in javascript") {
            Intent::Write { path: None, lang: Some(l), .. } => assert_eq!(l.name, "JavaScript"),
            other => panic!("{other:?}"),
        }
        match interpret("create server.py that serves files") {
            Intent::Write { path: Some(p), .. } => assert_eq!(p, "server.py"),
            other => panic!("{other:?}"),
        }
        match interpret("make a landing page for my bakery") {
            Intent::Write { lang: Some(l), .. } => assert_eq!(l.name, "HTML"),
            other => panic!("{other:?}"),
        }
        assert!(language_in("write a function to go through a list").is_none());
        assert!(language_in("sort less than 3 pm").is_none());
        assert_eq!(language_in("a web server in go").unwrap().name, "Go");
        assert_eq!(language_in("a c program that prints primes").unwrap().name, "C");
        assert_eq!(interpret("complete src/app.py:42"), Intent::Complete("src/app.py:42".into()));
        assert_eq!(interpret("fix `cargo test`"), Intent::Fix(Some("cargo test".into())));
        assert_eq!(interpret("fix the tests"), Intent::Fix(None));
        assert_eq!(interpret("run the tests"), Intent::Test);
        assert_eq!(interpret("run main.py"), Intent::Run("main.py".into()));
        assert_eq!(interpret("where is parse_config defined"), Intent::Grep("parse_config".into()));
        assert_eq!(interpret("show app.rs"), Intent::Open("app.rs".into()));
        assert_eq!(interpret("what is a monad?"), Intent::Question);
        assert_eq!(interpret("explain this code"), Intent::Question);
        assert_eq!(interpret("hello"), Intent::Greeting);
        assert_eq!(interpret("banana"), Intent::Unknown);
    }

    #[test]
    fn completion_points() {
        let text = "def f():\n    # TODO\n\nprint(f())\n";
        assert_eq!(completion_point(text, None), (9, 20)); // the TODO line is replaced
        assert_eq!(completion_point(text, Some(1)), (9, 9)); // after line 1
        assert_eq!(completion_point("a = 1\n", None), (6, 6)); // the end
        assert_eq!(completion_point("a\n\nb\n", Some(2)), (2, 3)); // an empty line is filled
    }

    #[test]
    fn globs() {
        assert!(glob_regex("*.py").is_match("src/app.py") && !glob_regex("*.py").is_match("src/app.pyc"));
        assert!(glob_regex("src/*.rs").is_match("src/main.rs") && !glob_regex("src/*.rs").is_match("lib/src/main.rs"));
        assert!(glob_regex("config").is_match("app/config_loader.py") && !glob_regex("config").is_match("config/app.py"));
    }
}
