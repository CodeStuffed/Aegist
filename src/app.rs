//! The interactive session: `aegist` with no arguments.
//!
//! The screen thread only draws and reads keys; a worker thread does each
//! piece of work and sends back what to show. So typing, the spinner and
//! esc-to-stop stay responsive while the model writes or a program runs.

use crate::actions::{self, Intent, COMMANDS};
use crate::config::Settings;
use crate::session::{self, BrainState, Ctx, Msg, Out, Shared};
use crate::ui::highlight;
use crate::ui::input::{Action, Completion, Editor};
use crate::ui::screen::Screen;
use crate::ui::style::{self, pal, Style};
use crate::ui::text;
use crate::ui::widgets;
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

struct Busy {
    label: String,
    started: Instant,
    handle: JoinHandle<()>,
    stopping: bool,
}

struct StreamView {
    title: String,
    lang: Option<&'static str>,
    text: String,
    spans: Vec<(usize, usize, f32)>,
    started: Instant,
}

struct AskView {
    question: String,
    choices: Vec<(char, String)>,
    sel: usize,
    reply: Sender<char>,
}

/// Puts the terminal back however the session ends (panics included).
struct TermGuard {
    enhanced: bool,
}

impl TermGuard {
    fn enter() -> Result<TermGuard> {
        let mut out = std::io::stdout();
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(out, event::EnableBracketedPaste)?;
        let enhanced = matches!(crossterm::terminal::supports_keyboard_enhancement(), Ok(true))
            && crossterm::execute!(out, event::PushKeyboardEnhancementFlags(event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)).is_ok();
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore(enhanced);
            default_hook(info);
        }));
        Ok(TermGuard { enhanced })
    }
}

fn restore(enhanced: bool) {
    let mut out = std::io::stdout();
    if enhanced {
        let _ = crossterm::execute!(out, event::PopKeyboardEnhancementFlags);
    }
    let _ = crossterm::execute!(out, event::DisableBracketedPaste, crossterm::cursor::Show);
    let _ = crossterm::terminal::disable_raw_mode();
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        restore(self.enhanced);
    }
}

pub struct App {
    ctx: Ctx,
    rx: Receiver<Msg>,
    screen: Screen,
    editor: Editor,
    busy: Option<Busy>,
    stream: Option<StreamView>,
    ask: Option<AskView>,
    queue: VecDeque<String>,
    pending: Vec<String>,
    tick: u64,
    last_interrupt: Option<Instant>,
    quit: bool,
    history_path: PathBuf,
}

fn load_history(path: &PathBuf) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<String>(l).ok())
        .collect()
}

fn save_history(path: &PathBuf, items: &[String]) {
    let keep = &items[items.len().saturating_sub(500)..];
    let text: String = keep.iter().filter_map(|i| serde_json::to_string(i).ok()).map(|l| l + "\n").collect();
    let _ = std::fs::write(path, text);
}

pub fn run(settings: Settings) -> Result<()> {
    let settings = Arc::new(settings);
    let cwd = std::env::current_dir()?;
    let shared = Arc::new(Mutex::new(session::new_shared(&settings, cwd.clone())));
    let (tx, rx) = mpsc::channel();
    let screen = Screen::new();
    let ctx = Ctx {
        settings: settings.clone(),
        shared: shared.clone(),
        out: Out::Screen(tx),
        cancel: Arc::new(AtomicBool::new(false)),
        width: Arc::new(AtomicUsize::new(screen.width)),
    };
    // everything slow starts in the background
    highlight::warm_up();
    session::load_brain(settings.clone(), shared.clone());
    {
        let shared = shared.clone();
        std::thread::spawn(move || {
            let p = crate::project::Project::open(&cwd);
            let mut s = shared.lock().expect("lock");
            if s.project.is_none() {
                s.project = Some(p);
            }
        });
    }
    {
        let (shared, settings) = (shared.clone(), settings.clone());
        std::thread::spawn(move || {
            let names = Arc::new(crate::learn::known_names(&settings));
            shared.lock().expect("lock").learned_names = names;
        });
    }
    let history_path = settings.data_path("").join("history.jsonl");
    let mut app = App {
        editor: Editor::new(load_history(&history_path)),
        ctx,
        rx,
        screen,
        busy: None,
        stream: None,
        ask: None,
        queue: VecDeque::new(),
        pending: Vec::new(),
        tick: 0,
        last_interrupt: None,
        quit: false,
        history_path,
    };
    let guard = TermGuard::enter()?;
    crossterm::execute!(std::io::stdout(), crossterm::style::ResetColor)?; // (turns on escape codes on Windows)
    let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::SetTitle("✦ Aegist"));
    app.pending.extend(actions::welcome(&app.ctx));
    let result = app.run_loop();
    let mut out = std::io::stdout();
    app.screen.close(&mut out);
    drop(guard);
    save_history(&app.history_path, app.editor.history());
    {
        let s = shared.lock().expect("lock");
        s.jobs.stop_all();
    }
    println!("  {}", style::gradient_styled("✦ until next time", 0.2, true));
    result
}

impl App {
    fn run_loop(&mut self) -> Result<()> {
        let mut out = std::io::stdout();
        let mut dirty = true;
        let mut last_draw = Instant::now();
        while !self.quit {
            while let Ok(m) = self.rx.try_recv() {
                self.on_msg(m);
                dirty = true;
            }
            if self.busy.as_ref().is_some_and(|b| b.handle.is_finished()) {
                self.busy = None;
                self.stream = None;
                self.ask = None;
                dirty = true;
                if let Some(next) = self.queue.pop_front() {
                    self.submit(next);
                }
            }
            if event::poll(Duration::from_millis(33))? {
                match event::read()? {
                    Event::Key(k) if k.kind != KeyEventKind::Release => self.on_key(k),
                    Event::Paste(s) if self.ask.is_none() => {
                        self.editor.paste(&s);
                        self.refresh_menu();
                    }
                    Event::Resize(w, h) => {
                        self.screen.resize(w, h);
                        self.ctx.width.store(self.screen.width, Ordering::Relaxed);
                    }
                    _ => {}
                }
                dirty = true;
            }
            let animating = self.busy.is_some();
            if dirty || (animating && last_draw.elapsed() >= Duration::from_millis(60)) || last_draw.elapsed() >= Duration::from_millis(500) {
                self.tick += 1;
                self.draw(&mut out);
                last_draw = Instant::now();
                dirty = false;
            }
        }
        Ok(())
    }

    fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Print(lines) => {
                if let Ok(mut sh) = self.ctx.shared.lock() {
                    sh.transcript.extend(lines.iter().flat_map(|l| l.split('\n').map(|x| text::strip_ansi(x).trim_end().to_string())));
                    let over = sh.transcript.len().saturating_sub(50_000);
                    sh.transcript.drain(..over);
                }
                for l in lines {
                    self.pending.extend(l.split('\n').map(str::to_string));
                }
            }
            Msg::Status(s) => {
                if let Some(b) = self.busy.as_mut() {
                    b.label = s;
                }
            }
            Msg::StreamStart { title, lang } => {
                self.stream = Some(StreamView { title, lang, text: String::new(), spans: Vec::new(), started: Instant::now() });
            }
            Msg::Stream(t, p) => {
                if let Some(s) = self.stream.as_mut() {
                    let a = s.text.len();
                    s.text.push_str(&t);
                    s.spans.push((a, s.text.len(), p));
                }
            }
            Msg::StreamEnd => self.stream = None,
            Msg::Ask { question, choices, reply } => self.ask = Some(AskView { question, choices, sel: 0, reply }),
            Msg::Done => {}
        }
    }

    fn interrupt_work(&mut self) {
        if let Some(b) = self.busy.as_mut() {
            self.ctx.cancel.store(true, Ordering::Relaxed);
            b.stopping = true;
            b.label = "Stopping…".into();
        }
        if let Some(a) = self.ask.take() {
            let _ = a.reply.send('n');
        }
    }

    fn on_key(&mut self, k: event::KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        if let Some(a) = self.ask.as_mut() {
            let n = a.choices.len();
            let choice = match k.code {
                KeyCode::Left | KeyCode::BackTab | KeyCode::Up => {
                    a.sel = (a.sel + n - 1) % n;
                    None
                }
                KeyCode::Right | KeyCode::Tab | KeyCode::Down => {
                    a.sel = (a.sel + 1) % n;
                    None
                }
                KeyCode::Enter => Some(a.choices[a.sel].0),
                KeyCode::Esc => Some('n'),
                KeyCode::Char('c') if ctrl => {
                    self.interrupt_work();
                    return;
                }
                KeyCode::Char(c) => a.choices.iter().find(|x| x.0 == c.to_ascii_lowercase()).map(|x| x.0),
                _ => None,
            };
            if let Some(c) = choice {
                let a = self.ask.take().expect("asking");
                let label = a.choices.iter().find(|x| x.0 == c).map_or(String::new(), |x| x.1.clone());
                self.pending.push(format!("    {} {} {}", style::faint(&a.question), style::fg(pal::VIOLET, "›"), Style::new().bold().paint(&label)));
                let _ = a.reply.send(c);
            }
            return;
        }
        if self.busy.is_some() && (k.code == KeyCode::Esc || (ctrl && k.code == KeyCode::Char('c') && self.editor.is_empty())) {
            self.interrupt_work();
            return;
        }
        match self.editor.key(k) {
            Action::Submit(text) => {
                if self.busy.is_some() {
                    if !text.trim().is_empty() {
                        self.pending.push(format!("  {} {}", style::faint("queued:"), style::dim(text.trim())));
                        self.queue.push_back(text);
                    }
                } else {
                    self.submit(text);
                }
            }
            Action::Interrupt => {
                if self.last_interrupt.is_some_and(|t| t.elapsed() < Duration::from_millis(1500)) {
                    self.quit = true;
                } else {
                    self.last_interrupt = Some(Instant::now());
                }
            }
            Action::Eof => self.quit = true,
            Action::ClearScreen => {
                let mut out = std::io::stdout();
                self.screen.clear_all(&mut out);
            }
            Action::Escape | Action::None => {}
        }
        self.refresh_menu();
    }

    /// Completions for the word being typed: commands after '/', else paths.
    fn refresh_menu(&mut self) {
        let buf = self.editor.buf.clone();
        let word = self.editor.current_word().to_string();
        let items: Vec<Completion> = if buf.starts_with('/') && !buf.contains(' ') {
            COMMANDS.iter().filter(|c| format!("/{}", c.name).starts_with(&buf))
                .map(|c| Completion { insert: format!("/{}", c.name), label: format!("/{} {}", c.name, c.args), detail: c.about.to_string() })
                .collect()
        } else if !word.is_empty() && (word.contains('/') || word.starts_with('.') || word.starts_with('~')) && !word.contains("://") {
            path_completions(&self.ctx.cwd(), &word)
        } else {
            Vec::new()
        };
        self.editor.set_menu(items);
    }

    fn submit(&mut self, text: String) {
        let t = text.trim().to_string();
        if t.is_empty() {
            return;
        }
        {
            let mut sh = self.ctx.shared.lock().expect("lock");
            sh.history.push(t.clone());
            sh.transcript.push(String::new());
            sh.transcript.extend(t.lines().map(|l| format!("> {l}")));
        }
        self.pending.push(String::new());
        for (i, l) in t.lines().enumerate() {
            let lead = if i == 0 { Style::new().fg(pal::VIOLET).bold().paint("❯") } else { " ".into() };
            self.pending.push(format!("{lead} {}", Style::new().bold().paint(l)));
        }
        let (name, args) = match t.strip_prefix('/') {
            Some(rest) => {
                let (n, a) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                (n.to_lowercase(), a.trim().to_string())
            }
            None if t.starts_with('!') => ("run".to_string(), t[1..].trim().to_string()),
            None => (String::new(), t.clone()),
        };
        match name.as_str() {
            "exit" | "quit" | "q" => {
                self.quit = true;
                return;
            }
            "clear" | "cls" => {
                let mut out = std::io::stdout();
                self.screen.clear_all(&mut out);
                return;
            }
            _ => {}
        }
        let label = match name.as_str() {
            "" => "Thinking it over…".to_string(),
            "write" | "complete" | "fix" => "Reading…".to_string(),
            "train" => "Starting to train…".to_string(),
            "learn" => "Reading code…".to_string(),
            other => format!("{other}…"),
        };
        let ctx = self.ctx.clone();
        ctx.cancel.store(false, Ordering::Relaxed);
        let handle = std::thread::spawn(move || {
            let result = dispatch(&ctx, &name, &args);
            if let Err(e) = result {
                let w = ctx.width().saturating_sub(6);
                let msg = e.to_string();
                let mut lines = Vec::new();
                for (i, l) in text::wrap(&msg, w).into_iter().enumerate() {
                    lines.push(if i == 0 { format!("  {} {l}", style::fg(pal::RED, "✗")) } else { format!("    {}", style::dim(&l)) });
                }
                ctx.print(lines);
            }
            ctx.send(Msg::Done);
        });
        self.busy = Some(Busy { label, started: Instant::now(), handle, stopping: false });
    }

    fn status_bar(&self) -> String {
        let w = self.screen.width.saturating_sub(1);
        let hint = if self.busy.is_some() {
            format!("{} {}", style::fg(pal::VIOLET, "esc"), style::faint("to stop · enter queues a message"))
        } else {
            format!("{} {}", style::fg(pal::VIOLET, "/"), style::faint("for commands · ! runs a shell command · ctrl+c twice to leave"))
        };
        let hint = if self.last_interrupt.is_some_and(|t| t.elapsed() < Duration::from_millis(1500)) {
            style::fg(pal::YELLOW, "press ctrl+c again to leave")
        } else {
            hint
        };
        let right = {
            let s = self.ctx.shared.try_lock();
            match s.as_deref() {
                Ok(Shared { brain, project, jobs, .. }) => {
                    let model = match brain {
                        BrainState::Loading => style::fg(pal::FAINT, "◌ loading model"),
                        BrainState::Ready(b) => format!("{} {}", style::fg(pal::CYAN, "◆"),
                            style::dim(&format!("{:.1}M · {}", b.num_params as f64 / 1e6, b.model.precision))),
                        BrainState::Missing(_) => style::fg(pal::YELLOW, "◇ no model yet"),
                        BrainState::Failed(_) => style::fg(pal::RED, "✗ model failed to load"),
                    };
                    let proj = match project {
                        Some(p) => style::faint(&format!("{} · {} files", p.root.file_name().map_or(String::new(), |n| n.to_string_lossy().to_string()),
                                                          p.file_count())),
                        None => style::faint("reading project…"),
                    };
                    let running = jobs.list.iter().filter(|j| j.state() == crate::ide::JobState::Running).count();
                    let jobs = if running > 0 { format!(" {}", style::fg(pal::GREEN, &format!("● {running} running"))) } else { String::new() };
                    format!("{model} {} {proj}{jobs}", style::faint("·"))
                }
                Err(_) => String::new(),
            }
        };
        let gap = w.saturating_sub(text::width(&hint) + text::width(&right) + 4);
        text::truncate(&format!("  {hint}{}{right}", " ".repeat(gap.max(1))), w)
    }

    fn live(&self) -> (Vec<String>, Option<(usize, usize)>) {
        let w = self.screen.width;
        let mut lines: Vec<String> = Vec::new();
        if let Some(s) = &self.stream {
            let tokens = s.spans.len();
            let rate = tokens as f64 / s.started.elapsed().as_secs_f64().max(0.05);
            lines.push(String::new());
            lines.push(format!("  {} {} {}", style::fg(pal::VIOLET, "✦"), Style::new().bold().paint(&s.title),
                               style::faint(&format!("· {tokens} tokens · {rate:.0}/s"))));
            let unsure: Vec<(usize, usize)> = s.spans.iter().filter(|x| x.2 < self.ctx.settings.honesty.uncertain_below).map(|x| (x.0, x.1)).collect();
            let panel = widgets::code_panel(&s.text, s.lang, 1, w, &unsure);
            let room = self.screen.height.saturating_sub(12).clamp(3, 16);
            lines.extend(panel.iter().skip(panel.len().saturating_sub(room)).cloned());
        }
        if let Some(b) = &self.busy {
            if self.ask.is_none() {
                lines.push(String::new());
                let secs = b.started.elapsed().as_secs_f64();
                let spin = if b.stopping { style::fg(pal::YELLOW, &format!("■ {}", b.label)) } else { widgets::spinner(self.tick, &b.label) };
                lines.push(format!("  {spin} {}", style::faint(&format!("{secs:.1}s"))));
            }
        }
        if let Some(a) = &self.ask {
            lines.push(String::new());
            lines.push(format!("  {} {}", style::fg(pal::VIOLET, "?"), Style::new().bold().paint(&a.question)));
            let mut row = String::from("    ");
            for (i, (k, label)) in a.choices.iter().enumerate() {
                let body = format!("{label} ({k})");
                if i == a.sel {
                    row.push_str(&format!("{} {}   ", Style::new().fg(pal::VIOLET).bold().paint("❯"), Style::new().fg(pal::VIOLET).bold().underline().paint(&body)));
                } else {
                    row.push_str(&format!("  {}   ", style::dim(&body)));
                }
            }
            lines.push(row);
            lines.push(format!("    {}", style::faint("← → to choose · enter to confirm · esc for no")));
            return (lines, None);
        }
        lines.push(String::new());
        let placeholder = if self.busy.is_some() { "working… (type ahead: enter queues it)" } else { "write, complete or fix code - or /help" };
        let (boxed, cur) = self.editor.render(w, placeholder, self.busy.is_none());
        let top = lines.len();
        lines.extend(boxed);
        lines.extend(self.editor.render_menu(w));
        lines.push(self.status_bar());
        (lines, Some((top + cur.0, cur.1)))
    }

    fn draw(&mut self, out: &mut impl Write) {
        let (live, cursor) = self.live();
        let pending = std::mem::take(&mut self.pending);
        self.screen.frame(out, &pending, &live, cursor);
    }
}

fn path_completions(cwd: &std::path::Path, word: &str) -> Vec<Completion> {
    let expanded = match word.strip_prefix('~') {
        Some(rest) => crate::config::home_dir().map(|h| format!("{}{rest}", h.display())).unwrap_or_else(|| word.to_string()),
        None => word.to_string(),
    };
    let (dir_part, stem) = match expanded.rfind('/') {
        Some(i) => (&expanded[..=i], &expanded[i + 1..]),
        None => ("", expanded.as_str()),
    };
    let dir = if dir_part.starts_with('/') { PathBuf::from(dir_part) } else { cwd.join(dir_part) };
    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
    let shown_dir = &word[..word.len() - stem.len()];
    let mut items: Vec<Completion> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with(stem) || (name.starts_with('.') && !stem.starts_with('.')) {
                return None;
            }
            let is_dir = e.path().is_dir();
            let insert = format!("{shown_dir}{name}{}", if is_dir { "/" } else { "" });
            Some(Completion { label: insert.clone(), insert, detail: if is_dir { "folder".into() } else { String::new() } })
        })
        .collect();
    items.sort_by(|a, b| a.label.cmp(&b.label));
    items.truncate(12);
    if items.len() == 1 && items[0].insert == word {
        items.clear();
    }
    items
}

/// Do what a submitted line asks.
pub fn dispatch(ctx: &Ctx, name: &str, args: &str) -> Result<()> {
    let need = |what: &str| -> Result<()> {
        if args.trim().is_empty() {
            anyhow::bail!("{what}");
        }
        Ok(())
    };
    match name {
        "" => match crate::autopilot::route(args) {
            crate::autopilot::Route::Plan => crate::autopilot::run(ctx, args),
            crate::autopilot::Route::One(act) => crate::computer::run_one(ctx, act),
            crate::autopilot::Route::CantDo(why) => {
                crate::autopilot::dont_know(ctx, &why);
                Ok(())
            }
            crate::autopilot::Route::NotMine => plain(ctx, args),
        },
        "help" | "h" | "?" => {
            actions::help(ctx, args);
            Ok(())
        }
        "screen" | "screenshot" | "shot" | "windows" | "wins" | "where" | "cursor" | "pointer" | "focus" | "switch" | "launch" | "app-open"
        | "browse" | "type" | "key" | "press" | "keys" | "mouse" | "move" | "click" | "rclick" | "right-click" | "rightclick" | "dclick"
        | "double-click" | "doubleclick" | "drag" | "scroll" | "wait" => crate::computer::command(ctx, name, args),
        "auto" | "do" | "autopilot" => {
            need("/auto <steps in plain words>, e.g. /auto open notepad, type hello and press enter")?;
            crate::autopilot::run(ctx, args)
        }
        "safety" | "permissions" | "rules" => {
            crate::extras::safety(ctx);
            Ok(())
        }
        "status" | "st" => {
            crate::extras::status(ctx);
            Ok(())
        }
        "config" | "settings" | "set" => crate::extras::config(ctx, args),
        "history" => {
            crate::extras::history(ctx, args);
            Ok(())
        }
        "export" | "save-session" => crate::extras::export(ctx, args),
        "cd" | "chdir" => crate::extras::cd(ctx, args),
        "copy" | "yank" => crate::extras::copy(ctx, args),
        "git" => crate::extras::git(ctx, args),
        "install" => {
            let made = crate::extras::install()?;
            let mut lines = vec![String::new(), format!("  {} {}", style::fg(pal::GREEN, "✓"), Style::new().bold().paint("Aegist is in your apps now"))];
            for p in made {
                lines.push(format!("    {} {}", style::faint("·"), style::dim(&p.display().to_string())));
            }
            lines.push(format!("    {}", style::faint("open it from the apps menu (or the desktop) and it starts in its own window")));
            ctx.print(lines);
            Ok(())
        }
        _ => older(ctx, name, args),
    }
}

/// A plain-words request that isn't about the screen.
fn plain(ctx: &Ctx, args: &str) -> Result<()> {
    if matches!(args.trim().to_lowercase().as_str(), "copy" | "copy it" | "copy that" | "copy the code") {
        return crate::extras::copy(ctx, "");
    }
    if crate::abilities::converse(ctx, args)? {
        return Ok(());
    }
    match actions::interpret(args) {
        Intent::Write { path, lang, description } => actions::write(ctx, path.as_deref(), lang, &description),
        Intent::Complete(t) => actions::complete(ctx, &t),
        Intent::Fix(cmd) => actions::fix(ctx, cmd.as_deref()),
        Intent::Run(c) => actions::run(ctx, &c),
        Intent::Test => actions::test(ctx),
        Intent::Grep(t) => actions::grep(ctx, &t),
        Intent::Open(p) => actions::open(ctx, &p),
        Intent::Undo => actions::undo(ctx),
        other => {
            actions::decline(ctx, &other);
            Ok(())
        }
    }
}

/// The commands that were here first.
fn older(ctx: &Ctx, name: &str, args: &str) -> Result<()> {
    let need = |what: &str| -> Result<()> {
        if args.trim().is_empty() {
            anyhow::bail!("{what}");
        }
        Ok(())
    };
    match name {
        "write" | "new" => {
            need("/write <file> <what it should do>")?;
            let (first, rest) = args.split_once(char::is_whitespace).unwrap_or((args, ""));
            if crate::lang::is_code(std::path::Path::new(first)) && first.contains('.') {
                let description = if rest.trim().is_empty() { first.to_string() } else { rest.trim().to_string() };
                actions::write(ctx, Some(first), None, &description)
            } else {
                actions::write(ctx, None, actions::language_in(&args.to_lowercase()), args)
            }
        }
        "complete" | "c" => {
            need("/complete <file>[:line]")?;
            actions::complete(ctx, args.trim())
        }
        "fix" => actions::fix(ctx, (!args.trim().is_empty()).then(|| args.trim().trim_matches('`'))),
        "run" | "r" => {
            need("/run <command or file>")?;
            actions::run(ctx, args)
        }
        "test" | "tests" => actions::test(ctx),
        "start" | "bg" => {
            need("/start <command>")?;
            actions::start(ctx, args)
        }
        "jobs" | "ps" => {
            actions::jobs(ctx);
            Ok(())
        }
        "logs" => actions::logs(ctx, args),
        "stop" | "kill" => actions::stop(ctx, if args.trim().is_empty() { "all" } else { args }),
        "serve" => actions::serve(ctx, args),
        "preview" | "look" => actions::preview(ctx, args),
        "open" | "show-file" | "cat" | "view" => {
            need("/open <file>[:from-to]")?;
            actions::open(ctx, args)
        }
        "find" | "files" => {
            need("/find <name pattern>")?;
            actions::find(ctx, args)
        }
        "grep" | "search" => {
            need("/grep <text or regex>")?;
            actions::grep(ctx, args)
        }
        "tree" | "ls" => actions::tree(ctx, args),
        "diff" | "changes" => {
            actions::diff(ctx);
            Ok(())
        }
        "undo" => actions::undo(ctx),
        "show" => {
            actions::show_refused(ctx);
            Ok(())
        }
        "learn" => actions::learn(ctx, args),
        "agent" | "app" | "use" => crate::abilities::agent_command(ctx, args),
        "decide" | "choose" => crate::abilities::decide(ctx, args),
        "voice" | "speak" => crate::abilities::voice(ctx, args),
        "predict" | "forecast" => crate::abilities::predict_series(ctx, args),
        "estimate" => crate::abilities::estimate(ctx, args),
        "answer" => crate::abilities::answer(ctx, args),
        "good" | "right" => crate::abilities::feedback(ctx, true),
        "bad" | "wrong" => crate::abilities::feedback(ctx, false),
        "train" => actions::train(ctx, args),
        "model" | "doctor" | "about" => {
            actions::model(ctx);
            Ok(())
        }
        other => {
            let close: Vec<String> = COMMANDS.iter().filter(|c| c.name.starts_with(&other[..other.len().min(2)])).map(|c| format!("/{}", c.name)).collect();
            anyhow::bail!("There's no /{other} command.{}", if close.is_empty() { " /help lists them all.".to_string() } else { format!(" Did you mean {}?", close.join(" or ")) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_complete_from_the_folder() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src/nested")).unwrap();
        std::fs::write(tmp.path().join("src/main.rs"), "").unwrap();
        std::fs::write(tmp.path().join("src/.hidden"), "").unwrap();
        let got: Vec<String> = path_completions(tmp.path(), "src/").into_iter().map(|c| c.insert).collect();
        assert_eq!(got, vec!["src/main.rs", "src/nested/"]);
        let got: Vec<String> = path_completions(tmp.path(), "src/ma").into_iter().map(|c| c.insert).collect();
        assert_eq!(got, vec!["src/main.rs"]);
        assert!(path_completions(tmp.path(), "src/main.rs").is_empty());
    }

    #[test]
    fn plain_output_commands_work_without_a_screen() {
        let tmp = tempfile::tempdir().unwrap();
        let settings = Arc::new(crate::config::testing::settings(&tmp.path().join("data")));
        let proj = tmp.path().join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(proj.join("app.py"), "def hello():\n    return 'hi'\n").unwrap();
        let mut shared = session::new_shared(&settings, proj.clone());
        shared.project = Some(crate::project::Project::open(&proj));
        shared.brain = BrainState::Missing("no model".into());
        let ctx = Ctx { settings, shared: Arc::new(Mutex::new(shared)), out: Out::Plain { yes: false }, cancel: Arc::new(AtomicBool::new(false)),
                        width: Arc::new(AtomicUsize::new(80)) };
        dispatch(&ctx, "grep", "hello").unwrap();
        dispatch(&ctx, "open", "app.py:1-2").unwrap();
        dispatch(&ctx, "", "what is the meaning of life?").unwrap();
        assert!(dispatch(&ctx, "write", "x.py a thing").unwrap_err().to_string().contains("no model"));
        assert!(dispatch(&ctx, "frobnicate", "").unwrap_err().to_string().contains("no /frobnicate"));
        #[cfg(unix)]
        {
            dispatch(&ctx, "run", "echo from-run").unwrap();
            assert_eq!(ctx.shared.lock().unwrap().last_output, vec!["from-run"]);
        }
    }
}
