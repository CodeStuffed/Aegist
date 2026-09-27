//! Session housekeeping: /status, /config, /history, /export, /cd, /copy,
//! /git, /safety - and `aegist install`, which puts Aegist in your apps
//! menu so it opens in its own window like any other app.

use crate::config::Settings;
use crate::session::{BrainState, Ctx};
use crate::ui::style::{self, pal, Style};
use crate::ui::{text, widgets};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

fn key(k: &str) -> String {
    Style::new().fg(pal::VIOLET).bold().paint(&text::pad(k, 10))
}

fn ago(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {}s", secs / 60, secs % 60),
        _ => format!("{}h {}m", secs / 3600, secs % 3600 / 60),
    }
}

/// `/status`: everything about this session at a glance.
pub fn status(ctx: &Ctx) {
    let w = ctx.width();
    let mut card = Vec::new();
    {
        let s = ctx.shared.lock().expect("lock");
        let model = match &s.brain {
            BrainState::Ready(b) => format!("{} {}", style::fg(pal::CYAN, "◆"), b.describe()),
            BrainState::Loading => style::faint("loading…"),
            BrainState::Missing(_) => style::fg(pal::YELLOW, "◇ not trained yet (/learn, then /train)"),
            BrainState::Failed(e) => style::fg(pal::RED, &format!("✗ {e}")),
        };
        card.push(format!("{} {model}", key("model")));
        let project = match &s.project {
            Some(p) => format!("{} · {} files", p.root.display(), p.file_count()),
            None => "reading…".into(),
        };
        card.push(format!("{} {project}", key("project")));
        let running = s.jobs.list.iter().filter(|j| j.state() == crate::ide::JobState::Running).count();
        card.push(format!("{} {} running · {} total{}", key("jobs"), running, s.jobs.list.len(),
                          if s.servers.is_empty() { String::new() } else { format!(" · {} site{} served", s.servers.len(), if s.servers.len() == 1 { "" } else { "s" }) }));
        let desk = match crate::agent::desktop::open() {
            Ok(mut d) => {
                let size = d.screen_size().map(|(w, h)| format!(" · {w}×{h}")).unwrap_or_default();
                format!("{} {}{size} · {}", style::fg(pal::GREEN, "●"), d.name(),
                        if s.desktop_ok { "you've allowed mouse and keyboard this session" } else { "asks before using it" })
            }
            Err(e) => format!("{} {}", style::fg(pal::YELLOW, "◇"), text::truncate(&e.to_string(), w.saturating_sub(30))),
        };
        card.push(format!("{} {desk}", key("screen")));
        card.push(format!("{} {}", key("voice"), if s.voice { "on" } else { "off (/voice on)" }));
        card.push(format!("{} {} · {} request{} · {} line{} shown", key("session"), ago(s.started.elapsed().as_secs()), s.history.len(),
                          if s.history.len() == 1 { "" } else { "s" }, s.transcript.len(), if s.transcript.len() == 1 { "" } else { "s" }));
        let h = &ctx.settings.honesty;
        card.push(format!("{} {} candidates · refuses under {:.0}% certainty, or {:.0}% on any stretch", key("honesty"), h.samples,
                          h.min_confidence * 100.0, h.min_stretch * 100.0));
    }
    let mut lines = vec![String::new()];
    lines.extend(widgets::boxed(Some(&style::gradient_styled(&style::spaced("status"), 0.2, true)), &card, w.saturating_sub(4).min(100), pal::BORDER)
        .into_iter().map(|l| format!("  {l}")));
    ctx.print(lines);
}

/// Settings `/config` shows and can change: (key, what it does).
const TUNABLE: &[(&str, &str)] = &[
    ("inference.temperature", "0 = always the most likely token"),
    ("inference.max_new_tokens", "longest answer, in tokens"),
    ("inference.precision", "f32, int8 or int4 weights"),
    ("inference.speculative", "guess ahead by copying (faster)"),
    ("honesty.samples", "candidates written and checked"),
    ("honesty.min_confidence", "average certainty below this => I don't know"),
    ("honesty.min_stretch", "any 8 tokens below this => I don't know"),
    ("honesty.give_up_below", "stop writing when a stretch falls below this"),
    ("honesty.min_agreement", "candidates agreeing less => unverified"),
    ("honesty.max_invented_names", "made-up names allowed before refusing"),
    ("decide.abstain_below", "decisions less likely than this => I don't know"),
    ("agent.confirm", "ask before using the mouse and keyboard"),
    ("agent.max_actions", "most mouse/keyboard actions in one run"),
    ("agent.action_delay_ms", "pause after each action"),
    ("agent.failsafe_px", "you move the mouse this far => it stops"),
    ("voice.enabled", "speak replies aloud"),
];

fn current(settings: &Settings, key: &str) -> String {
    let v = |x: &dyn std::fmt::Display| x.to_string();
    match key {
        "inference.temperature" => v(&settings.inference.temperature),
        "inference.max_new_tokens" => v(&settings.inference.max_new_tokens),
        "inference.precision" => v(&settings.inference.precision),
        "inference.speculative" => v(&settings.inference.speculative),
        "honesty.samples" => v(&settings.honesty.samples),
        "honesty.min_confidence" => v(&settings.honesty.min_confidence),
        "honesty.min_stretch" => v(&settings.honesty.min_stretch),
        "honesty.give_up_below" => v(&settings.honesty.give_up_below),
        "honesty.min_agreement" => v(&settings.honesty.min_agreement),
        "honesty.max_invented_names" => v(&settings.honesty.max_invented_names),
        "decide.abstain_below" => v(&settings.decide.abstain_below),
        "agent.confirm" => v(&settings.agent.confirm),
        "agent.max_actions" => v(&settings.agent.max_actions),
        "agent.action_delay_ms" => v(&settings.agent.action_delay_ms),
        "agent.failsafe_px" => v(&settings.agent.failsafe_px),
        "voice.enabled" => v(&settings.voice.enabled),
        _ => "?".into(),
    }
}

/// `text` (a settings file) with `section.key` set to `value`, keeping
/// every comment and the layout. The key must already be in the file.
pub fn set_yaml_value(text: &str, dotted: &str, value: &str) -> Result<String> {
    let (section, key) = dotted.split_once('.').ok_or_else(|| anyhow::anyhow!("settings are named like honesty.min_confidence"))?;
    let mut out = Vec::new();
    let (mut in_section, mut done) = (false, false);
    for line in text.lines() {
        let top_level = !line.starts_with(' ') && !line.starts_with('\t') && !line.trim().is_empty() && !line.trim_start().starts_with('#');
        if top_level {
            in_section = line.split('#').next().unwrap_or("").trim().strip_suffix(':') == Some(section);
        }
        let body = line.trim_start();
        let indent = &line[..line.len() - body.len()];
        if in_section && !done && !top_level && body.starts_with(&format!("{key}:")) {
            let after = &body[key.len() + 1..];
            let comment = after.find(" #").map(|i| &after[i..]).unwrap_or("");
            let old_len = after.len() - comment.len();
            let new_val = format!(" {value}");
            let pad = if comment.is_empty() { String::new() } else { " ".repeat(old_len.saturating_sub(new_val.len())) };
            out.push(format!("{indent}{key}:{new_val}{pad}{comment}"));
            done = true;
            continue;
        }
        out.push(line.to_string());
    }
    if !done {
        bail!("{dotted} isn't in the settings file");
    }
    let mut s = out.join("\n");
    if text.ends_with('\n') {
        s.push('\n');
    }
    Ok(s)
}

/// `/config [key [value]]`
pub fn config(ctx: &Ctx, args: &str) -> Result<()> {
    let path = ctx.settings.settings_path();
    let words: Vec<&str> = args.split_whitespace().collect();
    match words.as_slice() {
        [] => {
            let mut lines = vec![String::new(), format!("  {} {}", widgets::heading("Settings", 0.2, 24), style::faint(&path.display().to_string()))];
            let kw = TUNABLE.iter().map(|(k, _)| k.len()).max().unwrap_or(20);
            for (k, what) in TUNABLE {
                lines.push(format!("    {} {} {}", Style::new().fg(pal::SKY).paint(&text::pad(k, kw)),
                                   Style::new().bold().paint(&text::pad(&current(&ctx.settings, k), 8)), style::faint(what)));
            }
            lines.push(format!("    {}", style::faint("change one: /config honesty.min_confidence 0.4 · takes effect the next time Aegist starts")));
            ctx.print(lines);
            Ok(())
        }
        [k] => {
            let Some((_, what)) = TUNABLE.iter().find(|(t, _)| t == k) else { bail!("/config doesn't know {k}; /config lists what it can change") };
            ctx.line(format!("  {} = {}  {}", Style::new().fg(pal::SKY).paint(k), Style::new().bold().paint(&current(&ctx.settings, k)), style::faint(what)));
            Ok(())
        }
        [k, v] => {
            if !TUNABLE.iter().any(|(t, _)| t == k) {
                bail!("/config can't change {k}; /config lists what it can (the rest is in {})", path.display());
            }
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            let changed = set_yaml_value(&text, k, v)?;
            // only save settings that still load
            Settings::from_str(&changed, &ctx.settings.root).map_err(|e| anyhow::anyhow!("{k} can't be {v}: {e}"))?;
            std::fs::write(&path, changed)?;
            ctx.line(format!("  {} {} {} {}", style::fg(pal::GREEN, "✓"), Style::new().fg(pal::SKY).paint(k), style::dim("is now"), Style::new().bold().paint(v)));
            ctx.line(format!("    {}", style::faint("saved; it takes effect the next time Aegist starts")));
            Ok(())
        }
        _ => bail!("/config, /config <setting>, or /config <setting> <value>"),
    }
}

/// `/history [n]`
pub fn history(ctx: &Ctx, args: &str) {
    let n: usize = args.trim().parse().unwrap_or(20);
    let items = ctx.shared.lock().expect("lock").history.clone();
    let mut lines = vec![String::new()];
    if items.is_empty() {
        lines.push(format!("  {}", style::dim("nothing yet this session (↑ recalls earlier sessions' requests)")));
    }
    let start = items.len().saturating_sub(n);
    for (i, h) in items.iter().enumerate().skip(start) {
        lines.push(text::truncate(&format!("  {} {}", style::faint(&format!("{:>3}", i + 1)), h.lines().next().unwrap_or("")), ctx.width()));
    }
    ctx.print(lines);
}

/// Days since 1970 to (year, month, day), for file names.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

pub fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (y, m, d) = civil(secs.div_euclid(86_400));
    let t = secs.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}-{:02}{:02}", t / 3600, t % 3600 / 60)
}

/// `/export [file]`: the session so far, as text.
pub fn export(ctx: &Ctx, args: &str) -> Result<()> {
    let name = if args.trim().is_empty() { format!("aegist-session-{}.txt", stamp()) } else { args.trim().to_string() };
    let path = ctx.cwd().join(&name);
    let lines = ctx.shared.lock().expect("lock").transcript.clone();
    if lines.is_empty() {
        bail!("nothing to export yet");
    }
    let mut body = lines.join("\n");
    body.push('\n');
    std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
    ctx.line(format!("  {} {} {}", style::fg(pal::GREEN, "✓"), style::dim(&format!("saved {} lines to", lines.len())), Style::new().fg(pal::SKY).paint(&name)));
    Ok(())
}

fn expand(ctx: &Ctx, p: &str) -> PathBuf {
    let p = p.trim().trim_matches(['"', '\'']);
    let expanded = match p.strip_prefix('~') {
        Some(rest) => crate::config::home_dir().map(|h| h.join(rest.trim_start_matches(['/', '\\']))).unwrap_or_else(|| PathBuf::from(p)),
        None => PathBuf::from(p),
    };
    if expanded.is_absolute() { expanded } else { ctx.cwd().join(expanded) }
}

/// `/cd <folder>`: work in another folder (and its project).
pub fn cd(ctx: &Ctx, args: &str) -> Result<()> {
    let target = if args.trim().is_empty() { crate::config::home_dir().ok_or_else(|| anyhow::anyhow!("where to?"))? } else { expand(ctx, args) };
    let dir = target.canonicalize().map_err(|_| anyhow::anyhow!("there's no folder {}", target.display()))?;
    if !dir.is_dir() {
        bail!("{} isn't a folder", dir.display());
    }
    std::env::set_current_dir(&dir)?;
    ctx.status("Reading the project…");
    let project = crate::project::Project::open(&dir);
    let files = project.file_count();
    let root = project.root.clone();
    {
        let mut s = ctx.shared.lock().expect("lock");
        s.cwd = dir.clone();
        s.project = Some(project);
    }
    ctx.line(format!("  {} {} {}", style::fg(pal::GREEN, "✓"), Style::new().fg(pal::SKY).bold().paint(&dir.display().to_string()),
                     style::faint(&format!("· project {} · {files} files", root.display()))));
    Ok(())
}

/// Put `text` on the system clipboard.
pub fn to_clipboard(text: &str) -> Result<&'static str> {
    use std::io::Write;
    let tools: &[(&'static str, &[&str])] = if cfg!(windows) {
        &[("powershell", &["-NoProfile", "-Command", "$input | Set-Clipboard"]), ("clip", &[])]
    } else if cfg!(target_os = "macos") {
        &[("pbcopy", &[])]
    } else if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        &[("wl-copy", &[]), ("xclip", &["-selection", "clipboard"]), ("xsel", &["--clipboard", "--input"])]
    } else {
        &[("xclip", &["-selection", "clipboard"]), ("xsel", &["--clipboard", "--input"]), ("wl-copy", &[])]
    };
    for (tool, args) in tools {
        if crate::proc::which(tool).is_none() {
            continue;
        }
        let mut child = std::process::Command::new(tool)
            .args(*args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        child.stdin.take().expect("piped").write_all(text.as_bytes())?;
        if child.wait()?.success() {
            return Ok(tool);
        }
    }
    bail!("there's no clipboard tool here ({})", tools.iter().map(|t| t.0).collect::<Vec<_>>().join(", "))
}

/// `/copy [text]`: the last code Aegist wrote (or the last output, or your text).
pub fn copy(ctx: &Ctx, args: &str) -> Result<()> {
    let (text, what) = if !args.trim().is_empty() {
        (args.trim().to_string(), "your text".to_string())
    } else {
        let s = ctx.shared.lock().expect("lock");
        match (&s.last_code, s.last_output.is_empty()) {
            (Some(c), _) => (c.clone(), "the last code I wrote".to_string()),
            (None, false) => (s.last_output.join("\n"), "the last program's output".to_string()),
            (None, true) => bail!("nothing to copy yet: no code written and nothing run this session"),
        }
    };
    let tool = to_clipboard(&text)?;
    ctx.line(format!("  {} {} {}", style::fg(pal::GREEN, "✓"), style::dim(&format!("copied {what}")),
                     style::faint(&format!("· {} lines · via {tool}", text.lines().count()))));
    Ok(())
}

/// Why a git command should be confirmed first, if it should.
pub fn git_risk(args: &str) -> Option<&'static str> {
    let a = format!(" {} ", args.to_lowercase());
    let rules: &[(&[&str], &str)] = &[
        (&[" push "], "publishes your commits"),
        (&[" reset --hard", " clean -f", " clean -d", " clean -x", " checkout -- ", " checkout . ", " restore . ", " restore --staged ."],
         "throws away changes that can't be brought back"),
        (&[" branch -d ", " branch -D ", " branch --delete "], "deletes a branch"),
        (&[" rebase ", " filter-branch", " reflog expire", " gc --prune"], "rewrites history"),
        (&[" stash drop", " stash clear"], "throws away stashed changes"),
    ];
    rules.iter().find(|(pats, _)| pats.iter().any(|p| a.contains(&p.to_lowercase()))).map(|(_, why)| *why)
}

/// `/git [args]`: git, with a short default and a question before anything that can't be undone.
pub fn git(ctx: &Ctx, args: &str) -> Result<()> {
    let a = args.trim();
    let line = match a {
        "" | "status" | "st" => "git status --short --branch".to_string(),
        "log" => "git log --oneline --graph --decorate -20".to_string(),
        "diff" => "git diff --stat".to_string(),
        "branches" | "branch" => "git branch -vv".to_string(),
        other => format!("git {other}"),
    };
    if crate::proc::which("git").is_none() {
        bail!("git isn't installed");
    }
    if let Some(why) = git_risk(a) {
        if ctx.ask(&format!("`{line}` {why}. Run it?"), &[('y', "yes"), ('n', "no")]) != 'y' {
            ctx.line(format!("  {}", style::dim("didn't run it")));
            return Ok(());
        }
    }
    crate::actions::run(ctx, &line)
}

/// `/safety`: what Aegist will and won't do on your computer.
pub fn safety(ctx: &Ctx) {
    let a = &ctx.settings.agent;
    let rows: [(&str, String); 9] = [
        ("asks", format!("before its first mouse or keyboard action each session{}", if a.confirm { "" } else { " (off: agent.confirm is false)" })),
        ("asks", "before keys that close, save, print, lock or delete, before typing line breaks, and before git commands that can't be undone".into()),
        ("stops", format!("the moment you move the mouse more than {}px from where it put it", a.failsafe_px)),
        ("stops", "when you press esc, and after agent.max_actions actions in one run".to_string() + &format!(" ({})", a.max_actions)),
        ("never", "clicks somewhere it guessed: it can't read your screen's text, so it asks you where".into()),
        ("never", "opens or handles passwords, sign-ins, payments, banking or anything that sends messages in your name".into()),
        ("never", "presses ctrl+alt+delete, or acts outside the paint app's window when using a learned app".into()),
        ("plans", "several steps all at once, shows you the plan, and runs none of it if one step isn't understood".into()),
        ("code", "is checked (compiler, made-up names, tests, certainty) before it touches your files - or it says I don't know".into()),
    ];
    let mut lines = vec![String::new(), format!("  {}", widgets::heading("Safety protocol", 0.3, ctx.width().saturating_sub(4)))];
    for (k, v) in rows {
        let color = match k {
            "never" => pal::RED,
            "stops" => pal::YELLOW,
            "asks" => pal::SKY,
            _ => pal::GREEN,
        };
        for (i, l) in text::wrap(&v, ctx.width().saturating_sub(16)).into_iter().enumerate() {
            let label = if i == 0 { Style::new().fg(color).bold().paint(&text::pad(k, 6)) } else { " ".repeat(6) };
            lines.push(format!("    {label}  {}", style::dim(&l)));
        }
    }
    ctx.print(lines);
}

// ---------------------------------------------------------------- the app

/// The Aegist app window (aegist-app) next to `exe`, if it was built or downloaded there.
pub fn app_beside(exe: &Path) -> Option<PathBuf> {
    let dir = exe.parent()?;
    let ext = if cfg!(windows) { ".exe" } else { "" };
    let direct = dir.join(format!("aegist-app{ext}"));
    if direct.is_file() {
        return Some(direct);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.to_lowercase().starts_with("aegist-app") && n.ends_with(ext)))
        .collect();
    found.sort();
    found.into_iter().next()
}

/// A Linux desktop entry: the Aegist app window when there is one, else
/// Aegist in a terminal.
pub fn desktop_entry(exe: &Path, app: Option<&Path>, icon: Option<&Path>) -> String {
    let (run, terminal) = match app {
        Some(a) => (a, "false"),
        None => (exe, "true"),
    };
    let icon = icon.map_or("utilities-terminal".to_string(), |p| p.display().to_string());
    format!("[Desktop Entry]\nType=Application\nName=Aegist\nGenericName=Coding AI\nComment=A coding AI grown from scratch - write, fix and run code, and use your screen\n\
             Exec=\"{}\"\nTerminal={terminal}\nIcon={icon}\nCategories=Development;Utility;\nKeywords=ai;code;terminal;agent;\nStartupNotify=true\n\
             StartupWMClass=aegist-app\n",
            run.display())
}

/// A PowerShell script that makes Start menu and desktop shortcuts on
/// Windows: to the Aegist app window when there is one, else to Aegist in
/// Windows Terminal (titled Aegist) or a console.
pub fn windows_shortcut_script(exe: &Path, app: Option<&Path>, wt: Option<&Path>, icon: Option<&Path>) -> String {
    let q = |p: &Path| p.display().to_string().replace('\'', "''");
    let (target, args) = match (app, wt) {
        (Some(a), _) => (q(a), String::new()),
        (None, Some(wt)) => (q(wt), format!("--title Aegist \"{}\"", exe.display()).replace('\'', "''")),
        (None, None) => (q(exe), String::new()),
    };
    let icon = icon.map_or_else(|| format!("{},0", q(app.unwrap_or(exe))), q);
    format!(
        "$ws = New-Object -ComObject WScript.Shell\n\
         foreach ($dir in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) {{\n\
         \x20 $lnk = Join-Path $dir 'Aegist.lnk'\n\
         \x20 $s = $ws.CreateShortcut($lnk)\n\
         \x20 $s.TargetPath = '{target}'\n\
         \x20 $s.Arguments = '{args}'\n\
         \x20 $s.WorkingDirectory = [Environment]::GetFolderPath('UserProfile')\n\
         \x20 $s.IconLocation = '{icon}'\n\
         \x20 $s.Description = 'Aegist - a coding AI grown from scratch'\n\
         \x20 $s.Save()\n\
         \x20 Write-Output $lnk\n\
         }}\n"
    )
}

/// `aegist install`: add Aegist to the apps menu (and desktop), opening the
/// Aegist app window when it's next to this program. Returns what was made.
pub fn install() -> Result<Vec<PathBuf>> {
    let exe = std::env::current_exe()?.canonicalize()?;
    let app = app_beside(&exe);
    let home = crate::config::home_dir().ok_or_else(|| anyhow::anyhow!("couldn't find your home folder"))?;
    if cfg!(windows) {
        // the icon, for the shortcuts
        let dir = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|| home.join("AppData").join("Local")).join("Aegist");
        std::fs::create_dir_all(&dir)?;
        let ico = dir.join("aegist.ico");
        std::fs::write(&ico, crate::icon::ico())?;
        let wt = crate::proc::which("wt");
        let script = windows_shortcut_script(&exe, app.as_deref(), wt.as_deref(), Some(&ico));
        let out = std::process::Command::new("powershell").args(["-NoProfile", "-NonInteractive", "-Command", &script]).output()
            .context("running PowerShell to make the shortcuts")?;
        if !out.status.success() {
            bail!("PowerShell couldn't make the shortcuts: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        let mut made: Vec<PathBuf> = String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.trim().is_empty()).map(PathBuf::from).collect();
        made.push(ico);
        return Ok(made);
    }
    if cfg!(target_os = "macos") {
        let dir = home.join("Applications");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("Aegist.command");
        let run = app.as_deref().unwrap_or(&exe);
        std::fs::write(&path, format!("#!/bin/sh\nprintf '\\033]0;Aegist\\007'\ncd \"$HOME\"\nexec \"{}\"\n", run.display()))?;
        make_executable(&path)?;
        return Ok(vec![path]);
    }
    // (an empty or relative XDG_DATA_HOME is to be ignored, per the spec)
    let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).filter(|p| p.is_absolute()).unwrap_or_else(|| home.join(".local/share"));
    let apps = data.join("applications");
    std::fs::create_dir_all(&apps)?;
    let icons = data.join("icons/hicolor/256x256/apps");
    std::fs::create_dir_all(&icons)?;
    let icon = icons.join("aegist.png");
    std::fs::write(&icon, crate::icon::png(256))?;
    let entry = desktop_entry(&exe, app.as_deref(), Some(&icon));
    let mut made = vec![apps.join("aegist.desktop")];
    std::fs::write(&made[0], &entry)?;
    let desktop = home.join("Desktop");
    if desktop.is_dir() {
        let p = desktop.join("aegist.desktop");
        std::fs::write(&p, &entry)?;
        make_executable(&p)?;
        made.push(p);
    }
    made.push(icon);
    Ok(made)
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
    }
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_change_in_place_keeping_comments() {
        let text = "inference:\n  temperature: 0.2           # 0 = greedy\n  top_p: 0.95\nhonesty:\n  samples: 4                 # candidates\n  min_confidence: 0.35\n";
        let out = set_yaml_value(text, "honesty.samples", "6").unwrap();
        assert!(out.contains("  samples: 6                 # candidates\n"), "{out}");
        assert!(out.contains("temperature: 0.2") && out.ends_with("min_confidence: 0.35\n"));
        let out = set_yaml_value(text, "inference.temperature", "0").unwrap();
        assert!(out.contains("  temperature: 0             # 0 = greedy"), "{out}");
        assert!(set_yaml_value(text, "honesty.nope", "1").is_err());
        assert!(set_yaml_value(text, "top_p", "1").is_err());
        // the real settings file: every setting /config offers is in it and still loads when changed
        let real = crate::config::DEFAULT_SETTINGS;
        for (k, _) in TUNABLE {
            let v = if k.ends_with("confirm") || k.ends_with("enabled") || k.ends_with("speculative") { "true" } else if k.ends_with("precision") { "int8" } else { "3" };
            let changed = set_yaml_value(real, k, v).unwrap_or_else(|e| panic!("{k}: {e}"));
            Settings::from_str(&changed, Path::new("/tmp")).unwrap_or_else(|e| panic!("{k}={v}: {e}"));
        }
    }

    #[test]
    fn dates_and_risky_git() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_722), (2026, 9, 26));
        assert_eq!(civil(11_016), (2000, 2, 29));
        assert!(git_risk("push origin main").is_some());
        assert!(git_risk("reset --hard HEAD~1").is_some());
        assert!(git_risk("status").is_none() && git_risk("log --oneline").is_none() && git_risk("commit -m 'push it'").is_none());
    }

    #[test]
    fn launchers_point_at_this_program() {
        let exe = Path::new("/opt/aegist/aegist");
        let e = desktop_entry(exe, None, None);
        assert!(e.starts_with("[Desktop Entry]") && e.contains("Exec=\"/opt/aegist/aegist\"") && e.contains("Terminal=true"));
        let e = desktop_entry(exe, Some(Path::new("/opt/aegist/aegist-app")), Some(Path::new("/i/aegist.png")));
        assert!(e.contains("Exec=\"/opt/aegist/aegist-app\"") && e.contains("Terminal=false") && e.contains("Icon=/i/aegist.png"));
        let s = windows_shortcut_script(Path::new(r"C:\Tools\it's\aegist.exe"), None, Some(Path::new(r"C:\wt.exe")), None);
        assert!(s.contains(r"TargetPath = 'C:\wt.exe'") && s.contains(r#"--title Aegist "C:\Tools\it''s\aegist.exe""#), "{s}");
        let s = windows_shortcut_script(Path::new(r"C:\a.exe"), None, None, None);
        assert!(s.contains(r"TargetPath = 'C:\a.exe'") && s.contains("Arguments = ''") && s.contains(r"IconLocation = 'C:\a.exe,0'"));
        // the app window wins, with the Aegist icon
        let s = windows_shortcut_script(Path::new(r"C:\a.exe"), Some(Path::new(r"C:\aegist-app.exe")), Some(Path::new(r"C:\wt.exe")),
                                        Some(Path::new(r"C:\Aegist\aegist.ico")));
        assert!(s.contains(r"TargetPath = 'C:\aegist-app.exe'") && s.contains("Arguments = ''") && s.contains(r"IconLocation = 'C:\Aegist\aegist.ico'"), "{s}");
        // found next to the program, under a downloaded name too
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("aegist");
        std::fs::write(&main, "").unwrap();
        assert!(app_beside(&main).is_none());
        let name = if cfg!(windows) { "aegist-app-windows-x86_64.exe" } else { "aegist-app-linux-x86_64" };
        std::fs::write(tmp.path().join(name), "").unwrap();
        assert_eq!(app_beside(&main).unwrap().file_name().unwrap(), name);
    }
}
