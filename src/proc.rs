//! Running programs: with a time limit, cancellable, output captured, and
//! the whole process tree stopped when it's over (a test runner's children
//! don't outlive it).

use anyhow::{Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default)]
pub struct Output {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub cancelled: bool,
    pub seconds: f64,
}

impl Output {
    pub fn ok(&self) -> bool {
        self.code == Some(0) && !self.timed_out && !self.cancelled
    }

    /// stdout then stderr, trimmed.
    pub fn combined(&self) -> String {
        let (o, e) = (self.stdout.trim_end(), self.stderr.trim_end());
        match (o.is_empty(), e.is_empty()) {
            (true, _) => e.to_string(),
            (_, true) => o.to_string(),
            _ => format!("{o}\n{e}"),
        }
    }
}

/// Where `program` is on PATH, if it's installed.
pub fn which(program: &str) -> Option<PathBuf> {
    let p = Path::new(program);
    if p.components().count() > 1 {
        return p.is_file().then(|| p.to_path_buf());
    }
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT").unwrap_or(".EXE;.CMD;.BAT;.COM".into()).split(';').map(|e| e.to_ascii_lowercase()).collect()
    } else {
        vec![String::new()]
    };
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|dir| {
        exts.iter().map(|e| dir.join(format!("{program}{e}"))).find(|c| c.is_file() && is_executable(c))
    })
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}
#[cfg(not(unix))]
fn is_executable(_: &Path) -> bool {
    true
}

/// A Python that really runs (on Windows, `python3` is often a stub that
/// offers to install Python from the Store), found once.
pub fn python() -> Option<&'static str> {
    static FOUND: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *FOUND.get_or_init(|| {
        let names: &[&'static str] = if cfg!(windows) { &["python", "python3", "py"] } else { &["python3", "python"] };
        names.iter().copied().find(|name| {
            which(name).is_some() && {
                let mut c = Command::new(name);
                c.args(["-c", "import sys; sys.exit(0)"]);
                run(c, Some(Duration::from_secs(15)), None).is_ok_and(|o| o.ok())
            }
        })
    })
}

/// A command line run by the system shell (sh, or cmd on Windows).
pub fn shell(line: &str) -> Command {
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", line]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", line]);
        c
    }
}

/// Quote an argument for the shell `shell` uses.
pub fn quote(arg: &str) -> String {
    if !arg.is_empty() && arg.chars().all(|c| c.is_ascii_alphanumeric() || "-_./=:,+@%".contains(c)) {
        return arg.to_string();
    }
    if cfg!(windows) {
        format!("\"{}\"", arg.replace('"', "\"\""))
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

fn prepare(cmd: &mut Command) {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.env("GIT_TERMINAL_PROMPT", "0").env("PYTHONUNBUFFERED", "1").env("PYTHONDONTWRITEBYTECODE", "1");
    #[cfg(unix)]
    {
        // its own process group, so stopping it stops everything it started
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
}

/// Stop a process and everything it started.
pub fn kill_tree(child: &mut Child) {
    #[cfg(unix)]
    // SAFETY: signalling our own child's process group.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill").args(["/T", "/F", "/PID", &child.id().to_string()]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Which stream a line of output came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

/// Run to completion (or `timeout`, or `cancel`), passing each line of
/// output to `on_line` as it arrives.
pub fn run_streaming(mut cmd: Command, timeout: Option<Duration>, cancel: Option<&AtomicBool>, on_line: &mut dyn FnMut(Stream, &str)) -> Result<Output> {
    prepare(&mut cmd);
    let start = Instant::now();
    let program = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd.spawn().with_context(|| format!("couldn't start {program}"))?;
    let (tx, rx) = mpsc::channel::<(Stream, Option<String>)>();
    let pipe = |mut r: Box<dyn Read + Send>, which: Stream, tx: mpsc::Sender<(Stream, Option<String>)>| {
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            let mut pending: Vec<u8> = Vec::new();
            loop {
                match r.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        pending.extend_from_slice(&buf[..n]);
                        while let Some(i) = pending.iter().position(|&b| b == b'\n') {
                            let line: Vec<u8> = pending.drain(..=i).collect();
                            let _ = tx.send((which, Some(String::from_utf8_lossy(&line).trim_end_matches(['\n', '\r']).to_string())));
                        }
                    }
                }
            }
            if !pending.is_empty() {
                let _ = tx.send((which, Some(String::from_utf8_lossy(&pending).into_owned())));
            }
            let _ = tx.send((which, None));
        })
    };
    let out_t = pipe(Box::new(child.stdout.take().expect("piped")), Stream::Out, tx.clone());
    let err_t = pipe(Box::new(child.stderr.take().expect("piped")), Stream::Err, tx);
    let mut result = Output::default();
    let mut open = 2;
    let mut status = None;
    let mut exited_at: Option<Instant> = None;
    loop {
        match rx.recv_timeout(Duration::from_millis(20)) {
            Ok((which, Some(line))) => {
                on_line(which, &line);
                let buf = if which == Stream::Out { &mut result.stdout } else { &mut result.stderr };
                // keep the output bounded: a runaway program can print forever
                if buf.len() < 4 << 20 {
                    buf.push_str(&line);
                    buf.push('\n');
                }
            }
            Ok((_, None)) => open -= 1,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => open = 0,
        }
        if status.is_none() {
            status = child.try_wait().ok().flatten();
            if status.is_some() {
                exited_at = Some(Instant::now());
            }
        }
        if status.is_some() && open == 0 {
            break;
        }
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            result.cancelled = true;
            kill_tree(&mut child);
            break;
        }
        if timeout.is_some_and(|t| start.elapsed() > t) {
            result.timed_out = true;
            kill_tree(&mut child);
            break;
        }
        if exited_at.is_some_and(|t| t.elapsed() > Duration::from_secs(2)) {
            // exited, but something it started still holds the pipes open
            kill_tree(&mut child);
            break;
        }
    }
    if status.is_none() {
        status = child.wait().ok();
    }
    drop(rx);
    let _ = (out_t, err_t); // readers end with the pipes; never block on them
    result.code = status.and_then(|s| s.code());
    result.seconds = start.elapsed().as_secs_f64();
    Ok(result)
}

/// Run to completion (or `timeout`, or `cancel`) and capture the output.
pub fn run(cmd: Command, timeout: Option<Duration>, cancel: Option<&AtomicBool>) -> Result<Output> {
    run_streaming(cmd, timeout, cancel, &mut |_, _| {})
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn runs_captures_times_out_and_cancels() {
        let out = run(shell("echo hi; echo oops >&2; exit 3"), Some(Duration::from_secs(10)), None).unwrap();
        assert_eq!((out.code, out.stdout.as_str(), out.stderr.as_str()), (Some(3), "hi\n", "oops\n"));
        assert!(!out.ok() && out.combined() == "hi\noops");
        let slow = run(shell("sleep 5; echo never"), Some(Duration::from_millis(300)), None).unwrap();
        assert!(slow.timed_out && slow.seconds < 3.0 && slow.stdout.is_empty());
        let stop = AtomicBool::new(true);
        assert!(run(shell("sleep 5"), None, Some(&stop)).unwrap().cancelled);
        let mut lines = Vec::new();
        run_streaming(shell("printf 'a\\nb\\nc'"), None, None, &mut |_, l| lines.push(l.to_string())).unwrap();
        assert_eq!(lines, ["a", "b", "c"]);
        assert!(which("sh").is_some() && which("no-such-program-aegist").is_none());
        assert_eq!(quote("plain"), "plain");
        assert_eq!(quote("it's"), "'it'\\''s'");
    }
}
