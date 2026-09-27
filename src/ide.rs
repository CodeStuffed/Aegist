//! The workbench around the model: run programs, keep servers and games
//! running in the background, serve a website folder, look at a page in a
//! headless browser (drawn right in the terminal, with its console errors),
//! and find and run a project's tests.

use crate::lang;
use crate::proc::{self, quote};
use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ------------------------------------------------------------------ tests

/// The command that runs this project's tests, if it has any.
pub fn test_command(root: &Path) -> Option<String> {
    let has = |f: &str| root.join(f).exists();
    if has("Cargo.toml") {
        return Some("cargo test".into());
    }
    if has("package.json") {
        let pkg = std::fs::read_to_string(root.join("package.json")).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&pkg).unwrap_or_default();
        if let Some(t) = v.pointer("/scripts/test").and_then(|t| t.as_str()) {
            if !t.contains("no test specified") {
                return Some("npm test --silent".into());
            }
        }
    }
    if has("go.mod") {
        return Some("go test ./...".into());
    }
    let python_tests = has("pytest.ini") || has("pyproject.toml") || has("setup.py") || has("tox.ini") || has("tests") || has("test")
        || std::fs::read_dir(root).map(|d| d.flatten().any(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            (n.starts_with("test_") || n.ends_with("_test.py")) && n.ends_with(".py")
        })).unwrap_or(false);
    if python_tests {
        let py = proc::python().unwrap_or(if cfg!(windows) { "python" } else { "python3" });
        let has_pytest = proc::run(
            { let mut c = Command::new(py); c.args(["-c", "import pytest"]); c },
            Some(Duration::from_secs(10)), None,
        ).is_ok_and(|o| o.ok());
        return Some(if has_pytest { format!("{py} -m pytest -q") } else { format!("{py} -m unittest discover -q") });
    }
    if let Ok(mk) = std::fs::read_to_string(root.join("Makefile")) {
        if mk.lines().any(|l| l.starts_with("test:") || l.starts_with("check:")) {
            return Some(if mk.lines().any(|l| l.starts_with("test:")) { "make test" } else { "make check" }.into());
        }
    }
    if has("pom.xml") {
        return Some("mvn -q test".into());
    }
    if has("build.gradle") || has("build.gradle.kts") {
        return Some(if has("gradlew") { "./gradlew test" } else { "gradle test" }.into());
    }
    if std::fs::read_dir(root).map(|d| d.flatten().any(|e| e.path().extension().is_some_and(|x| x == "csproj" || x == "sln"))).unwrap_or(false) {
        return Some("dotnet test".into());
    }
    None
}

/// A shell command that runs a single source file, if its language can be
/// run that way (compiled languages are built to a temporary program first).
pub fn run_file_command(path: &Path) -> Option<String> {
    let lang = lang::for_path(path)?;
    let python: [&str; 2];
    let template: &[&str] = if lang.id == "python" {
        python = [proc::python()?, "{}"];
        &python
    } else {
        lang.run.iter().find(|t| proc::which(t[0]).is_some())?
    };
    let out = std::env::temp_dir().join(format!("aegist-run-{}", std::process::id()));
    let out = if cfg!(windows) { out.with_extension("exe") } else { out };
    let file = path.to_string_lossy();
    let parts: Vec<String> = template
        .iter()
        .map(|a| match *a {
            "&&" => "&&".to_string(),
            a => quote(&a.replace("{}", &file).replace("{out}", &out.to_string_lossy())),
        })
        .collect();
    Some(parts.join(" "))
}

// ------------------------------------------------------------------- jobs

/// A program running in the background.
pub struct Job {
    pub id: usize,
    pub command: String,
    pub started: Instant,
    child: Arc<Mutex<Option<Child>>>,
    output: Arc<Mutex<VecDeque<String>>>,
    exit: Arc<Mutex<Option<Option<i32>>>>,
    url: Arc<Mutex<Option<String>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobState {
    Running,
    Exited(Option<i32>),
}

const JOB_LINES: usize = 2000;

impl Job {
    pub fn state(&self) -> JobState {
        match *self.exit.lock().expect("lock") {
            None => JobState::Running,
            Some(code) => JobState::Exited(code),
        }
    }

    /// The last `n` lines it printed.
    pub fn tail(&self, n: usize) -> Vec<String> {
        let out = self.output.lock().expect("lock");
        out.iter().skip(out.len().saturating_sub(n)).cloned().collect()
    }

    /// A local address it printed (a dev server's URL), if any.
    pub fn url(&self) -> Option<String> {
        self.url.lock().expect("lock").clone()
    }

    pub fn stop(&self) {
        if let Some(child) = self.child.lock().expect("lock").as_mut() {
            proc::kill_tree(child);
        }
    }
}

#[derive(Default)]
pub struct Jobs {
    pub list: Vec<Job>,
    next: usize,
}

impl Jobs {
    /// Start `command` (a shell command line) in `cwd`.
    pub fn start(&mut self, command: &str, cwd: &Path) -> Result<usize> {
        let mut cmd = proc::shell(command);
        cmd.current_dir(cwd).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.env("PYTHONUNBUFFERED", "1").env("FORCE_COLOR", "0");
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().with_context(|| format!("couldn't start `{command}`"))?;
        self.next += 1;
        let id = self.next;
        let output = Arc::new(Mutex::new(VecDeque::new()));
        let url = Arc::new(Mutex::new(None));
        let url_re = regex::Regex::new(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1?\])(?::\d+)?[^\s'\x22)]*").expect("valid regex");
        for stream in [child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>)]
            .into_iter()
            .flatten()
        {
            let (output, url, re) = (output.clone(), url.clone(), url_re.clone());
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    let clean = crate::ui::text::strip_ansi(&line);
                    if let Some(m) = re.find(&clean) {
                        let mut u = url.lock().expect("lock");
                        if u.is_none() {
                            *u = Some(m.as_str().replace("0.0.0.0", "localhost"));
                        }
                    }
                    let mut out = output.lock().expect("lock");
                    if out.len() >= JOB_LINES {
                        out.pop_front();
                    }
                    out.push_back(clean);
                }
            });
        }
        let child = Arc::new(Mutex::new(Some(child)));
        let exit = Arc::new(Mutex::new(None));
        {
            let (child, exit) = (child.clone(), exit.clone());
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_millis(150));
                let mut guard = child.lock().expect("lock");
                match guard.as_mut().map(|c| c.try_wait()) {
                    Some(Ok(Some(status))) => {
                        *exit.lock().expect("lock") = Some(status.code());
                        return;
                    }
                    Some(Ok(None)) => {}
                    _ => {
                        *exit.lock().expect("lock") = Some(None);
                        return;
                    }
                }
            });
        }
        self.list.push(Job { id, command: command.to_string(), started: Instant::now(), child, output, exit, url });
        Ok(id)
    }

    pub fn get(&self, id: usize) -> Option<&Job> {
        self.list.iter().find(|j| j.id == id)
    }

    pub fn stop_all(&self) {
        for j in &self.list {
            j.stop();
        }
    }
}

// ----------------------------------------------------------------- serve

/// A folder served over HTTP on this computer only.
pub struct Server {
    pub dir: PathBuf,
    pub url: String,
    stop: Arc<AtomicBool>,
}

impl Server {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("html" | "htm") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("wasm") => "application/wasm",
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("ogg") => "audio/ogg",
        Some("mp4") => "video/mp4",
        Some("webm") => "video/webm",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("txt" | "md" | "py" | "rs" | "c" | "h") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn percent_decode(s: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if let (Some(h), Some(l)) = (bytes.get(i + 1).and_then(|&b| hex(b)), bytes.get(i + 2).and_then(|&b| hex(b))) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn respond(mut stream: TcpStream, root: &Path) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut head = Vec::new();
    let mut buf = [0u8; 4096];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < 16 * 1024 {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => head.extend_from_slice(&buf[..n]),
        }
    }
    let head = String::from_utf8_lossy(&head);
    let mut parts = head.lines().next().unwrap_or("").split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    let reply = |stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8], send_body: bool| {
        let header = format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len());
        let _ = stream.write_all(header.as_bytes());
        if send_body {
            let _ = stream.write_all(body);
        }
    };
    if method != "GET" && method != "HEAD" {
        reply(&mut stream, "405 Method Not Allowed", "text/plain", b"method not allowed", true);
        return;
    }
    let path = percent_decode(target.split(['?', '#']).next().unwrap_or("/"));
    let mut file = root.to_path_buf();
    for part in path.split('/').filter(|p| !p.is_empty() && *p != "." && *p != "..") {
        file.push(part);
    }
    if file.is_dir() {
        file.push("index.html");
    }
    let inside = file.canonicalize().ok().zip(root.canonicalize().ok()).is_some_and(|(f, r)| f.starts_with(r));
    match std::fs::read(&file) {
        Ok(body) if inside => reply(&mut stream, "200 OK", content_type(&file), &body, method == "GET"),
        _ => reply(&mut stream, "404 Not Found", "text/plain; charset=utf-8", format!("not found: {path}").as_bytes(), method == "GET"),
    }
}

/// Serve `dir` on http://localhost (port 8000 or the next free one).
pub fn serve(dir: &Path) -> Result<Server> {
    if !dir.is_dir() {
        bail!("{} isn't a folder", dir.display());
    }
    let listener = (8000..8100).find_map(|p| TcpListener::bind(("127.0.0.1", p)).ok()).map_or_else(|| TcpListener::bind(("127.0.0.1", 0)), Ok)?;
    let port = listener.local_addr()?.port();
    listener.set_nonblocking(true)?;
    let stop = Arc::new(AtomicBool::new(false));
    let (root, flag) = (dir.to_path_buf(), stop.clone());
    std::thread::spawn(move || {
        while !flag.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let root = root.clone();
                    std::thread::spawn(move || respond(stream, &root));
                }
                Err(_) => std::thread::sleep(Duration::from_millis(25)),
            }
        }
    });
    Ok(Server { dir: dir.to_path_buf(), url: format!("http://localhost:{port}/"), stop })
}

// --------------------------------------------------------------- preview

/// A Chrome, Chromium or Edge to look at pages with.
pub fn find_browser() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AEGIST_BROWSER").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "microsoft-edge", "msedge", "brave-browser"] {
        if let Some(p) = proc::which(name) {
            return Some(p);
        }
    }
    let mut candidates: Vec<PathBuf> = vec![
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into(),
        "/Applications/Chromium.app/Contents/MacOS/Chromium".into(),
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge".into(),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe".into(),
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe".into(),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe".into(),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe".into(),
    ];
    // browsers Playwright installed
    for base in [std::env::var_os("PLAYWRIGHT_BROWSERS_PATH").map(PathBuf::from), Some("/opt/pw-browsers".into()),
                 crate::config::home_dir().map(|h| h.join(".cache/ms-playwright"))].into_iter().flatten() {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for e in entries.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("chromium-") {
                    candidates.push(e.path().join("chrome-linux/chrome"));
                    candidates.push(e.path().join("chrome-mac/Chromium.app/Contents/MacOS/Chromium"));
                    candidates.push(e.path().join("chrome-win/chrome.exe"));
                }
            }
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

pub struct Shot {
    pub png: PathBuf,
    /// What the page printed to its console, errors included.
    pub console: Vec<String>,
}

/// A local file becomes a file:// URL; URLs stay as they are.
pub fn to_url(target: &str, cwd: &Path) -> String {
    if target.contains("://") {
        return target.to_string();
    }
    if target.starts_with("localhost") || target.starts_with("127.0.0.1") {
        return format!("http://{target}");
    }
    let p = cwd.join(target);
    let p = p.canonicalize().unwrap_or(p);
    let s = p.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') { format!("file://{s}") } else { format!("file:///{s}") }
}

/// Load `url` in a headless browser, wait `wait_ms` for it to run, and
/// take a `w` x `h` screenshot.
pub fn screenshot(url: &str, w: u32, h: u32, wait_ms: u32) -> Result<Shot> {
    let Some(browser) = find_browser() else {
        bail!("no Chrome, Chromium or Edge found to look at pages with (set AEGIST_BROWSER to one)");
    };
    let dir = std::env::temp_dir().join(format!("aegist-preview-{}-{}", std::process::id(), Instant::now().elapsed().as_nanos()));
    std::fs::create_dir_all(&dir)?;
    let png = dir.join("shot.png");
    let mut cmd = Command::new(&browser);
    // the page's console messages come from the browser's log. On Windows
    // the browser has no stderr to write it to, so it goes to a file there.
    let log_file = dir.join("browser.log");
    cmd.args(["--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", "--mute-audio", "--no-first-run",
              "--no-default-browser-check", "--log-level=0"]);
    if cfg!(windows) {
        cmd.arg("--enable-logging").arg(format!("--log-file={}", log_file.display()));
    } else {
        cmd.arg("--enable-logging=stderr");
    }
    cmd
        .arg(format!("--user-data-dir={}", dir.join("profile").display()))
        .arg(format!("--window-size={w},{h}"))
        .arg(format!("--virtual-time-budget={wait_ms}"))
        .arg(format!("--screenshot={}", png.display()))
        .arg(url);
    let out = proc::run(cmd, Some(Duration::from_secs(60)), None)?;
    if !png.is_file() {
        bail!("the browser didn't produce a screenshot: {}", out.combined().lines().last().unwrap_or("no output"));
    }
    let re = regex::Regex::new(r#"CONSOLE(?::\d+|\(\d+\))\] "(.*)", source: (.*?) \((\d+)\)"#).expect("valid regex");
    let logged = format!("{}\n{}", out.stderr, std::fs::read_to_string(&log_file).unwrap_or_default());
    let mut seen = std::collections::HashSet::new();
    let console = logged.lines().filter_map(|l| re.captures(l)).map(|c| {
        let src = c[2].rsplit('/').next().unwrap_or(&c[2]).to_string();
        format!("{} ({src}:{})", &c[1], &c[3])
    }).filter(|l| seen.insert(l.clone())).collect();
    Ok(Shot { png, console })
}

/// A PNG's pixels: (width, height, row-major colors).
pub fn load_png(path: &Path) -> Result<(usize, usize, Vec<crate::ui::style::Rgb>)> {
    let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        _ => 1,
    };
    let pixels = (0..w * h)
        .map(|p| {
            let i = p * channels;
            match channels {
                1 | 2 => crate::ui::style::Rgb(buf[i], buf[i], buf[i]),
                _ => crate::ui::style::Rgb(buf[i], buf[i + 1], buf[i + 2]),
            }
        })
        .collect();
    Ok((w, h, pixels))
}

/// A PNG drawn with half-block characters: each character cell shows two
/// pixels, one above the other. `cols` wide; height follows the image.
pub fn render_png(path: &Path, cols: usize, max_rows: usize) -> Result<Vec<String>> {
    let (w, h, pixels) = load_png(path)?;
    Ok(render_pixels(w, h, &pixels, cols, max_rows))
}

/// `w`×`h` pixels (row by row) drawn with half-block characters.
pub fn render_pixels(w: usize, h: usize, pixels: &[crate::ui::style::Rgb], cols: usize, max_rows: usize) -> Vec<String> {
    if w == 0 || h == 0 || pixels.len() < w * h {
        return Vec::new();
    }
    let cols = cols.min(w).max(1);
    let scale = w as f32 / cols as f32;
    let rows = ((h as f32 / scale / 2.0).ceil() as usize).min(max_rows).max(1);
    // average the block of source pixels behind each target pixel
    let sample = |tx: usize, ty: usize| -> crate::ui::style::Rgb {
        let (x0, x1) = ((tx as f32 * scale) as usize, ((tx + 1) as f32 * scale) as usize);
        let (y0, y1) = ((ty as f32 * scale) as usize, ((ty + 1) as f32 * scale) as usize);
        let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
        for y in y0.min(h - 1)..y1.max(y0 + 1).min(h) {
            for x in x0.min(w - 1)..x1.max(x0 + 1).min(w) {
                let p = pixels[y * w + x];
                r += p.0 as u32;
                g += p.1 as u32;
                b += p.2 as u32;
                n += 1;
            }
        }
        let n = n.max(1);
        crate::ui::style::Rgb((r / n) as u8, (g / n) as u8, (b / n) as u8)
    };
    let mut out = Vec::new();
    for row in 0..rows {
        let mut line = String::new();
        for col in 0..cols {
            let (top, bottom) = (sample(col, row * 2), sample(col, row * 2 + 1));
            line.push_str(&crate::ui::style::Style::new().fg(top).bg(bottom).paint("▀"));
        }
        out.push(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_test_commands() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(test_command(tmp.path()), None);
        std::fs::write(tmp.path().join("package.json"), r#"{"scripts": {"test": "echo \"Error: no test specified\" && exit 1"}}"#).unwrap();
        assert_eq!(test_command(tmp.path()), None);
        std::fs::write(tmp.path().join("package.json"), r#"{"scripts": {"test": "jest"}}"#).unwrap();
        assert_eq!(test_command(tmp.path()).unwrap(), "npm test --silent");
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        assert_eq!(test_command(tmp.path()).unwrap(), "cargo test");
    }

    #[test]
    fn serves_a_folder_safely() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("site/css")).unwrap();
        std::fs::write(tmp.path().join("site/index.html"), "<h1>hi</h1>").unwrap();
        std::fs::write(tmp.path().join("site/css/a b.css"), "body{}").unwrap();
        std::fs::write(tmp.path().join("secret.txt"), "no").unwrap();
        let server = serve(&tmp.path().join("site")).unwrap();
        let get = |path: &str| -> String {
            let addr = server.url.trim_start_matches("http://").trim_end_matches('/').replace("localhost", "127.0.0.1");
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes()).unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            out
        };
        let home = get("/");
        assert!(home.starts_with("HTTP/1.1 200") && home.contains("text/html") && home.ends_with("<h1>hi</h1>"));
        assert!(get("/css/a%20b.css").contains("text/css"));
        assert!(get("/../secret.txt").starts_with("HTTP/1.1 404"));
        assert!(get("/nope.js").starts_with("HTTP/1.1 404"));
        server.stop();
    }

    #[cfg(unix)]
    #[test]
    fn background_jobs_capture_output_and_stop() {
        let tmp = tempfile::tempdir().unwrap();
        let mut jobs = Jobs::default();
        let id = jobs.start("echo 'Listening on http://localhost:5173/'; sleep 30", tmp.path()).unwrap();
        let job = jobs.get(id).unwrap();
        let t = Instant::now();
        while job.url().is_none() && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(job.url().as_deref(), Some("http://localhost:5173/"));
        assert_eq!(job.state(), JobState::Running);
        job.stop();
        let t = Instant::now();
        while job.state() == JobState::Running && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(matches!(job.state(), JobState::Exited(_)));
        let quick = jobs.start("echo done", tmp.path()).unwrap();
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(jobs.get(quick).unwrap().tail(5), vec!["done"]);
    }

    #[test]
    fn previews_pages_when_a_browser_is_installed() {
        let Some(_) = find_browser() else { return };
        let tmp = tempfile::tempdir().unwrap();
        let page = tmp.path().join("p.html");
        std::fs::write(&page, "<body style='margin:0;background:#ff0000'><script>console.log('ready'); nope();</script></body>").unwrap();
        let shot = screenshot(&to_url("p.html", tmp.path()), 64, 48, 500).unwrap();
        assert!(shot.console.iter().any(|l| l.starts_with("ready")), "{:?}", shot.console);
        assert!(shot.console.iter().any(|l| l.contains("nope is not defined")));
        let (w, h, px) = load_png(&shot.png).unwrap();
        assert_eq!((w, h), (64, 48));
        assert_eq!(px[0], crate::ui::style::Rgb(255, 0, 0));
        let rows = render_png(&shot.png, 16, 20).unwrap();
        assert_eq!(rows.len(), 6);
        assert!(rows.iter().all(|r| crate::ui::text::width(r) == 16));
    }
}
