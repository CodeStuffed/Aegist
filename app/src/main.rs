//! Aegist's own window: the Aegist session in an app of its own, with its
//! icon, its colors and its font, instead of inside a terminal.
//!
//! The session itself runs unchanged: this program starts `aegist` in a
//! pseudo-terminal (ConPTY on Windows) and draws what it writes, so every
//! command works exactly as it does in a terminal.
//!
//! Keys: ctrl+shift+c / ctrl+shift+v copy and paste (selecting text with the
//! mouse copies it too, and right-click pastes), ctrl+= / ctrl+- / ctrl+0
//! change the text size, the wheel or shift+page up/down scroll back.

#![cfg_attr(windows, windows_subsystem = "windows")]

mod draw;
mod keys;
mod term;

use draw::{Fonts, Painter, View};
use keys::{Key, Mods};
use portable_pty::{CommandBuilder, MasterPty, PtySize};
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};
use term::Emulator;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key as WKey, ModifiersState, NamedKey};
use winit::window::{Icon, Theme, Window, WindowId};

const FONT_PT: f32 = 14.0;
const BLINK: Duration = Duration::from_millis(530);
const HINT: &str = "ctrl+shift+c/v copy·paste   ctrl +/- size";

enum UserEvent {
    Output(Vec<u8>),
    Exited { ok: bool },
}

/// The running session.
struct Session {
    master: Box<dyn MasterPty>,
    writer: Box<dyn Write + Send>,
    killer: Box<dyn portable_pty::ChildKiller + Send + Sync>,
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    args: Vec<String>,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    painter: Option<Painter>,
    emu: Emulator,
    session: Option<Session>,
    mods: ModifiersState,
    view: View,
    font_pt: f32,
    blink_at: Instant,
    mouse: (f64, f64),
    selecting: Option<(usize, usize)>,
    clipboard: Option<arboard::Clipboard>,
    /// The session ended; any key closes the window.
    ended: bool,
}

/// The `aegist` program: next to this one (under any name a download gave
/// it), or on the PATH.
fn find_aegist() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let ext = if cfg!(windows) { ".exe" } else { "" };
    let direct = dir.join(format!("aegist{ext}"));
    if direct.is_file() {
        return Some(direct);
    }
    // a downloaded release: aegist-windows-x86_64.exe and the like
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
                name.starts_with("aegist") && !name.starts_with("aegist-app") && name.ends_with(ext) && p.is_file()
                    && (cfg!(windows) || !name.contains('.'))
            })
            .collect();
        found.sort();
        if let Some(p) = found.into_iter().next() {
            return Some(p);
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(format!("aegist{ext}"))).find(|p| p.is_file())
}

/// Where the session starts: the folder you launched from, unless that's
/// just where the program lives or a system folder (as from a shortcut).
fn start_dir() -> PathBuf {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from);
    let cwd = std::env::current_dir().ok();
    let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf()));
    match (cwd, home) {
        (Some(c), Some(h)) if Some(&c) == exe_dir.as_ref() || c.to_string_lossy().to_lowercase().contains("system32") || c == std::path::Path::new("/") => h,
        (Some(c), _) => c,
        (None, Some(h)) => h,
        (None, None) => PathBuf::from("."),
    }
}

impl App {
    fn start_session(&mut self, cols: usize, rows: usize) -> Result<Session, String> {
        let program = find_aegist().ok_or("couldn't find the aegist program: put aegist-app next to aegist (or aegist on your PATH)")?;
        let pair = portable_pty::native_pty_system()
            .openpty(PtySize { rows: rows as u16, cols: cols as u16, pixel_width: 0, pixel_height: 0 })
            .map_err(|e| format!("couldn't open a terminal for the session: {e}"))?;
        let mut cmd = CommandBuilder::new(&program);
        cmd.args(&self.args);
        cmd.cwd(start_dir());
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        cmd.env("TERM_PROGRAM", "aegist-app");
        let mut child = pair.slave.spawn_command(cmd).map_err(|e| format!("couldn't start {}: {e}", program.display()))?;
        drop(pair.slave);
        let killer = child.clone_killer();
        let mut reader = pair.master.try_clone_reader().map_err(|e| e.to_string())?;
        let writer = pair.master.take_writer().map_err(|e| e.to_string())?;
        let proxy = self.proxy.clone();
        std::thread::spawn(move || {
            let mut buf = vec![0u8; 64 * 1024];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if proxy.send_event(UserEvent::Output(buf[..n].to_vec())).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        let proxy = self.proxy.clone();
        std::thread::spawn(move || {
            let ok = child.wait().map(|s| s.success()).unwrap_or(false);
            let _ = proxy.send_event(UserEvent::Exited { ok });
        });
        Ok(Session { master: pair.master, writer, killer })
    }

    fn send(&mut self, bytes: &[u8]) {
        if let Some(s) = self.session.as_mut() {
            let _ = s.writer.write_all(bytes);
            let _ = s.writer.flush();
        }
    }

    fn redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn fit(&mut self) {
        let (Some(w), Some(p)) = (&self.window, &self.painter) else { return };
        let size = w.inner_size();
        let (cols, rows) = p.grid_size(size.width as usize, size.height as usize);
        if (cols, rows) != (self.emu.term.cols, self.emu.term.rows) {
            self.emu.term.resize(cols, rows);
            if let Some(s) = &self.session {
                let _ = s.master.resize(PtySize { rows: rows as u16, cols: cols as u16, pixel_width: 0, pixel_height: 0 });
            }
        }
        self.redraw();
    }

    fn set_font(&mut self, pt: f32) {
        self.font_pt = pt.clamp(8.0, 36.0);
        if let Some(p) = self.painter.as_mut() {
            p.set_font_size(self.font_pt);
        }
        self.fit();
    }

    fn clipboard(&mut self) -> Option<&mut arboard::Clipboard> {
        if self.clipboard.is_none() {
            self.clipboard = arboard::Clipboard::new().ok();
        }
        self.clipboard.as_mut()
    }

    fn copy_selection(&mut self) {
        let Some((a, b)) = self.view.selection else { return };
        let text = self.emu.term.text_between(a, b);
        if !text.is_empty() {
            if let Some(c) = self.clipboard() {
                let _ = c.set_text(text);
            }
        }
    }

    fn paste(&mut self) {
        let text = self.clipboard().and_then(|c| c.get_text().ok()).unwrap_or_default();
        if !text.is_empty() {
            let bytes = keys::paste(&text, self.emu.term.bracketed_paste);
            self.view.offset = 0;
            self.send(&bytes);
        }
    }

    /// The line (counting from the oldest scrollback line) and column under the mouse.
    fn point(&self) -> Option<(usize, usize)> {
        let p = self.painter.as_ref()?;
        let t = &self.emu.term;
        let (col, row) = p.cell_at(self.mouse.0, self.mouse.1, t.cols, t.rows)?;
        let first = t.total_lines().saturating_sub(t.rows + self.view.offset);
        Some((first + row, col))
    }

    fn scroll_view(&mut self, lines: isize) {
        let max = self.emu.term.scrollback.len() as isize;
        self.view.offset = (self.view.offset as isize + lines).clamp(0, max) as usize;
        self.redraw();
    }

    fn paint(&mut self) {
        let (Some(w), Some(surface), Some(p)) = (&self.window, self.surface.as_mut(), self.painter.as_mut()) else { return };
        let size = w.inner_size();
        let (Some(nw), Some(nh)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else { return };
        if surface.resize(nw, nh).is_err() {
            return;
        }
        let Ok(mut buf) = surface.buffer_mut() else { return };
        p.draw(&mut buf, size.width as usize, size.height as usize, &self.emu.term, &self.view);
        let _ = buf.present();
        self.emu.term.dirty = false;
    }

    fn on_key(&mut self, event: winit::event::KeyEvent, el: &ActiveEventLoop) {
        if event.state != ElementState::Pressed {
            return;
        }
        if self.ended {
            el.exit();
            return;
        }
        let m = Mods { shift: self.mods.shift_key(), ctrl: self.mods.control_key(), alt: self.mods.alt_key() };
        // the window's own shortcuts
        if let WKey::Character(c) = &event.logical_key {
            let c = c.to_lowercase();
            if m.ctrl && m.shift && c == "c" {
                self.copy_selection();
                return;
            }
            if m.ctrl && m.shift && c == "v" {
                self.paste();
                return;
            }
            if m.ctrl && !m.alt && (c == "=" || c == "+") {
                self.set_font(self.font_pt + 1.0);
                return;
            }
            if m.ctrl && !m.alt && c == "-" && !m.shift {
                self.set_font(self.font_pt - 1.0);
                return;
            }
            if m.ctrl && !m.alt && c == "0" {
                self.set_font(FONT_PT);
                return;
            }
        }
        let key = match &event.logical_key {
            WKey::Named(n) => match n {
                NamedKey::Enter => Key::Enter,
                NamedKey::Backspace => Key::Backspace,
                NamedKey::Tab => Key::Tab,
                NamedKey::Escape => Key::Escape,
                NamedKey::Space => Key::Space,
                NamedKey::ArrowUp => Key::Up,
                NamedKey::ArrowDown => Key::Down,
                NamedKey::ArrowLeft => Key::Left,
                NamedKey::ArrowRight => Key::Right,
                NamedKey::Home => Key::Home,
                NamedKey::End => Key::End,
                NamedKey::PageUp if m.shift => return self.scroll_view(self.emu.term.rows as isize - 1),
                NamedKey::PageDown if m.shift => return self.scroll_view(-(self.emu.term.rows as isize - 1)),
                NamedKey::PageUp => Key::PageUp,
                NamedKey::PageDown => Key::PageDown,
                NamedKey::Insert if m.shift => return self.paste(),
                NamedKey::Insert => Key::Insert,
                NamedKey::Delete => Key::Delete,
                NamedKey::F1 => Key::F(1),
                NamedKey::F2 => Key::F(2),
                NamedKey::F3 => Key::F(3),
                NamedKey::F4 => Key::F(4),
                NamedKey::F5 => Key::F(5),
                NamedKey::F6 => Key::F(6),
                NamedKey::F7 => Key::F(7),
                NamedKey::F8 => Key::F(8),
                NamedKey::F9 => Key::F(9),
                NamedKey::F10 => Key::F(10),
                NamedKey::F11 => Key::F(11),
                NamedKey::F12 => Key::F(12),
                _ => return,
            },
            WKey::Character(c) if m.ctrl || m.alt => Key::Text(c.to_string()),
            WKey::Character(c) => Key::Text(event.text.as_ref().map_or_else(|| c.to_string(), |t| t.to_string())),
            _ => match &event.text {
                Some(t) => Key::Text(t.to_string()),
                None => return,
            },
        };
        if let Some(bytes) = keys::encode(&key, m, self.emu.term.app_cursor) {
            self.view.offset = 0;
            self.view.selection = None;
            self.view.cursor_on = true;
            self.blink_at = Instant::now() + BLINK;
            self.send(&bytes);
        }
    }
}

fn window_icon() -> Option<Icon> {
    Icon::from_rgba(aegist::icon::rgba(64), 64, 64).ok()
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Aegist")
            .with_inner_size(LogicalSize::new(1120.0, 740.0))
            .with_min_inner_size(LogicalSize::new(420.0, 260.0))
            .with_theme(Some(Theme::Dark))
            .with_window_icon(window_icon());
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_taskbar_icon(Icon::from_rgba(aegist::icon::rgba(256), 256, 256).ok())
        };
        let window = match el.create_window(attrs) {
            Ok(w) => Rc::new(w),
            Err(e) => {
                eprintln!("couldn't open the Aegist window: {e}");
                el.exit();
                return;
            }
        };
        let ctx = softbuffer::Context::new(window.clone()).expect("a drawing context for the window");
        self.surface = Some(softbuffer::Surface::new(&ctx, window.clone()).expect("a drawing surface for the window"));
        let fonts = match Fonts::load() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{e}");
                el.exit();
                return;
            }
        };
        self.painter = Some(Painter::new(fonts, self.font_pt, window.scale_factor() as f32));
        self.window = Some(window);
        let size = self.window.as_ref().expect("just made").inner_size();
        let (cols, rows) = self.painter.as_ref().expect("just made").grid_size(size.width as usize, size.height as usize);
        self.emu.term.resize(cols, rows);
        match self.start_session(cols, rows) {
            Ok(s) => self.session = Some(s),
            Err(e) => {
                self.emu.feed(format!("\x1b[31m✗\x1b[0m {e}\r\n\r\n\x1b[2mpress any key to close\x1b[0m").as_bytes());
                self.ended = true;
            }
        }
        self.view.focused = true;
        self.view.cursor_on = true;
        self.view.hint = HINT.into();
        self.redraw();
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Output(bytes) => {
                self.emu.feed(&bytes);
                let replies = self.emu.take_replies();
                if !replies.is_empty() {
                    self.send(&replies);
                }
                if self.view.offset > 0 {
                    // keep the same lines in view while output arrives below
                    self.view.offset = self.view.offset.min(self.emu.term.scrollback.len());
                }
                self.redraw();
            }
            UserEvent::Exited { ok } => {
                // the interactive session ended normally (/exit): close with it
                if ok && self.args.is_empty() {
                    _el.exit();
                    return;
                }
                // it ended with an error, or it was a one-off command: keep its output on screen
                self.emu.feed(b"\r\n\x1b[2m[the session ended - press any key to close]\x1b[0m");
                self.ended = true;
                self.session = None;
                self.redraw();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(s) = self.session.as_mut() {
                    let _ = s.killer.kill();
                }
                el.exit();
            }
            WindowEvent::Resized(_) => self.fit(),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(p) = self.painter.as_mut() {
                    p.set_scale(scale_factor as f32, self.font_pt);
                }
                self.fit();
            }
            WindowEvent::ModifiersChanged(m) => self.mods = m.state(),
            WindowEvent::Focused(f) => {
                self.view.focused = f;
                self.redraw();
            }
            WindowEvent::KeyboardInput { event, is_synthetic: false, .. } => self.on_key(event, el),
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => (y * 3.0).round() as isize,
                    MouseScrollDelta::PixelDelta(p) => (p.y / self.painter.as_ref().map_or(16.0, |p| p.cell_h as f64)).round() as isize,
                };
                self.scroll_view(lines);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse = (position.x, position.y);
                if let (Some(anchor), Some(now)) = (self.selecting, self.point()) {
                    self.view.selection = Some((anchor, now));
                    self.redraw();
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => match state {
                ElementState::Pressed => {
                    self.selecting = self.point();
                    self.view.selection = None;
                    self.redraw();
                }
                ElementState::Released => {
                    self.selecting = None;
                    if self.view.selection.is_some_and(|(a, b)| a != b) {
                        self.copy_selection();
                    } else {
                        self.view.selection = None;
                    }
                    self.redraw();
                }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Right, .. } => self.paste(),
            WindowEvent::RedrawRequested => self.paint(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let now = Instant::now();
        if now >= self.blink_at {
            self.view.cursor_on = !self.view.cursor_on;
            self.blink_at = now + BLINK;
            if self.view.focused && self.emu.term.cursor_visible {
                self.redraw();
            }
        }
        el.set_control_flow(ControlFlow::WaitUntil(self.blink_at));
    }
}

fn main() {
    let event_loop = match EventLoop::<UserEvent>::with_user_event().build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("couldn't start the window system: {e}");
            std::process::exit(1);
        }
    };
    let mut app = App {
        proxy: event_loop.create_proxy(),
        args: std::env::args().skip(1).collect(),
        window: None,
        surface: None,
        painter: None,
        emu: Emulator::new(100, 30),
        session: None,
        mods: ModifiersState::empty(),
        view: View::default(),
        font_pt: FONT_PT,
        blink_at: Instant::now() + BLINK,
        mouse: (0.0, 0.0),
        selecting: None,
        clipboard: None,
        ended: false,
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("{e}");
    }
    if let Some(s) = app.session.as_mut() {
        let _ = s.killer.kill();
    }
}
