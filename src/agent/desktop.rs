//! Eyes and hands on a real screen: take screenshots, move and click the
//! mouse, press keys. Talks to the operating system directly (X11 on Linux,
//! the Win32 API on Windows), no helper programs needed.

use super::vision::{Image, Rect};
use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    /// The system's handle for it.
    pub id: u64,
    pub title: String,
    pub rect: Rect,
}

pub trait Desktop {
    fn name(&self) -> String;
    fn capture(&mut self) -> Result<Image>;
    fn move_to(&mut self, x: i32, y: i32) -> Result<()>;
    /// Left button down or up.
    fn button(&mut self, down: bool) -> Result<()>;
    /// A key or combination: "ctrl+z", "escape", "enter", "a".
    fn key(&mut self, combo: &str) -> Result<()>;
    fn type_text(&mut self, text: &str) -> Result<()>;
    /// Where the pointer really is (to notice you moving it).
    fn cursor(&mut self) -> Option<(i32, i32)>;
    /// The window you're working in.
    fn focused(&mut self) -> Option<Window>;
    /// Every visible window with a title.
    fn windows(&mut self) -> Vec<Window>;
    /// Bring a window to the front and give it the keyboard.
    fn activate(&mut self, w: &Window) -> Result<()>;
    fn wait(&mut self, ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}

/// The desktop of this computer, if Aegist can reach it.
pub fn open() -> Result<Box<dyn Desktop>> {
    #[cfg(windows)]
    {
        return Ok(Box::new(win::WinDesktop::new()?));
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if std::env::var_os("DISPLAY").is_some() {
            return Ok(Box::new(x11::X11Desktop::open()?));
        }
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            bail!("Wayland doesn't let programs see or control the screen. Log in to an X11 session (\"Ubuntu on Xorg\"), or run under XWayland with DISPLAY set.");
        }
        bail!("there's no screen here (DISPLAY isn't set). Practice in the built-in paint app with --sim.");
    }
    #[allow(unreachable_code)]
    {
        bail!("using the screen isn't supported on this system yet; practice in the built-in paint app with --sim")
    }
}

/// Split "ctrl+shift+z" into modifiers and the key.
pub fn split_combo(combo: &str) -> (Vec<String>, String) {
    let parts: Vec<String> = combo.split('+').map(|p| p.trim().to_lowercase()).filter(|p| !p.is_empty()).collect();
    match parts.split_last() {
        Some((k, mods)) => (mods.to_vec(), k.clone()),
        None => (Vec::new(), String::new()),
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
#[allow(clippy::unnecessary_cast)] // c_ulong is 32 bits on some systems
pub mod x11 {
    //! X11 through libX11 and the XTest extension, loaded at run time.
    use super::*;
    use libloading::Library;
    use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CStr, CString};

    type Display = c_void;
    type XWindow = c_ulong;

    #[repr(C)]
    struct XImage {
        width: c_int,
        height: c_int,
        xoffset: c_int,
        format: c_int,
        data: *mut c_char,
        byte_order: c_int,
        bitmap_unit: c_int,
        bitmap_bit_order: c_int,
        bitmap_pad: c_int,
        depth: c_int,
        bytes_per_line: c_int,
        bits_per_pixel: c_int,
        red_mask: c_ulong,
        green_mask: c_ulong,
        blue_mask: c_ulong,
        obdata: *mut c_char,
        create_image: *const c_void,
        destroy_image: Option<unsafe extern "C" fn(*mut XImage) -> c_int>,
    }

    struct Api {
        _x: Library,
        _xtst: Library,
        open_display: unsafe extern "C" fn(*const c_char) -> *mut Display,
        default_root: unsafe extern "C" fn(*mut Display) -> XWindow,
        get_geometry: unsafe extern "C" fn(*mut Display, XWindow, *mut XWindow, *mut c_int, *mut c_int, *mut c_uint, *mut c_uint,
                                           *mut c_uint, *mut c_uint) -> c_int,
        get_image: unsafe extern "C" fn(*mut Display, XWindow, c_int, c_int, c_uint, c_uint, c_ulong, c_int) -> *mut XImage,
        query_pointer: unsafe extern "C" fn(*mut Display, XWindow, *mut XWindow, *mut XWindow, *mut c_int, *mut c_int, *mut c_int,
                                            *mut c_int, *mut c_uint) -> c_int,
        get_input_focus: unsafe extern "C" fn(*mut Display, *mut XWindow, *mut c_int) -> c_int,
        query_tree: unsafe extern "C" fn(*mut Display, XWindow, *mut XWindow, *mut XWindow, *mut *mut XWindow, *mut c_uint) -> c_int,
        fetch_name: unsafe extern "C" fn(*mut Display, XWindow, *mut *mut c_char) -> c_int,
        translate: unsafe extern "C" fn(*mut Display, XWindow, XWindow, c_int, c_int, *mut c_int, *mut c_int, *mut XWindow) -> c_int,
        free: unsafe extern "C" fn(*mut c_void) -> c_int,
        sync: unsafe extern "C" fn(*mut Display, c_int) -> c_int,
        string_to_keysym: unsafe extern "C" fn(*const c_char) -> c_ulong,
        keysym_to_keycode: unsafe extern "C" fn(*mut Display, c_ulong) -> u8,
        raise: unsafe extern "C" fn(*mut Display, XWindow) -> c_int,
        intern_atom: unsafe extern "C" fn(*mut Display, *const c_char, c_int) -> c_ulong,
        get_property: unsafe extern "C" fn(*mut Display, XWindow, c_ulong, c_long, c_long, c_int, c_ulong, *mut c_ulong, *mut c_int,
                                           *mut c_ulong, *mut c_ulong, *mut *mut u8) -> c_int,
        set_input_focus: unsafe extern "C" fn(*mut Display, XWindow, c_int, c_ulong) -> c_int,
        fake_motion: unsafe extern "C" fn(*mut Display, c_int, c_int, c_int, c_ulong) -> c_int,
        fake_button: unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int,
        fake_key: unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int,
    }

    pub struct X11Desktop {
        api: Api,
        dpy: *mut Display,
        root: XWindow,
    }

    fn load(names: &[&str]) -> Result<Library> {
        for n in names {
            // SAFETY: loading a system library by name; its initializers are the platform's own.
            if let Ok(l) = unsafe { Library::new(n) } {
                return Ok(l);
            }
        }
        bail!("{} isn't installed", names[0])
    }

    impl X11Desktop {
        pub fn open() -> Result<X11Desktop> {
            let x = load(&["libX11.so.6", "libX11.so"])?;
            let xtst = load(&["libXtst.so.6", "libXtst.so"]).map_err(|_| {
                anyhow::anyhow!("the XTest library (libXtst) isn't installed; it's how programs press keys and click. Install libxtst6.")
            })?;
            // SAFETY: the symbol types match Xlib's and XTest's documented C signatures.
            let api = unsafe {
                Api {
                    open_display: *x.get(b"XOpenDisplay\0")?,
                    default_root: *x.get(b"XDefaultRootWindow\0")?,
                    get_geometry: *x.get(b"XGetGeometry\0")?,
                    get_image: *x.get(b"XGetImage\0")?,
                    query_pointer: *x.get(b"XQueryPointer\0")?,
                    get_input_focus: *x.get(b"XGetInputFocus\0")?,
                    query_tree: *x.get(b"XQueryTree\0")?,
                    fetch_name: *x.get(b"XFetchName\0")?,
                    translate: *x.get(b"XTranslateCoordinates\0")?,
                    free: *x.get(b"XFree\0")?,
                    sync: *x.get(b"XSync\0")?,
                    string_to_keysym: *x.get(b"XStringToKeysym\0")?,
                    keysym_to_keycode: *x.get(b"XKeysymToKeycode\0")?,
                    raise: *x.get(b"XRaiseWindow\0")?,
                    intern_atom: *x.get(b"XInternAtom\0")?,
                    get_property: *x.get(b"XGetWindowProperty\0")?,
                    set_input_focus: *x.get(b"XSetInputFocus\0")?,
                    fake_motion: *xtst.get(b"XTestFakeMotionEvent\0")?,
                    fake_button: *xtst.get(b"XTestFakeButtonEvent\0")?,
                    fake_key: *xtst.get(b"XTestFakeKeyEvent\0")?,
                    _x: x,
                    _xtst: xtst,
                }
            };
            // SAFETY: null means "the display in $DISPLAY".
            let dpy = unsafe { (api.open_display)(std::ptr::null()) };
            if dpy.is_null() {
                bail!("couldn't connect to the X display {:?}", std::env::var("DISPLAY").unwrap_or_default());
            }
            // SAFETY: dpy is a live connection.
            let root = unsafe { (api.default_root)(dpy) };
            Ok(X11Desktop { api, dpy, root })
        }

        fn geometry(&self, w: XWindow) -> Option<Rect> {
            let (mut r, mut x, mut y, mut ww, mut hh, mut b, mut d) = (0, 0, 0, 0, 0, 0, 0);
            // SAFETY: out-pointers to locals; w is a window id from the server.
            let ok = unsafe { (self.api.get_geometry)(self.dpy, w, &mut r, &mut x, &mut y, &mut ww, &mut hh, &mut b, &mut d) };
            if ok == 0 {
                return None;
            }
            let (mut ax, mut ay, mut child) = (0, 0, 0);
            // SAFETY: as above.
            unsafe { (self.api.translate)(self.dpy, w, self.root, 0, 0, &mut ax, &mut ay, &mut child) };
            Some(Rect::new(ax, ay, ww as i32, hh as i32))
        }

        /// The window's title: _NET_WM_NAME (UTF-8, what modern apps set), else WM_NAME.
        fn name_of(&self, w: XWindow) -> Option<String> {
            // SAFETY: atoms by name; the property buffer is copied, then freed with XFree.
            unsafe {
                let net = (self.api.intern_atom)(self.dpy, c"_NET_WM_NAME".as_ptr(), 0);
                let utf8 = (self.api.intern_atom)(self.dpy, c"UTF8_STRING".as_ptr(), 0);
                let (mut ty, mut fmt, mut n, mut after, mut prop) = (0, 0, 0, 0, std::ptr::null_mut());
                if (self.api.get_property)(self.dpy, w, net, 0, 1024, 0, utf8, &mut ty, &mut fmt, &mut n, &mut after, &mut prop) == 0
                    && !prop.is_null()
                {
                    let s = String::from_utf8_lossy(std::slice::from_raw_parts(prop, n as usize)).into_owned();
                    (self.api.free)(prop as *mut c_void);
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
            }
            let mut p: *mut c_char = std::ptr::null_mut();
            // SAFETY: XFetchName allocates the name, freed with XFree below.
            unsafe {
                if (self.api.fetch_name)(self.dpy, w, &mut p) != 0 && !p.is_null() {
                    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
                    (self.api.free)(p as *mut c_void);
                    return Some(s);
                }
            }
            None
        }

        /// (parent, children)
        fn tree(&self, w: XWindow) -> Option<(XWindow, Vec<XWindow>)> {
            let (mut root, mut parent, mut kids, mut n) = (0, 0, std::ptr::null_mut(), 0);
            // SAFETY: out-pointers; the children list is copied, then freed with XFree.
            unsafe {
                if (self.api.query_tree)(self.dpy, w, &mut root, &mut parent, &mut kids, &mut n) == 0 {
                    return None;
                }
                let children = if kids.is_null() { Vec::new() } else { std::slice::from_raw_parts(kids, n as usize).to_vec() };
                if !kids.is_null() {
                    (self.api.free)(kids as *mut c_void);
                }
                Some((parent, children))
            }
        }

        fn parent(&self, w: XWindow) -> Option<XWindow> {
            let (parent, _) = self.tree(w)?;
            (parent != 0 && parent != self.root).then_some(parent)
        }

        fn keycode(&self, name: &str) -> Option<u8> {
            let name = match name {
                "ctrl" | "control" => "Control_L",
                "shift" => "Shift_L",
                "alt" => "Alt_L",
                "super" | "win" | "meta" => "Super_L",
                "escape" | "esc" => "Escape",
                "enter" | "return" => "Return",
                "tab" => "Tab",
                "space" | " " => "space",
                "backspace" => "BackSpace",
                "delete" | "del" => "Delete",
                "up" => "Up",
                "down" => "Down",
                "left" => "Left",
                "right" => "Right",
                "home" => "Home",
                "end" => "End",
                other => other,
            };
            let c = CString::new(name).ok()?;
            // SAFETY: c is a valid C string; the display is live.
            let code = unsafe { (self.api.keysym_to_keycode)(self.dpy, (self.api.string_to_keysym)(c.as_ptr())) };
            (code != 0).then_some(code)
        }

        fn press(&mut self, code: u8, down: bool) {
            // SAFETY: XTest on a live display.
            unsafe {
                (self.api.fake_key)(self.dpy, code as c_uint, down as c_int, 0);
                (self.api.sync)(self.dpy, 0);
            }
        }
    }

    impl Desktop for X11Desktop {
        fn name(&self) -> String {
            format!("X11 display {}", std::env::var("DISPLAY").unwrap_or_default())
        }

        fn capture(&mut self) -> Result<Image> {
            let r = self.geometry(self.root).ok_or_else(|| anyhow::anyhow!("couldn't read the screen's size"))?;
            // SAFETY: ZPixmap (2) image of the whole root window; freed with its own destroy function.
            unsafe {
                let img = (self.api.get_image)(self.dpy, self.root, 0, 0, r.w as c_uint, r.h as c_uint, !0, 2);
                if img.is_null() {
                    bail!("couldn't take a screenshot");
                }
                let im = &*img;
                if im.bits_per_pixel != 32 {
                    let bpp = im.bits_per_pixel;
                    if let Some(d) = im.destroy_image {
                        d(img);
                    }
                    bail!("the screen uses {bpp} bits per pixel; only 32 is supported");
                }
                let (w, h, stride) = (im.width as usize, im.height as usize, im.bytes_per_line as usize);
                let data = std::slice::from_raw_parts(im.data as *const u8, stride * h);
                let shift = |m: c_ulong| m.trailing_zeros();
                let (rs, gs, bs) = (shift(im.red_mask), shift(im.green_mask), shift(im.blue_mask));
                let mut px = Vec::with_capacity(w * h * 3);
                for y in 0..h {
                    for x in 0..w {
                        let o = y * stride + x * 4;
                        let v = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as c_ulong;
                        px.extend_from_slice(&[(v >> rs) as u8, (v >> gs) as u8, (v >> bs) as u8]);
                    }
                }
                if let Some(d) = im.destroy_image {
                    d(img);
                }
                Ok(Image { w, h, px })
            }
        }

        fn move_to(&mut self, x: i32, y: i32) -> Result<()> {
            // SAFETY: XTest on a live display.
            unsafe {
                (self.api.fake_motion)(self.dpy, -1, x, y, 0);
                (self.api.sync)(self.dpy, 0);
            }
            Ok(())
        }

        fn button(&mut self, down: bool) -> Result<()> {
            // SAFETY: XTest on a live display.
            unsafe {
                (self.api.fake_button)(self.dpy, 1, down as c_int, 0);
                (self.api.sync)(self.dpy, 0);
            }
            Ok(())
        }

        fn key(&mut self, combo: &str) -> Result<()> {
            let (mods, key) = split_combo(combo);
            let mut codes = Vec::new();
            for m in mods.iter().chain(std::iter::once(&key)) {
                codes.push(self.keycode(m).ok_or_else(|| anyhow::anyhow!("unknown key {m:?}"))?);
            }
            for &c in &codes {
                self.press(c, true);
            }
            for &c in codes.iter().rev() {
                self.press(c, false);
            }
            Ok(())
        }

        fn type_text(&mut self, text: &str) -> Result<()> {
            let shift = self.keycode("shift");
            for ch in text.chars() {
                let (name, shifted) = match ch {
                    '\n' => ("Return".to_string(), false),
                    c if c.is_ascii_uppercase() => (c.to_ascii_lowercase().to_string(), true),
                    c if "~!@#$%^&*()_+{}|:\"<>?".contains(c) => {
                        let base = "`1234567890-=[]\\;',./";
                        let i = "~!@#$%^&*()_+{}|:\"<>?".find(c).unwrap_or(0);
                        (base.chars().nth(i).unwrap_or(' ').to_string(), true)
                    }
                    c => (c.to_string(), false),
                };
                let name = match name.as_str() {
                    " " => "space".into(), "`" => "grave".into(), "-" => "minus".into(), "=" => "equal".into(),
                    "[" => "bracketleft".into(), "]" => "bracketright".into(), "\\" => "backslash".into(),
                    ";" => "semicolon".into(), "'" => "apostrophe".into(), "," => "comma".into(), "." => "period".into(),
                    "/" => "slash".into(), _ => name,
                };
                let Some(code) = self.keycode(&name) else { continue };
                if shifted {
                    if let Some(s) = shift {
                        self.press(s, true);
                    }
                }
                self.press(code, true);
                self.press(code, false);
                if shifted {
                    if let Some(s) = shift {
                        self.press(s, false);
                    }
                }
            }
            Ok(())
        }

        fn cursor(&mut self) -> Option<(i32, i32)> {
            let (mut r, mut c, mut x, mut y, mut wx, mut wy, mut m) = (0, 0, 0, 0, 0, 0, 0);
            // SAFETY: out-pointers to locals.
            let ok = unsafe { (self.api.query_pointer)(self.dpy, self.root, &mut r, &mut c, &mut x, &mut y, &mut wx, &mut wy, &mut m) };
            (ok != 0).then_some((x, y))
        }

        fn focused(&mut self) -> Option<Window> {
            let (mut w, mut revert) = (0, 0);
            // SAFETY: out-pointers to locals.
            unsafe { (self.api.get_input_focus)(self.dpy, &mut w, &mut revert) };
            // PointerRoot (1) / None (0): no window has the focus
            let mut w = if w <= 1 { return None } else { w };
            for _ in 0..8 {
                if let Some(title) = self.name_of(w) {
                    // the top-level window (the frame's child) holds the title
                    let rect = self.geometry(w)?;
                    return Some(Window { id: w as u64, title, rect });
                }
                w = self.parent(w)?;
            }
            None
        }

        fn windows(&mut self) -> Vec<Window> {
            let mut out = Vec::new();
            let mut todo = vec![(self.root, 0)];
            while let Some((w, depth)) = todo.pop() {
                let Some((_, kids)) = self.tree(w) else { continue };
                for k in kids {
                    match self.name_of(k) {
                        Some(title) if !title.is_empty() => {
                            if let Some(rect) = self.geometry(k).filter(|r| r.w > 1 && r.h > 1) {
                                out.push(Window { id: k as u64, title, rect });
                            }
                        }
                        _ if depth < 2 => todo.push((k, depth + 1)),
                        _ => {}
                    }
                }
            }
            out
        }

        fn activate(&mut self, w: &Window) -> Result<()> {
            // SAFETY: a window id from this server; RevertToParent = 2.
            unsafe {
                (self.api.raise)(self.dpy, w.id as XWindow);
                (self.api.set_input_focus)(self.dpy, w.id as XWindow, 2, 0);
                (self.api.sync)(self.dpy, 0);
            }
            Ok(())
        }
    }
}

#[cfg(windows)]
pub mod win {
    //! Windows through user32 and gdi32.
    use super::*;
    use std::ffi::c_void;

    type Hwnd = *mut c_void;
    type Hdc = *mut c_void;

    #[repr(C)]
    #[derive(Default)]
    struct Point {
        x: i32,
        y: i32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct WinRect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MouseInput {
        dx: i32,
        dy: i32,
        mouse_data: u32,
        flags: u32,
        time: u32,
        extra: usize,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct KeybdInput {
        vk: u16,
        scan: u16,
        flags: u32,
        time: u32,
        extra: usize,
    }
    #[repr(C)]
    union InputUnion {
        mi: MouseInput,
        ki: KeybdInput,
    }
    #[repr(C)]
    struct Input {
        kind: u32,
        u: InputUnion,
    }
    #[repr(C)]
    struct BitmapInfoHeader {
        size: u32,
        width: i32,
        height: i32,
        planes: u16,
        bit_count: u16,
        compression: u32,
        size_image: u32,
        x_ppm: i32,
        y_ppm: i32,
        clr_used: u32,
        clr_important: u32,
    }
    #[repr(C)]
    struct BitmapInfo {
        header: BitmapInfoHeader,
        colors: [u32; 3],
    }

    #[link(name = "user32")]
    extern "system" {
        fn GetSystemMetrics(index: i32) -> i32;
        fn SetCursorPos(x: i32, y: i32) -> i32;
        fn GetCursorPos(p: *mut Point) -> i32;
        fn SendInput(n: u32, inputs: *const Input, size: i32) -> u32;
        fn GetForegroundWindow() -> Hwnd;
        fn GetWindowTextW(h: Hwnd, buf: *mut u16, max: i32) -> i32;
        fn GetWindowRect(h: Hwnd, r: *mut WinRect) -> i32;
        fn GetDC(h: Hwnd) -> Hdc;
        fn ReleaseDC(h: Hwnd, dc: Hdc) -> i32;
        fn SetProcessDPIAware() -> i32;
        fn VkKeyScanW(ch: u16) -> i16;
        fn EnumWindows(cb: extern "system" fn(Hwnd, isize) -> i32, param: isize) -> i32;
        fn IsWindowVisible(h: Hwnd) -> i32;
        fn IsIconic(h: Hwnd) -> i32;
        fn ShowWindow(h: Hwnd, cmd: i32) -> i32;
        fn SetForegroundWindow(h: Hwnd) -> i32;
    }
    #[link(name = "gdi32")]
    extern "system" {
        fn CreateCompatibleDC(dc: Hdc) -> Hdc;
        fn CreateCompatibleBitmap(dc: Hdc, w: i32, h: i32) -> *mut c_void;
        fn SelectObject(dc: Hdc, obj: *mut c_void) -> *mut c_void;
        fn BitBlt(dst: Hdc, x: i32, y: i32, w: i32, h: i32, src: Hdc, sx: i32, sy: i32, rop: u32) -> i32;
        fn GetDIBits(dc: Hdc, bmp: *mut c_void, start: u32, lines: u32, bits: *mut c_void, info: *mut BitmapInfo, usage: u32) -> i32;
        fn DeleteObject(obj: *mut c_void) -> i32;
        fn DeleteDC(dc: Hdc) -> i32;
    }

    const INPUT_MOUSE: u32 = 0;
    const INPUT_KEYBOARD: u32 = 1;
    const MOUSEEVENTF_LEFTDOWN: u32 = 0x2;
    const MOUSEEVENTF_LEFTUP: u32 = 0x4;
    const KEYEVENTF_KEYUP: u32 = 0x2;
    const KEYEVENTF_UNICODE: u32 = 0x4;
    const SRCCOPY: u32 = 0x00CC0020;
    const CAPTUREBLT: u32 = 0x40000000;

    pub struct WinDesktop {
        origin: (i32, i32),
        size: (i32, i32),
    }

    impl WinDesktop {
        pub fn new() -> Result<WinDesktop> {
            // SAFETY: plain Win32 calls without pointers.
            unsafe {
                SetProcessDPIAware();
                // the virtual screen: every monitor
                let origin = (GetSystemMetrics(76), GetSystemMetrics(77));
                let size = (GetSystemMetrics(78), GetSystemMetrics(79));
                if size.0 <= 0 || size.1 <= 0 {
                    bail!("couldn't read the screen's size");
                }
                Ok(WinDesktop { origin, size })
            }
        }

        fn send(&self, inputs: &[Input]) -> Result<()> {
            // SAFETY: a slice of properly initialized INPUT structs.
            let n = unsafe { SendInput(inputs.len() as u32, inputs.as_ptr(), std::mem::size_of::<Input>() as i32) };
            if n as usize != inputs.len() {
                bail!("Windows blocked the input (a program running as administrator is in front?)");
            }
            Ok(())
        }

        fn vk(name: &str) -> Option<u16> {
            Some(match name {
                "ctrl" | "control" => 0x11,
                "shift" => 0x10,
                "alt" => 0x12,
                "super" | "win" | "meta" => 0x5B,
                "escape" | "esc" => 0x1B,
                "enter" | "return" => 0x0D,
                "tab" => 0x09,
                "space" => 0x20,
                "backspace" => 0x08,
                "delete" | "del" => 0x2E,
                "left" => 0x25,
                "up" => 0x26,
                "right" => 0x27,
                "down" => 0x28,
                "home" => 0x24,
                "end" => 0x23,
                k if k.len() == 1 => {
                    let c = k.chars().next()?;
                    // SAFETY: plain call.
                    let v = unsafe { VkKeyScanW(c as u16) };
                    if v == -1 {
                        return None;
                    }
                    (v & 0xff) as u16
                }
                k if k.starts_with('f') => 0x6F + k[1..].parse::<u16>().ok().filter(|n| (1..=12).contains(n))?,
                _ => return None,
            })
        }

        fn describe(&self, h: Hwnd) -> Window {
            let mut buf = [0u16; 512];
            let mut r = WinRect::default();
            // SAFETY: the buffer's length is passed; out-pointer to a local.
            let n = unsafe {
                GetWindowRect(h, &mut r);
                GetWindowTextW(h, buf.as_mut_ptr(), buf.len() as i32)
            };
            Window {
                id: h as u64,
                title: String::from_utf16_lossy(&buf[..n.max(0) as usize]),
                rect: Rect::new(r.left - self.origin.0, r.top - self.origin.1, r.right - r.left, r.bottom - r.top),
            }
        }

        fn key_input(vk: u16, scan: u16, flags: u32) -> Input {
            Input { kind: INPUT_KEYBOARD, u: InputUnion { ki: KeybdInput { vk, scan, flags, time: 0, extra: 0 } } }
        }
    }

    impl Desktop for WinDesktop {
        fn name(&self) -> String {
            "Windows desktop".into()
        }

        fn capture(&mut self) -> Result<Image> {
            let (w, h) = self.size;
            // SAFETY: standard GDI screen capture; every object is released below.
            unsafe {
                let screen = GetDC(std::ptr::null_mut());
                let mem = CreateCompatibleDC(screen);
                let bmp = CreateCompatibleBitmap(screen, w, h);
                let old = SelectObject(mem, bmp);
                let ok = BitBlt(mem, 0, 0, w, h, screen, self.origin.0, self.origin.1, SRCCOPY | CAPTUREBLT);
                SelectObject(mem, old);
                let mut info = BitmapInfo {
                    header: BitmapInfoHeader { size: std::mem::size_of::<BitmapInfoHeader>() as u32, width: w, height: -h, planes: 1,
                                               bit_count: 32, compression: 0, size_image: 0, x_ppm: 0, y_ppm: 0, clr_used: 0,
                                               clr_important: 0 },
                    colors: [0; 3],
                };
                let mut buf = vec![0u8; (w * h * 4) as usize];
                let lines = GetDIBits(mem, bmp, 0, h as u32, buf.as_mut_ptr() as *mut c_void, &mut info, 0);
                DeleteObject(bmp);
                DeleteDC(mem);
                ReleaseDC(std::ptr::null_mut(), screen);
                if ok == 0 || lines == 0 {
                    bail!("couldn't take a screenshot");
                }
                let px = buf.chunks(4).flat_map(|p| [p[2], p[1], p[0]]).collect();
                Ok(Image { w: w as usize, h: h as usize, px })
            }
        }

        fn move_to(&mut self, x: i32, y: i32) -> Result<()> {
            // SAFETY: plain call.
            if unsafe { SetCursorPos(x + self.origin.0, y + self.origin.1) } == 0 {
                bail!("couldn't move the mouse");
            }
            Ok(())
        }

        fn button(&mut self, down: bool) -> Result<()> {
            let flags = if down { MOUSEEVENTF_LEFTDOWN } else { MOUSEEVENTF_LEFTUP };
            self.send(&[Input { kind: INPUT_MOUSE, u: InputUnion { mi: MouseInput { dx: 0, dy: 0, mouse_data: 0, flags, time: 0, extra: 0 } } }])
        }

        fn key(&mut self, combo: &str) -> Result<()> {
            let (mods, key) = split_combo(combo);
            let mut codes = Vec::new();
            for m in mods.iter().chain(std::iter::once(&key)) {
                codes.push(Self::vk(m).ok_or_else(|| anyhow::anyhow!("unknown key {m:?}"))?);
            }
            let mut inputs: Vec<Input> = codes.iter().map(|&c| Self::key_input(c, 0, 0)).collect();
            inputs.extend(codes.iter().rev().map(|&c| Self::key_input(c, 0, KEYEVENTF_KEYUP)));
            self.send(&inputs)
        }

        fn type_text(&mut self, text: &str) -> Result<()> {
            let mut inputs = Vec::new();
            for u in text.encode_utf16() {
                inputs.push(Self::key_input(0, u, KEYEVENTF_UNICODE));
                inputs.push(Self::key_input(0, u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
            }
            self.send(&inputs)
        }

        fn cursor(&mut self) -> Option<(i32, i32)> {
            let mut p = Point::default();
            // SAFETY: out-pointer to a local.
            (unsafe { GetCursorPos(&mut p) } != 0).then_some((p.x - self.origin.0, p.y - self.origin.1))
        }

        fn focused(&mut self) -> Option<Window> {
            // SAFETY: plain call.
            let h = unsafe { GetForegroundWindow() };
            (!h.is_null()).then(|| self.describe(h))
        }

        fn windows(&mut self) -> Vec<Window> {
            extern "system" fn each(h: Hwnd, param: isize) -> i32 {
                // SAFETY: param is the &mut Vec passed to EnumWindows below, alive for the call.
                let found = unsafe { &mut *(param as *mut Vec<Hwnd>) };
                // SAFETY: plain calls on a window handle from EnumWindows.
                if unsafe { IsWindowVisible(h) } != 0 {
                    found.push(h);
                }
                1
            }
            let mut found: Vec<Hwnd> = Vec::new();
            // SAFETY: the callback only touches `found`, which outlives the call.
            unsafe { EnumWindows(each, &mut found as *mut Vec<Hwnd> as isize) };
            found.into_iter().map(|h| self.describe(h)).filter(|w| !w.title.is_empty() && w.rect.w > 1).collect()
        }

        fn activate(&mut self, w: &Window) -> Result<()> {
            let h = w.id as Hwnd;
            // SAFETY: a window handle from EnumWindows; SW_RESTORE = 9.
            unsafe {
                if IsIconic(h) != 0 {
                    ShowWindow(h, 9);
                }
                SetForegroundWindow(h);
            }
            Ok(())
        }
    }
}
