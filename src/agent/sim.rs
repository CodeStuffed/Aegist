//! A small paint program that lives inside Aegist, with a screen, a mouse
//! and a keyboard of its own: somewhere to practice (and to test) using an
//! app without touching your real desktop. Its buttons have no labels and
//! can be shuffled, so the only way to know what they do is to try them.

use super::desktop::{Desktop, Window};
use super::vision::{Image, Rect, Rgb};
use anyhow::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Pencil,
    Line,
    Rectangle,
    Ellipse,
    Fill,
    Eraser,
    Color(Rgb),
    /// Opens a dialog over the canvas (Escape closes it).
    Menu,
    /// Wipes the canvas.
    Clear,
}

const SCREEN: (usize, usize) = (800, 600);
const WINDOW: Rect = Rect { x: 40, y: 30, w: 720, h: 540 };
const CANVAS: Rect = Rect { x: 48, y: 110, w: 704, h: 452 };
const BUTTON: i32 = 36;
const WHITE: Rgb = [255, 255, 255];

pub struct SimPaint {
    pub items: Vec<Item>,
    canvas: Image,
    undo: Vec<Image>,
    tool: Item,
    color: Rgb,
    dialog: bool,
    mouse: (i32, i32),
    down: Option<Vec<(i32, i32)>>,
    /// Every input it received, for tests.
    pub inputs: usize,
}

pub fn draw_line(img: &mut Image, a: (i32, i32), b: (i32, i32), c: Rgb, thick: i32, clip: Rect) {
    let n = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
    let (lo, hi) = (-(thick - 1) / 2, thick / 2);
    for i in 0..=n {
        let (x, y) = (a.0 + (b.0 - a.0) * i / n, a.1 + (b.1 - a.1) * i / n);
        for dy in lo..=hi {
            for dx in lo..=hi {
                if clip.contains(x + dx, y + dy) {
                    img.set(x + dx, y + dy, c);
                }
            }
        }
    }
}

impl SimPaint {
    /// The paint app, its buttons in an order shuffled by `seed` (0 = the standard order).
    pub fn new(seed: u64) -> SimPaint {
        let mut items = vec![Item::Pencil, Item::Line, Item::Rectangle, Item::Ellipse, Item::Fill, Item::Eraser,
                             Item::Color([0, 0, 0]), Item::Color([230, 30, 30]), Item::Color([34, 177, 76]),
                             Item::Color([30, 70, 230]), Item::Color([255, 230, 0]), Item::Menu, Item::Clear];
        if seed != 0 {
            let mut rng = crate::rng::Rng::new(seed);
            for i in (1..items.len()).rev() {
                items.swap(i, rng.below(i + 1));
            }
        }
        SimPaint { items, canvas: Image::new(CANVAS.w as usize, CANVAS.h as usize, WHITE), undo: Vec::new(), tool: Item::Pencil,
                   color: [0, 0, 0], dialog: false, mouse: (0, 0), down: None, inputs: 0 }
    }

    /// Where item `i`'s button is on the screen.
    pub fn button_rect(&self, i: usize) -> Rect {
        Rect::new(WINDOW.x + 8 + i as i32 * (BUTTON + 6), WINDOW.y + 40, BUTTON, BUTTON)
    }

    pub fn canvas_rect() -> Rect {
        CANVAS
    }

    fn dialog_rect() -> Rect {
        Rect::new(WINDOW.x + 160, WINDOW.y + 170, 400, 240)
    }

    fn save_undo(&mut self) {
        self.undo.push(self.canvas.clone());
        if self.undo.len() > 30 {
            self.undo.remove(0);
        }
    }

    fn apply(&mut self, path: &[(i32, i32)]) {
        let local: Vec<(i32, i32)> = path.iter().map(|&(x, y)| (x - CANVAS.x, y - CANVAS.y)).collect();
        let clip = Rect::new(0, 0, CANVAS.w, CANVAS.h);
        let (s, e) = (local[0], *local.last().unwrap());
        self.save_undo();
        let c = self.color;
        match self.tool {
            Item::Pencil => {
                for w in local.windows(2) {
                    draw_line(&mut self.canvas, w[0], w[1], c, 2, clip);
                }
                if local.len() == 1 {
                    draw_line(&mut self.canvas, s, s, c, 2, clip);
                }
            }
            Item::Eraser => {
                for w in local.windows(2) {
                    draw_line(&mut self.canvas, w[0], w[1], WHITE, 12, clip);
                }
            }
            Item::Line => draw_line(&mut self.canvas, s, e, c, 2, clip),
            Item::Rectangle => {
                let corners = [s, (e.0, s.1), e, (s.0, e.1), s];
                for w in corners.windows(2) {
                    draw_line(&mut self.canvas, w[0], w[1], c, 2, clip);
                }
            }
            Item::Ellipse => {
                let (cx, cy) = ((s.0 + e.0) as f32 / 2.0, (s.1 + e.1) as f32 / 2.0);
                let (rx, ry) = ((e.0 - s.0).abs() as f32 / 2.0, (e.1 - s.1).abs() as f32 / 2.0);
                let pt = |t: f32| ((cx + rx * t.cos()).round() as i32, (cy + ry * t.sin()).round() as i32);
                for i in 0..240 {
                    let (t0, t1) = (i as f32 / 240.0 * std::f32::consts::TAU, (i + 1) as f32 / 240.0 * std::f32::consts::TAU);
                    draw_line(&mut self.canvas, pt(t0), pt(t1), c, 2, clip);
                }
            }
            Item::Fill => {
                let target = self.canvas.get(s.0, s.1);
                if target == c || !clip.contains(s.0, s.1) {
                    return;
                }
                let mut stack = vec![s];
                while let Some((x, y)) = stack.pop() {
                    if clip.contains(x, y) && self.canvas.get(x, y) == target {
                        self.canvas.set(x, y, c);
                        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]);
                    }
                }
            }
            _ => {}
        }
    }

    fn click(&mut self, x: i32, y: i32) {
        let Some(i) = (0..self.items.len()).find(|&i| self.button_rect(i).contains(x, y)) else { return };
        match self.items[i] {
            Item::Color(c) => self.color = c,
            Item::Menu => self.dialog = true,
            Item::Clear => {
                self.save_undo();
                self.canvas = Image::new(CANVAS.w as usize, CANVAS.h as usize, WHITE);
            }
            tool => self.tool = tool,
        }
    }

    fn glyph(&self, img: &mut Image, item: Item, r: Rect) {
        let ink = [40, 40, 48];
        let (x0, y0, x1, y1) = (r.x + 9, r.y + 9, r.right() - 10, r.bottom() - 10);
        let all = img.bounds();
        match item {
            Item::Pencil => {
                draw_line(img, (x0, y1), (x1, y0), ink, 3, all);
                draw_line(img, (x0, y1), (x0 + 3, y1), [230, 150, 60], 3, all);
            }
            Item::Line => draw_line(img, (x0, y1), (x1, y0), ink, 1, all),
            Item::Rectangle => {
                for w in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)].windows(2) {
                    draw_line(img, w[0], w[1], ink, 1, all);
                }
            }
            Item::Ellipse => {
                for i in 0..64 {
                    let t = i as f32 / 64.0 * std::f32::consts::TAU;
                    let (cx, cy) = (r.x as f32 + 18.0, r.y as f32 + 18.0);
                    img.set((cx + 8.0 * t.cos()) as i32, (cy + 8.0 * t.sin()) as i32, ink);
                }
            }
            Item::Fill => {
                for k in 0..8 {
                    draw_line(img, (x0 + k, y1 - k), (x1 - k, y1 - k), ink, 1, all);
                }
                draw_line(img, (x1, y0 + 4), (x1, y1 - 6), [30, 70, 230], 2, all);
            }
            Item::Eraser => img.fill_rect(Rect::new(x0, y0 + 4, x1 - x0, y1 - y0 - 8), [240, 150, 180]),
            Item::Color(c) => img.fill_rect(r.inflate(-5), c),
            Item::Menu => {
                for k in 0..3 {
                    draw_line(img, (x0, y0 + 2 + k * 6), (x1, y0 + 2 + k * 6), ink, 2, all);
                }
            }
            Item::Clear => {
                draw_line(img, (x0, y0), (x1, y1), [200, 40, 40], 2, all);
                draw_line(img, (x0, y1), (x1, y0), [200, 40, 40], 2, all);
            }
        }
    }
}

impl Desktop for SimPaint {
    fn name(&self) -> String {
        "the practice paint app".into()
    }

    fn capture(&mut self) -> Result<Image> {
        let mut img = Image::new(SCREEN.0, SCREEN.1, [0, 110, 110]);
        img.fill_rect(WINDOW, [226, 226, 232]);
        img.fill_rect(Rect::new(WINDOW.x, WINDOW.y, WINDOW.w, 26), [40, 44, 60]);
        for (i, &item) in self.items.iter().enumerate() {
            let r = self.button_rect(i);
            let selected = item == self.tool || item == Item::Color(self.color);
            img.fill_rect(r, if selected { [70, 110, 230] } else { [120, 120, 130] });
            img.fill_rect(r.inflate(if selected { -2 } else { -1 }), [248, 248, 250]);
            self.glyph(&mut img, item, r);
        }
        for y in 0..CANVAS.h {
            let row = (y as usize * self.canvas.w) * 3;
            let dst = ((CANVAS.y + y) as usize * img.w + CANVAS.x as usize) * 3;
            img.px[dst..dst + CANVAS.w as usize * 3].copy_from_slice(&self.canvas.px[row..row + CANVAS.w as usize * 3]);
        }
        if self.dialog {
            let d = Self::dialog_rect();
            img.fill_rect(d, [90, 90, 100]);
            img.fill_rect(d.inflate(-2), [236, 236, 242]);
            img.fill_rect(Rect::new(d.x + 2, d.y + 2, d.w - 4, 24), [60, 70, 110]);
        }
        Ok(img)
    }

    fn move_to(&mut self, x: i32, y: i32) -> Result<()> {
        self.inputs += 1;
        self.mouse = (x, y);
        if let Some(p) = &mut self.down {
            p.push((x, y));
        }
        Ok(())
    }

    fn button(&mut self, down: bool) -> Result<()> {
        self.inputs += 1;
        let (x, y) = self.mouse;
        if down {
            self.down = Some(vec![(x, y)]);
            return Ok(());
        }
        let Some(path) = self.down.take() else { return Ok(()) };
        if self.dialog {
            return Ok(()); // the dialog takes every click
        }
        if CANVAS.contains(path[0].0, path[0].1) {
            self.apply(&path);
        } else if path.len() <= 2 {
            self.click(x, y);
        }
        Ok(())
    }

    fn key(&mut self, combo: &str) -> Result<()> {
        self.inputs += 1;
        match combo {
            "escape" => self.dialog = false,
            "ctrl+z" if !self.dialog => {
                if let Some(prev) = self.undo.pop() {
                    self.canvas = prev;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn type_text(&mut self, _text: &str) -> Result<()> {
        self.inputs += 1;
        Ok(())
    }

    fn cursor(&mut self) -> Option<(i32, i32)> {
        Some(self.mouse)
    }

    fn focused(&mut self) -> Option<Window> {
        Some(Window { id: 1, title: "Untitled - Practice Paint".into(), rect: WINDOW })
    }

    fn windows(&mut self) -> Vec<Window> {
        self.focused().into_iter().collect()
    }

    fn activate(&mut self, _w: &Window) -> Result<()> {
        Ok(())
    }

    fn wait(&mut self, _ms: u64) {}
}
