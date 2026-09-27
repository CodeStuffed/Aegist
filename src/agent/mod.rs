//! Using apps on the screen, and learning them by trying.
//!
//! Give it an app it has never seen (a paint program, say) and it works out
//! what the buttons do the way a person would: click one, try it on the
//! canvas, look at what changed, undo. What it finds goes into its memory
//! of that app. Then you give it goals ("draw a red rectangle"); it plans
//! from memory, acts, checks the screen to see whether it worked, and
//! updates how much it trusts each button. Practice runs goals it sets
//! itself, trying less-proven buttons in proportion to their chance of being
//! best, so it keeps getting better from its own past decisions.
//!
//! It says no before acting on anything harmful (`risk`), never leaves the
//! app's window, stops when you touch the mouse, and says "I don't know"
//! when its memory has nothing for a goal instead of clicking around.

pub mod desktop;
pub mod memory;
pub mod risk;
pub mod sim;
pub mod vision;

use crate::config::Settings;
use crate::rng::Rng;
use anyhow::{bail, Result};
use desktop::{Desktop, Window};
use memory::{AppMemory, Attempt, Effect, Known};
use risk::{Hands, Stopped};
use std::time::Instant;
use vision::{color_name, diff, find_canvas, find_elements, fit, parse_color, same_color, Change, Image, Print, Rect, Rgb, Shape};

/// Something to do in the app.
#[derive(Clone, Debug, PartialEq)]
pub enum Goal {
    /// Find out what every button does.
    Learn,
    Draw { shape: Shape, color: Option<Rgb> },
    Fill { color: Option<Rgb> },
    /// Rub out the last thing it drew.
    EraseLast,
    /// Wipe the canvas (throws the drawing away: needs your go-ahead).
    Clear,
    /// Set itself `n` goals and learn from how they go.
    Practice(usize),
}

impl Goal {
    pub fn describe(&self) -> String {
        let c = |c: &Option<Rgb>| c.map(|c| format!("{} ", color_name(c))).unwrap_or_default();
        match self {
            Goal::Learn => "learn what the buttons do".into(),
            Goal::Draw { shape, color } => format!("draw a {}{}", c(color), match shape {
                Shape::Freehand => "squiggle",
                Shape::Line => "line",
                Shape::Rectangle => "rectangle",
                Shape::Ellipse => "ellipse",
            }),
            Goal::Fill { color } => format!("fill the canvas {}", color.map(|c| format!("with {}", color_name(c))).unwrap_or_default()).trim().into(),
            Goal::EraseLast => "erase what I drew last".into(),
            Goal::Clear => "clear the canvas".into(),
            Goal::Practice(n) => format!("practice ({n} goals)"),
        }
    }
}

/// Understand a goal, or say plainly that it can't.
pub fn parse_goal(text: &str) -> Result<Goal, String> {
    if let Some(why) = risk::refuse_goal(text) {
        return Err(format!("No. I won't do that: {why}."));
    }
    let t = text.to_lowercase();
    let words: Vec<&str> = t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |w: &[&str]| words.iter().any(|x| w.contains(x));
    let color = words.windows(2).find_map(|p| parse_color(&format!("{} {}", p[0], p[1]))).or_else(|| words.iter().find_map(|w| parse_color(w)));
    if has(&["learn", "explore", "study", "discover"]) {
        return Ok(Goal::Learn);
    }
    if has(&["practice", "practise", "train"]) {
        let n = words.iter().find_map(|w| w.parse::<usize>().ok()).unwrap_or(20).clamp(1, 500);
        return Ok(Goal::Practice(n));
    }
    if has(&["clear", "wipe"]) {
        return Ok(Goal::Clear);
    }
    if has(&["erase", "rub", "undraw"]) {
        return Ok(Goal::EraseLast);
    }
    if has(&["fill", "paint", "color", "colour"]) && !has(&["draw", "line", "rectangle", "circle", "ellipse", "box", "square", "oval"]) {
        return Ok(Goal::Fill { color });
    }
    let shape = if has(&["rectangle", "rect", "box", "square"]) {
        Some(Shape::Rectangle)
    } else if has(&["ellipse", "circle", "oval", "ring"]) {
        Some(Shape::Ellipse)
    } else if has(&["line", "stroke", "diagonal"]) {
        Some(Shape::Line)
    } else if has(&["squiggle", "scribble", "freehand", "doodle", "sketch", "zigzag"]) {
        Some(Shape::Freehand)
    } else {
        None
    };
    match shape {
        Some(shape) => Ok(Goal::Draw { shape, color }),
        None if has(&["draw"]) => Err("I don't know how to draw that. I can draw lines, rectangles, ellipses and squiggles, \
                                        in any color the app has."
            .into()),
        None => Err("I don't know how to do that in an app yet. I can learn a drawing app, then draw lines, rectangles, \
                     ellipses and squiggles, fill, erase, and practice."
            .into()),
    }
}

/// What the agent is doing, as it happens.
#[derive(Clone, Debug)]
pub enum Note {
    Step(String),
    Learned(String),
    Warn(String),
}

/// How a goal went.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub worked: bool,
    /// False: it didn't try, because it doesn't know how (the message says why).
    pub tried: bool,
    pub message: String,
    /// How likely it thought success was, before trying.
    pub expected: f32,
    /// Time spent deciding what to do (not doing it).
    pub decide_us: f64,
}

impl Outcome {
    fn dont_know(message: String) -> Outcome {
        Outcome { worked: false, tried: false, message, expected: 0.0, decide_us: 0.0 }
    }
}

pub struct Agent<'a> {
    pub hands: Hands<'a>,
    pub memory: AppMemory,
    rng: Rng,
    note: &'a mut dyn FnMut(Note),
    /// Try less-proven buttons now and then (practice), or always the most trusted.
    explore: bool,
    /// The last thing drawn (screen coordinates).
    last_mark: Option<Rect>,
    spot: usize,
    canvas_bg: Rgb,
    /// What the screen looked like when it stopped early.
    pub last_seen: Option<Image>,
}

const TITLE_BAR: i32 = 36;

impl<'a> Agent<'a> {
    pub fn new(hands: Hands<'a>, settings: &Settings, note: &'a mut dyn FnMut(Note)) -> Agent<'a> {
        let memory = AppMemory::load(settings, &hands.window.title);
        Agent { hands, memory, rng: Rng::from_time(), note, explore: false, last_mark: None, spot: 0, canvas_bg: [255; 3], last_seen: None }
    }

    pub fn seeded(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    fn win(&self) -> Rect {
        self.hands.window.rect
    }

    fn to_screen(&self, r: Rect) -> Rect {
        r.offset(self.win().x, self.win().y)
    }

    fn to_window(&self, r: Rect) -> Rect {
        r.offset(-self.win().x, -self.win().y)
    }

    fn step(&mut self, s: String) {
        (self.note)(Note::Step(s));
    }

    fn canvas(&self) -> Option<Rect> {
        self.memory.canvas.map(|c| self.to_screen(c))
    }

    /// A spot on the canvas to draw at: the next of a grid of cells, `size`
    /// pixels across (smaller if the canvas is small).
    fn spot(&mut self, img: &Image) -> Result<Rect> {
        let Some(canvas) = self.canvas() else { bail!("I don't know where this app's canvas is") };
        let cell = (canvas.w.min(canvas.h) / 3).clamp(24, 160);
        let (cols, rows) = ((canvas.w / cell).max(1), (canvas.h / cell).max(1));
        // prefer an empty cell
        for k in 0..(cols * rows) as usize {
            let i = (self.spot + k) % (cols * rows) as usize;
            let r = Rect::new(canvas.x + (i as i32 % cols) * cell, canvas.y + (i as i32 / cols) * cell, cell, cell).inflate(-cell / 6);
            let blank = (0..8).all(|a| (0..8).all(|b| img.get(r.x + r.w * a / 8, r.y + r.h * b / 8) == self.canvas_bg));
            if blank {
                self.spot = i + 1;
                return Ok(r);
            }
        }
        self.spot += 1;
        let i = self.spot % (cols * rows) as usize;
        Ok(Rect::new(canvas.x + (i as i32 % cols) * cell, canvas.y + (i as i32 / cols) * cell, cell, cell).inflate(-cell / 6))
    }

    /// Across the top, then down the right: tells tools apart (see `Fit::probe_shape`).
    fn probe_path(r: Rect) -> Vec<(i32, i32)> {
        vec![(r.x, r.y), (r.right(), r.y), (r.right(), r.bottom())]
    }

    fn stroke(&mut self, path: &[(i32, i32)]) -> Result<(Change, Image)> {
        let canvas = self.canvas().unwrap_or(self.win());
        let before = self.hands.see()?;
        self.hands.drag(path)?;
        let after = self.hands.see()?;
        Ok((diff(&before, &after, canvas), after))
    }

    /// Undo, and check the canvas really went back to `want`. Some apps
    /// also record strokes that changed nothing, so it may take a few.
    fn undo_to(&mut self, want: &Image) -> Result<bool> {
        let canvas = self.canvas().unwrap_or(self.win());
        let mut ok = false;
        for _ in 0..3 {
            self.hands.key("ctrl+z")?;
            let now = self.hands.see()?;
            if diff(want, &now, canvas).count < 16 {
                ok = true;
                break;
            }
        }
        if self.memory.undo_works.is_none() || !ok {
            self.memory.undo_works = Some(ok);
        }
        Ok(ok)
    }

    /// Every changed pixel is now the blank canvas's color.
    fn became(&self, c: &Change, now: &Image) -> bool {
        let bg = self.canvas_bg;
        c.count > 0
            && (c.region.y..c.region.bottom())
                .flat_map(|y| (c.region.x..c.region.right()).map(move |x| (x, y)))
                .filter(|&(x, y)| c.at(x, y))
                .all(|(x, y)| vision::color_dist(now.get(x, y), bg) < 16)
    }

    /// Changes in the window that aren't buttons lighting up.
    fn big_change(&self, a: &Image, b: &Image, buttons: &[Rect]) -> usize {
        let c = diff(a, b, self.win());
        let mut n = 0;
        for y in c.region.y..c.region.bottom() {
            for x in c.region.x..c.region.right() {
                if c.at(x, y) && !buttons.iter().any(|r| r.inflate(4).contains(x, y)) {
                    n += 1;
                }
            }
        }
        n
    }

    // ---------------------------------------------------------------- learn

    /// Try every button and remember what it does.
    pub fn learn(&mut self) -> Result<Vec<(usize, Effect)>> {
        let img = self.hands.see()?;
        let win = self.win();
        let Some(canvas) = find_canvas(&img, win) else {
            bail!("I can't find a drawing area in {:?}. So far I only know how to learn drawing apps.", self.hands.window.title);
        };
        self.canvas_bg = img.modal_color(canvas);
        self.memory.canvas = Some(self.to_window(canvas));
        self.memory.canvas_color = Some(self.canvas_bg);
        let buttons: Vec<Rect> = find_elements(&img, win, Some(canvas))
            .into_iter()
            .filter(|r| r.y >= win.y + TITLE_BAR && r.intersect(&canvas.inflate(4)).area() == 0)
            .take(60)
            .collect();
        (self.note)(Note::Step(format!("found a {}x{} canvas and {} things that look like buttons", canvas.w, canvas.h, buttons.len())));
        if buttons.is_empty() {
            bail!("I can't see any buttons to try in {:?}", self.hands.window.title);
        }

        // what dragging does before touching anything
        let spot = self.spot(&img)?;
        let path = Self::probe_path(spot);
        let (base, _) = self.stroke(&path)?;
        let mut mode = self.kind(&base, &path);
        let mut color = base.new_color;
        if base.count > 0 {
            self.undo_to(&img)?;
        }

        let mut found = Vec::new();
        let mut unsure = Vec::new();
        for &b in &buttons {
            let (effect, sure) = match self.probe(b, &buttons, mode, color) {
                Ok(e) => e,
                Err(e) => {
                    if e.downcast_ref::<Stopped>().is_some() {
                        self.memory.save_note(format!("stopped learning: {e}"));
                    }
                    self.last_seen = self.hands.see().ok();
                    return Err(e);
                }
            };
            match effect {
                Effect::Draws { .. } | Effect::Fills => mode = Some(effect),
                Effect::Erases | Effect::Nothing => mode = None,
                Effect::SetsColor { color: c } => color = Some(c),
                _ => {}
            }
            if !sure {
                unsure.push(b);
            }
            let img = self.hands.see()?;
            let id = self.memory.learn(self.to_window(b), Print::of(&img, b), effect);
            if sure {
                (self.note)(Note::Learned(format!("button at ({}, {}) {}", b.x - win.x, b.y - win.y, effect.describe())));
            }
            found.push((id, effect));
        }

        // Buttons that behaved just like the tool before them might be that
        // tool - or do nothing at all. Switch to a different tool and try
        // them again, then try the rest on a blank spot and on a drawing
        // (a color only shows in a drawing tool's ink; an eraser only on ink).
        for b in unsure {
            let effect = self.recheck(b)?;
            let print = Print::of(&self.hands.see()?, b);
            let id = self.memory.learn(self.to_window(b), print, effect);
            if effect != Effect::Nothing {
                (self.note)(Note::Learned(format!("button at ({}, {}) {}", b.x - win.x, b.y - win.y, effect.describe())));
            }
            if let Some(f) = found.iter_mut().find(|f| f.0 == id) {
                f.1 = effect;
            }
        }
        let quiet: Vec<Rect> = buttons
            .iter()
            .copied()
            .filter(|&b| self.memory.elements.iter().any(|k| self.to_screen(k.rect) == b && k.effect == Effect::Nothing))
            .collect();
        for b in quiet {
            let effect = self.on_drawing(b)?;
            if effect != Effect::Nothing {
                let print = Print::of(&self.hands.see()?, b);
                let id = self.memory.learn(self.to_window(b), print, effect);
                (self.note)(Note::Learned(format!("button at ({}, {}) {}", b.x - win.x, b.y - win.y, effect.describe())));
                if let Some(f) = found.iter_mut().find(|f| f.0 == id) {
                    f.1 = effect;
                }
            }
        }
        Ok(found)
    }

    /// What a drag did: drew a shape, filled, or neither.
    fn kind(&self, change: &Change, path: &[(i32, i32)]) -> Option<Effect> {
        let canvas = self.canvas()?;
        if change.count == 0 {
            None
        } else if change.fraction_of(canvas) > 0.3 {
            Some(Effect::Fills)
        } else {
            fit(change, path).probe_shape().map(|shape| Effect::Draws { shape })
        }
    }

    /// Click a button, then drag on the canvas, and see what happened.
    /// `mode` and `color` are what dragging did before the click. Returns
    /// the effect and whether it's certain.
    fn probe(&mut self, b: Rect, buttons: &[Rect], mode: Option<Effect>, color: Option<Rgb>) -> Result<(Effect, bool)> {
        let canvas = self.canvas().expect("learned before probing");
        let before = self.hands.see()?;
        let (x, y) = b.center();
        self.hands.click(x, y)?;
        if let Some(title) = self.hands.strayed() {
            (self.note)(Note::Warn(format!("that button opened another window ({title:?}); I stopped so nothing else gets touched")));
            bail!(Stopped(format!("stopped: a new window opened ({title:?})")));
        }
        let clicked = self.hands.see()?;
        let big = self.big_change(&before, &clicked, buttons);
        if big as i64 > self.win().area() / 200 {
            let on_canvas = diff(&before, &clicked, canvas);
            if big == on_canvas.count && self.became(&on_canvas, &clicked) {
                self.undo_to(&before)?;
                return Ok((Effect::Clears, true));
            }
            self.hands.key("escape")?;
            let now = self.hands.see()?;
            if self.big_change(&before, &now, buttons) as i64 > self.win().area() / 200 {
                (self.note)(Note::Warn("that button opened something Escape didn't close; I stopped so nothing else gets touched".into()));
                bail!(Stopped("stopped: something opened that I couldn't close".into()));
            }
            return Ok((Effect::OpensWindow, true));
        }
        let spot = self.spot(&clicked)?;
        let path = Self::probe_path(spot);
        let (change, _) = self.stroke(&path)?;
        if change.count == 0 {
            // an eraser, or a color while an eraser is selected: find out later
            return Ok((Effect::Nothing, false));
        }
        self.undo_to(&clicked)?;
        let new = change.new_color.unwrap_or([0; 3]);
        let print = Print::of(&clicked, b);
        let recolored = color.is_some_and(|c| !same_color(c, new));
        Ok(match self.kind(&change, &path) {
            None => (Effect::Nothing, false),
            Some(k) if Some(k) == mode => {
                if recolored || (print.plain() && same_color(print.core, new)) {
                    (Effect::SetsColor { color: new }, true)
                } else {
                    (k, false)
                }
            }
            Some(k) => (k, true),
        })
    }

    /// A button did the same as the tool before it: switch to a tool that
    /// does something else, click the button again and see whether it
    /// still does what it did.
    fn recheck(&mut self, b: Rect) -> Result<Effect> {
        let img = self.hands.see()?;
        let print = Print::of(&img, b);
        let (x, y) = b.center();
        self.hands.click(x, y)?;
        let img = self.hands.see()?;
        let spot = self.spot(&img)?;
        let path = Self::probe_path(spot);
        let (first, _) = self.stroke(&path)?;
        let Some(own) = self.kind(&first, &path) else { return Ok(Effect::Nothing) };
        self.undo_to(&img)?;
        let other = self
            .memory
            .elements
            .iter()
            .find(|k| matches!(k.effect, Effect::Draws { .. } | Effect::Fills) && k.effect != own && self.to_screen(k.rect) != b)
            .map(|k| self.to_screen(k.rect));
        let Some(other) = other else {
            // nothing to compare with: go with what it did
            return Ok(own);
        };
        self.hands.click(other.center().0, other.center().1)?;
        self.hands.click(x, y)?;
        let img = self.hands.see()?;
        let (again, _) = self.stroke(&path)?;
        let now = self.kind(&again, &path);
        if again.count > 0 {
            self.undo_to(&img)?;
        }
        Ok(match now {
            Some(k) if k == own => own,
            _ if print.plain() => Effect::SetsColor { color: print.core },
            _ => Effect::Nothing,
        })
    }

    /// A button that showed nothing on a blank canvas: see whether it
    /// changes a drawing tool's color, then whether it removes a drawing.
    fn on_drawing(&mut self, b: Rect) -> Result<Effect> {
        let Some(pen) = self.best(|e| matches!(e, Effect::Draws { .. })).map(|k| self.to_screen(k.rect)) else {
            return Ok(Effect::Nothing);
        };
        let canvas = self.canvas().expect("learned");
        let clean = self.hands.see()?;
        self.hands.click(pen.center().0, pen.center().1)?;
        let spot = self.spot(&clean)?;
        let path = Self::probe_path(spot);
        let (mark, _) = self.stroke(&path)?;
        if mark.count == 0 {
            return Ok(Effect::Nothing);
        }
        let marked = self.hands.see()?;
        self.hands.click(b.center().0, b.center().1)?;
        let clicked = self.hands.see()?;
        let gone = diff(&marked, &clicked, canvas);
        if gone.count * 2 > mark.count && self.became(&gone, &clicked) {
            self.undo_to(&marked)?;
            self.undo_to(&clean)?;
            return Ok(Effect::Clears);
        }
        // the pen again, somewhere blank: a different color means this button picks colors
        let blank = self.spot(&clicked)?;
        let path2 = Self::probe_path(blank);
        let (ink, _) = self.stroke(&path2)?;
        if ink.count > 0 {
            let with = self.hands.see()?;
            self.undo_to(&clicked)?;
            let (a, b2) = (mark.new_color.unwrap_or([0; 3]), ink.new_color.unwrap_or([0; 3]));
            let _ = with;
            if !same_color(a, b2) {
                self.undo_to(&clean)?;
                return Ok(Effect::SetsColor { color: b2 });
            }
            if Print::of(&clicked, b).plain() && same_color(Print::of(&clicked, b).core, b2) {
                self.undo_to(&clean)?;
                return Ok(Effect::SetsColor { color: b2 });
            }
            self.undo_to(&clean)?;
            return Ok(Effect::Nothing);
        }
        // it doesn't draw on a blank spot: does it rub out ink?
        let (rub, rubbed) = self.stroke(&path)?;
        let erased = rub.count > 0 && self.became(&rub, &rubbed);
        if rub.count > 0 {
            self.undo_to(&marked)?;
        }
        self.undo_to(&clean)?;
        Ok(if erased { Effect::Erases } else { Effect::Nothing })
    }

    // ---------------------------------------------------------------- goals

    /// The most trusted button with this effect (or, when practicing, a
    /// Thompson draw among them).
    fn best(&mut self, want: impl Fn(&Effect) -> bool) -> Option<Known> {
        let options: Vec<Known> = self.memory.with_effect(&want).into_iter().cloned().collect();
        if self.explore {
            let rng = &mut self.rng;
            options.into_iter().map(|k| (k.sample(rng), k)).max_by(|a, b| a.0.total_cmp(&b.0)).map(|x| x.1)
        } else {
            options.into_iter().max_by(|a, b| a.belief().total_cmp(&b.belief()).then(b.id.cmp(&a.id)))
        }
    }

    /// The button is still where memory says (it looks the same)?
    fn locate(&mut self, k: &Known, img: &Image) -> Option<Rect> {
        let r = self.to_screen(k.rect);
        if Print::of(img, r).similar(&k.print) {
            return Some(r);
        }
        // moved? look for it among the buttons on screen now
        find_elements(img, self.win(), self.canvas()).into_iter().find(|&c| Print::of(img, c).similar(&k.print))
    }

    pub fn attempt(&mut self, goal: &Goal, risky_ok: bool) -> Result<Outcome> {
        if self.memory.canvas.is_none() || self.memory.elements.is_empty() {
            return Ok(Outcome::dont_know(format!(
                "I don't know this app yet ({}). Let me learn it first: say \"learn\".", self.memory.app)));
        }
        let img = self.hands.see()?;
        let start = Instant::now();
        if let Some(c) = self.memory.canvas_color {
            self.canvas_bg = c;
        }
        let known = |me: &AppMemory, what: &str| {
            let list: Vec<String> = me.elements.iter().filter(|k| k.effect != Effect::Nothing).map(|k| k.effect.describe()).collect();
            format!("I don't know how to {what} in {} yet. What I know: {}.", me.app, if list.is_empty() { "nothing".into() } else { list.join(", ") })
        };
        // plan: which buttons, in order, then what to do on the canvas
        let mut plan: Vec<Known> = Vec::new();
        match goal {
            Goal::Draw { shape, color } => {
                let s = *shape;
                let Some(tool) = self.best(|e| *e == Effect::Draws { shape: s }) else {
                    return Ok(Outcome::dont_know(known(&self.memory, &format!("draw {}", s.name()))));
                };
                plan.push(tool);
                if let Some(c) = color {
                    let c = *c;
                    let Some(swatch) = self.best(|e| matches!(e, Effect::SetsColor { color } if same_color(*color, c))) else {
                        return Ok(Outcome::dont_know(known(&self.memory, &format!("pick {}", color_name(c)))));
                    };
                    plan.push(swatch);
                }
            }
            Goal::Fill { color } => {
                let Some(tool) = self.best(|e| *e == Effect::Fills) else { return Ok(Outcome::dont_know(known(&self.memory, "fill"))) };
                plan.push(tool);
                if let Some(c) = color {
                    let c = *c;
                    let Some(swatch) = self.best(|e| matches!(e, Effect::SetsColor { color } if same_color(*color, c))) else {
                        return Ok(Outcome::dont_know(known(&self.memory, &format!("pick {}", color_name(c)))));
                    };
                    plan.push(swatch);
                }
            }
            Goal::EraseLast => {
                if self.last_mark.is_none() {
                    return Ok(Outcome::dont_know("I haven't drawn anything this time, so there's nothing of mine to erase.".into()));
                }
                let Some(tool) = self.best(|e| *e == Effect::Erases) else { return Ok(Outcome::dont_know(known(&self.memory, "erase"))) };
                plan.push(tool);
            }
            Goal::Clear => {
                if !risky_ok {
                    return Ok(Outcome::dont_know("No: clearing throws the whole drawing away. Say it again with --yes (or \"yes, clear it\") if you mean it.".into()));
                }
                let Some(tool) = self.best(|e| *e == Effect::Clears) else { return Ok(Outcome::dont_know(known(&self.memory, "clear the canvas"))) };
                plan.push(tool);
            }
            Goal::Learn | Goal::Practice(_) => bail!("not a single goal"),
        }
        let expected: f32 = plan.iter().map(|k| k.belief()).product();
        let decide_us = start.elapsed().as_secs_f64() * 1e6;
        self.step(format!("plan: {} ({:.0}% sure it works, decided in {:.0} µs)",
            plan.iter().map(|k| k.effect.describe()).collect::<Vec<_>>().join(", then "), expected * 100.0, decide_us));

        let mut used = Vec::new();
        for k in &plan {
            let Some(r) = self.locate(k, &img) else {
                let msg = format!("the button that {} isn't where I remember it, and I can't find it anywhere. Say \"learn\" to look again.",
                                  k.effect.describe());
                return Ok(Outcome::dont_know(msg));
            };
            if k.effect.risky() && !risky_ok {
                return Ok(Outcome::dont_know(format!("No: that button {}, which is risky.", k.effect.describe())));
            }
            self.hands.click(r.center().0, r.center().1)?;
            used.push(k.id);
        }
        let img = self.hands.see()?;
        let canvas = self.canvas().expect("known");
        let (worked, detail) = match goal {
            Goal::Draw { shape, color } => {
                let r = self.spot(&img)?;
                let path = match shape {
                    Shape::Freehand => vec![(r.x, r.y), (r.x + r.w / 3, r.bottom()), (r.x + 2 * r.w / 3, r.y), (r.right(), r.bottom())],
                    _ => vec![(r.x, r.y), (r.right(), r.bottom())],
                };
                let (change, _) = self.stroke(&path)?;
                let f = fit(&change, &path);
                let shape_ok = f.matches(*shape);
                let color_ok = color.is_none_or(|c| change.new_color.is_some_and(|n| same_color(n, c)));
                if change.count > 0 {
                    self.last_mark = change.bbox;
                }
                let seen = change.new_color.map(color_name).unwrap_or("nothing");
                (shape_ok && color_ok, match (shape_ok, color_ok) {
                    (true, true) => format!("drew it (checked on the screen: {} in {seen})", shape.name()),
                    (false, _) if change.count == 0 => "nothing appeared on the canvas".to_string(),
                    (false, _) => format!("something appeared, but it isn't {} (outline match {:.0}%)", shape.name(),
                                          100.0 * [f.path, f.chord, f.boxed, f.ellipse][*shape as usize]),
                    (true, false) => format!("drew the shape, but in {seen}"),
                })
            }
            Goal::Fill { color } => {
                let r = self.spot(&img)?;
                let before = self.hands.see()?;
                self.hands.click(r.center().0, r.center().1)?;
                let after = self.hands.see()?;
                let change = diff(&before, &after, canvas);
                let ok = change.fraction_of(canvas) > 0.1 && color.is_none_or(|c| change.new_color.is_some_and(|n| same_color(n, c)));
                (ok, if ok { format!("filled {:.0}% of the canvas", 100.0 * change.fraction_of(canvas)) } else { "the canvas didn't fill".into() })
            }
            Goal::EraseLast => {
                let m = self.last_mark.expect("checked").inflate(6).intersect(&canvas);
                let mut path = Vec::new();
                let mut y = m.y;
                while y <= m.bottom() {
                    path.push((m.x, y));
                    path.push((m.right(), y));
                    y += 6;
                }
                let (_, after) = self.stroke(&path)?;
                let left = (m.y..m.bottom()).flat_map(|y| (m.x..m.right()).map(move |x| (x, y)))
                    .filter(|&(x, y)| vision::color_dist(after.get(x, y), self.canvas_bg) >= 16).count();
                let ok = (left as f32) < m.area() as f32 * 0.02;
                if ok {
                    self.last_mark = None;
                }
                (ok, if ok { "erased it".into() } else { format!("{left} pixels of it are still there") })
            }
            Goal::Clear => {
                let after = self.hands.see()?;
                let ok = diff(&after, &Image::new(after.w, after.h, self.canvas_bg), canvas).count < 16;
                (ok, if ok { "cleared".into() } else { "the canvas isn't blank".into() })
            }
            _ => unreachable!(),
        };
        self.memory.record(&used, worked);
        self.memory.attempts.push(Attempt { goal: goal.describe(), used: used.clone(), expected, worked, feedback: None,
                                            at: crate::util::iso_now() });
        if self.memory.attempts.len() > 2000 {
            self.memory.attempts.drain(..500);
        }
        // a button that keeps failing gets looked at again: memory can be wrong
        if !worked {
            for id in used {
                let k = self.memory.get(id).cloned();
                if let Some(k) = k.filter(|k| k.failures >= 2 && k.belief() < 0.35) {
                    let r = self.to_screen(k.rect);
                    self.step(format!("the button I thought {} keeps failing; trying it again to see what it really does", k.effect.describe()));
                    let all: Vec<Rect> = self.memory.elements.iter().map(|k| self.to_screen(k.rect)).collect();
                    let (effect, _) = self.probe(r, &all, None, None)?;
                    let effect = if effect == Effect::Nothing { self.on_drawing(r)? } else { effect };
                    let print = Print::of(&self.hands.see()?, r);
                    self.memory.learn(k.rect, print, effect);
                    (self.note)(Note::Learned(format!("that button actually {}", effect.describe())));
                }
            }
        }
        Ok(Outcome { worked, tried: true, message: detail, expected, decide_us })
    }

    /// Set itself goals from what it knows, try them, and learn from how
    /// they go. Returns (success rate in the first half, in the second).
    pub fn practice(&mut self, rounds: usize) -> Result<(f32, f32)> {
        self.explore = true;
        let mut goals = Vec::new();
        for s in Shape::ALL {
            if self.memory.elements.iter().any(|k| k.effect == Effect::Draws { shape: s }) {
                goals.push(Goal::Draw { shape: s, color: None });
                for k in self.memory.with_effect(|e| matches!(e, Effect::SetsColor { .. })) {
                    if let Effect::SetsColor { color } = k.effect {
                        goals.push(Goal::Draw { shape: s, color: Some(color) });
                    }
                }
            }
        }
        if goals.is_empty() {
            self.explore = false;
            bail!("I don't know any drawing tools in {} yet, so there's nothing to practice. Say \"learn\" first.", self.memory.app);
        }
        let mut results = Vec::new();
        for i in 0..rounds {
            let goal = goals[self.rng.below(goals.len())].clone();
            let before = self.hands.see()?;
            let out = self.attempt(&goal, false)?;
            self.step(format!("practice {}/{rounds}: {} - {}", i + 1, goal.describe(), if out.worked { "worked" } else { "failed" }));
            results.push(out.worked);
            if out.tried && self.memory.undo_works != Some(false) {
                self.undo_to(&before)?;
            }
        }
        self.explore = false;
        let half = results.len() / 2;
        let rate = |r: &[bool]| if r.is_empty() { 0.0 } else { r.iter().filter(|&&w| w).count() as f32 / r.len() as f32 };
        Ok((rate(&results[..half]), rate(&results[half..])))
    }
}

impl AppMemory {
    fn save_note(&mut self, note: String) {
        self.notes.push(format!("{}: {note}", crate::util::iso_now()));
    }
}

/// Pick the app's window: the focused one, or the one whose title contains `name`.
pub fn find_window(desk: &mut dyn Desktop, name: Option<&str>) -> Result<Window> {
    match name {
        Some(n) => {
            let n = n.to_lowercase();
            let wins = desk.windows();
            let hit = wins.iter().filter(|w| w.title.to_lowercase().contains(&n)).max_by_key(|w| w.rect.area()).cloned();
            match hit {
                Some(w) => {
                    desk.activate(&w)?;
                    desk.wait(300);
                    Ok(desk.windows().into_iter().find(|x| x.id == w.id).unwrap_or(w))
                }
                None => bail!("there's no open window called {n:?}. Open the app first{}.",
                              if wins.is_empty() { String::new() } else {
                                  format!(" (open now: {})", wins.iter().take(8).map(|w| format!("{:?}", w.title)).collect::<Vec<_>>().join(", "))
                              }),
            }
        }
        None => desk.focused().ok_or_else(|| anyhow::anyhow!("no window has the focus; name the app, e.g. --app paint")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::SimPaint;

    fn settings() -> (tempfile::TempDir, Settings) {
        let tmp = tempfile::tempdir().unwrap();
        let s = crate::config::testing::settings(tmp.path());
        (tmp, s)
    }

    #[test]
    fn goals_are_understood_or_refused() {
        assert_eq!(parse_goal("draw a red circle"), Ok(Goal::Draw { shape: Shape::Ellipse, color: Some([230, 30, 30]) }));
        assert_eq!(parse_goal("Draw a line"), Ok(Goal::Draw { shape: Shape::Line, color: None }));
        assert_eq!(parse_goal("fill the canvas with light blue"), Ok(Goal::Fill { color: Some([0, 162, 232]) }));
        assert_eq!(parse_goal("learn paint"), Ok(Goal::Learn));
        assert_eq!(parse_goal("practice 30 times"), Ok(Goal::Practice(30)));
        assert!(parse_goal("type my password into the bank").unwrap_err().starts_with("No."));
        assert!(parse_goal("draw a horse").unwrap_err().contains("don't know"));
        assert!(parse_goal("book a flight").unwrap_err().contains("don't know"));
    }

    #[test]
    fn it_learns_an_unlabeled_paint_app_by_trying_its_buttons() {
        let (_tmp, s) = settings();
        for seed in [0, 7, 42] {
            let mut sim = SimPaint::new(seed);
            let items = sim.items.clone();
            let rects: Vec<Rect> = (0..items.len()).map(|i| sim.button_rect(i)).collect();
            let win = sim.focused().unwrap();
            let mut notes = Vec::new();
            let mut note = |n: Note| notes.push(n);
            let hands = Hands::new(&mut sim, win.clone(), &crate::config::AgentSettings { max_actions: 2000, ..Default::default() });
            let mut agent = Agent::new(hands, &s, &mut note).seeded(1);
            agent.learn().unwrap();
            let mem = agent.memory.clone();
            for (item, r) in items.iter().zip(&rects) {
                let rel = r.offset(-win.rect.x, -win.rect.y);
                let k = mem.elements.iter().find(|k| k.rect.intersect(&rel).area() > rel.area() / 2);
                let got = k.map(|k| k.effect);
                use sim::Item::*;
                let want = match item {
                    Pencil => Effect::Draws { shape: Shape::Freehand },
                    Line => Effect::Draws { shape: Shape::Line },
                    Rectangle => Effect::Draws { shape: Shape::Rectangle },
                    Ellipse => Effect::Draws { shape: Shape::Ellipse },
                    Fill => Effect::Fills,
                    Eraser => Effect::Erases,
                    Color(c) => Effect::SetsColor { color: *c },
                    Menu => Effect::OpensWindow,
                    Clear => Effect::Clears,
                };
                match (got, want) {
                    (Some(Effect::SetsColor { color: a }), Effect::SetsColor { color: b }) => {
                        assert!(same_color(a, b), "seed {seed}: {item:?} -> {a:?}")
                    }
                    (g, w) => assert_eq!(g, Some(w), "seed {seed}: {item:?} (order {items:?})"),
                }
            }
            assert_eq!(mem.undo_works, Some(true));
        }
    }

    #[test]
    fn it_reaches_goals_checks_them_and_learns_from_practice() {
        let (_tmp, s) = settings();
        let mut sim = SimPaint::new(5);
        let win = sim.focused().unwrap();
        let mut note = |_n: Note| {};
        let hands = Hands::new(&mut sim, win, &crate::config::AgentSettings { max_actions: 5000, ..Default::default() });
        let mut agent = Agent::new(hands, &s, &mut note).seeded(2);
        let early = agent.attempt(&Goal::Draw { shape: Shape::Line, color: None }, false).unwrap();
        assert!(!early.tried && early.message.contains("learn"));
        agent.learn().unwrap();
        for goal in ["draw a red rectangle", "draw a blue circle", "draw a green line", "draw a squiggle"] {
            let out = agent.attempt(&parse_goal(goal).unwrap(), false).unwrap();
            assert!(out.worked, "{goal}: {}", out.message);
            assert!(out.expected > 0.0 && out.decide_us < 50_000.0);
        }
        let erase = agent.attempt(&Goal::EraseLast, false).unwrap();
        assert!(erase.worked, "{}", erase.message);
        let fill = agent.attempt(&parse_goal("fill with yellow").unwrap(), false).unwrap();
        assert!(fill.worked, "{}", fill.message);
        let clear = agent.attempt(&Goal::Clear, false).unwrap();
        assert!(!clear.tried && clear.message.starts_with("No"));
        assert!(agent.attempt(&Goal::Clear, true).unwrap().worked);
        let pink = agent.attempt(&parse_goal("draw a pink line").unwrap(), false).unwrap();
        assert!(!pink.tried && pink.message.contains("don't know how to pick pink"), "{}", pink.message);
        let before = agent.memory.attempts.len();
        let (_, late) = agent.practice(12).unwrap();
        assert!(late > 0.8, "{late}");
        assert_eq!(agent.memory.attempts.len(), before + 12);
        assert!(agent.memory.calibration(12).unwrap() < 0.25);
        agent.memory.save(&s).unwrap();
        assert!(!AppMemory::load(&s, "Untitled - Practice Paint").elements.is_empty());
    }

    #[test]
    fn it_corrects_a_wrong_memory_after_repeated_failures() {
        let (_tmp, s) = settings();
        let mut sim = SimPaint::new(0);
        let win = sim.focused().unwrap();
        let mut note = |_n: Note| {};
        let hands = Hands::new(&mut sim, win, &crate::config::AgentSettings { max_actions: 5000, ..Default::default() });
        let mut agent = Agent::new(hands, &s, &mut note).seeded(3);
        agent.learn().unwrap();
        // corrupt memory: claim the rectangle button draws lines, and drop the real line tool
        let rect_id = agent.memory.elements.iter().find(|k| k.effect == Effect::Draws { shape: Shape::Rectangle }).unwrap().id;
        agent.memory.elements.retain(|k| k.effect != Effect::Draws { shape: Shape::Line });
        agent.memory.elements.iter_mut().find(|k| k.id == rect_id).unwrap().effect = Effect::Draws { shape: Shape::Line };
        let goal = Goal::Draw { shape: Shape::Line, color: None };
        for _ in 0..3 {
            assert!(!agent.attempt(&goal, false).unwrap().worked);
        }
        assert_eq!(agent.memory.get(rect_id).unwrap().effect, Effect::Draws { shape: Shape::Rectangle });
        let now = agent.attempt(&goal, false).unwrap();
        assert!(!now.tried, "{}", now.message);
    }
}
