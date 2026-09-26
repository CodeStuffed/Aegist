//! What the agent has learned about each app, kept on disk
//! (data/memory/apps/<app>.json): which button does what, how often using
//! it worked, what it tried for each goal and how that went. It learns
//! from its own past decisions: every success or failure, checked on the
//! screen or told by you, changes which button it trusts next time.

use super::vision::{color_name, Print, Rect, Rgb, Shape};
use crate::rng::Rng;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// What using a button does, as seen on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    /// A drawing tool: dragging on the canvas draws this shape.
    Draws { shape: Shape },
    /// Clicking the canvas fills an area.
    Fills,
    /// Dragging over a drawing removes it.
    Erases,
    /// Makes the next drawing this color.
    SetsColor { color: Rgb },
    /// Wipes the whole canvas at once.
    Clears,
    /// Opens a menu or dialog.
    OpensWindow,
    /// Nothing visible happened.
    Nothing,
}

impl Effect {
    pub fn describe(&self) -> String {
        match self {
            Effect::Draws { shape } => format!("draws {}", shape.name()),
            Effect::Fills => "fills areas with color".into(),
            Effect::Erases => "erases".into(),
            Effect::SetsColor { color } => format!("picks {}", color_name(*color)),
            Effect::Clears => "clears the whole canvas".into(),
            Effect::OpensWindow => "opens a menu or dialog".into(),
            Effect::Nothing => "did nothing I could see".into(),
        }
    }

    /// Using it could throw work away or leave the app somewhere unexpected.
    pub fn risky(&self) -> bool {
        matches!(self, Effect::Clears | Effect::OpensWindow)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Known {
    pub id: usize,
    /// Relative to the app's window.
    pub rect: Rect,
    pub print: Print,
    pub effect: Effect,
    /// Times it was used for a goal, and how that went.
    pub successes: u32,
    pub failures: u32,
}

impl Known {
    /// How likely using it for its effect works, from its record (a
    /// Beta(1+s, 1+f) belief: 50% with no record, then following it).
    pub fn belief(&self) -> f32 {
        (1.0 + self.successes as f32) / (2.0 + (self.successes + self.failures) as f32)
    }

    /// A random draw from that belief (Thompson sampling): practice tries
    /// buttons it's less sure of, in proportion to their chance of being best.
    pub fn sample(&self, rng: &mut Rng) -> f32 {
        let a = gamma(1.0 + self.successes as f32, rng);
        let b = gamma(1.0 + self.failures as f32, rng);
        a / (a + b)
    }
}

/// Gamma(shape, 1) (Marsaglia & Tsang).
fn gamma(shape: f32, rng: &mut Rng) -> f32 {
    if shape < 1.0 {
        return gamma(shape + 1.0, rng) * rng.uniform().max(1e-9).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.uniform().max(1e-9);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// One attempt at a goal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub goal: String,
    /// Buttons used, by id.
    pub used: Vec<usize>,
    /// How likely it thought success was, before trying.
    pub expected: f32,
    /// What the screen showed.
    pub worked: bool,
    /// What you said about it, if anything (overrides the screen).
    pub feedback: Option<bool>,
    pub at: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AppMemory {
    pub app: String,
    /// The drawing area, relative to the window.
    pub canvas: Option<Rect>,
    /// Its color when blank.
    #[serde(default)]
    pub canvas_color: Option<Rgb>,
    pub elements: Vec<Known>,
    pub undo_works: Option<bool>,
    pub attempts: Vec<Attempt>,
    pub notes: Vec<String>,
}

/// "Untitled - Paint" -> "paint"
pub fn app_key(title: &str) -> String {
    let name = title.rsplit(" - ").next().unwrap_or(title);
    let key: String = name.to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect();
    let key = key.trim_matches('-').to_string();
    if key.is_empty() { "app".into() } else { key }
}

pub fn dir(settings: &crate::config::Settings) -> PathBuf {
    settings.data_path("memory/apps")
}

impl AppMemory {
    pub fn path(settings: &crate::config::Settings, app: &str) -> PathBuf {
        dir(settings).join(format!("{}.json", app_key(app)))
    }

    pub fn load(settings: &crate::config::Settings, app: &str) -> AppMemory {
        std::fs::read_to_string(Self::path(settings, app))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_else(|| AppMemory { app: app_key(app), ..Default::default() })
    }

    pub fn save(&self, settings: &crate::config::Settings) -> Result<()> {
        crate::util::write_atomic(&Self::path(settings, &self.app), serde_json::to_string_pretty(self)?.as_bytes())
    }

    /// Every app it has memories of.
    pub fn all(settings: &crate::config::Settings) -> Vec<AppMemory> {
        let mut v: Vec<AppMemory> = std::fs::read_dir(dir(settings))
            .map(|rd| rd.flatten().filter_map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()).collect())
            .unwrap_or_default();
        v.sort_by(|a, b| a.app.cmp(&b.app));
        v
    }

    /// Record what a button does (replacing what was known about the button
    /// in the same place, or one that looks the same).
    pub fn learn(&mut self, rect: Rect, print: Print, effect: Effect) -> usize {
        if let Some(k) = self.elements.iter_mut().find(|k| k.rect == rect || (k.print == print && k.rect.intersect(&rect).area() > 0)) {
            if k.effect != effect {
                // it does something else than thought: its record starts over,
                // with the one time it was just seen doing this
                k.successes = (effect != Effect::Nothing) as u32;
                k.failures = 0;
            }
            k.effect = effect;
            k.rect = rect;
            k.print = print;
            return k.id;
        }
        let id = self.elements.iter().map(|k| k.id + 1).max().unwrap_or(0);
        // seeing it work while learning counts as its first success
        self.elements.push(Known { id, rect, print, effect, successes: (effect != Effect::Nothing) as u32, failures: 0 });
        id
    }

    pub fn get(&self, id: usize) -> Option<&Known> {
        self.elements.iter().find(|k| k.id == id)
    }

    pub fn with_effect(&self, f: impl Fn(&Effect) -> bool) -> Vec<&Known> {
        self.elements.iter().filter(|k| f(&k.effect)).collect()
    }

    /// Record how using these buttons went.
    pub fn record(&mut self, used: &[usize], worked: bool) {
        for k in self.elements.iter_mut().filter(|k| used.contains(&k.id)) {
            if worked {
                k.successes += 1;
            } else {
                k.failures += 1;
            }
        }
    }

    /// You said how the last attempt really went: correct its record.
    pub fn feedback(&mut self, good: bool) -> Option<Attempt> {
        let a = self.attempts.last_mut()?;
        let seen = a.feedback.unwrap_or(a.worked);
        a.feedback = Some(good);
        let (used, changed) = (a.used.clone(), seen != good);
        let attempt = a.clone();
        if changed {
            for k in self.elements.iter_mut().filter(|k| used.contains(&k.id)) {
                if good {
                    k.failures = k.failures.saturating_sub(1);
                    k.successes += 1;
                } else {
                    k.successes = k.successes.saturating_sub(1);
                    k.failures += 1;
                }
            }
        }
        Some(attempt)
    }

    /// How well its expectations matched what happened (Brier score:
    /// 0 is perfect, 0.25 is no better than a coin), over the last `n`.
    pub fn calibration(&self, n: usize) -> Option<f32> {
        let recent = &self.attempts[self.attempts.len().saturating_sub(n)..];
        if recent.is_empty() {
            return None;
        }
        let outcome = |a: &Attempt| a.feedback.unwrap_or(a.worked) as u8 as f32;
        Some(recent.iter().map(|a| (a.expected - outcome(a)).powi(2)).sum::<f32>() / recent.len() as f32)
    }

    pub fn success_rate(&self, n: usize) -> Option<f32> {
        let recent = &self.attempts[self.attempts.len().saturating_sub(n)..];
        (!recent.is_empty()).then(|| recent.iter().filter(|a| a.feedback.unwrap_or(a.worked)).count() as f32 / recent.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beliefs_follow_the_record_and_feedback_corrects_it() {
        let mut m = AppMemory { app: app_key("Untitled - Paint"), ..Default::default() };
        assert_eq!(m.app, "paint");
        let p = Print { shape: 1, color: [0; 3], core: [0; 3], solid: false };
        let pen = m.learn(Rect::new(0, 0, 10, 10), p, Effect::Draws { shape: Shape::Freehand });
        let red = m.learn(Rect::new(20, 0, 10, 10), Print { shape: 0, color: [230, 30, 30], core: [230, 30, 30], solid: true }, Effect::SetsColor { color: [230, 30, 30] });
        assert_eq!(m.learn(Rect::new(0, 0, 10, 10), p, Effect::Draws { shape: Shape::Freehand }), pen);
        assert!((m.get(pen).unwrap().belief() - 2.0 / 3.0).abs() < 1e-6);
        m.record(&[pen, red], true);
        m.record(&[pen], true);
        assert!(m.get(pen).unwrap().belief() > 0.7);
        m.attempts.push(Attempt { goal: "draw".into(), used: vec![pen], expected: 0.75, worked: true, feedback: None, at: String::new() });
        m.feedback(false).unwrap();
        assert_eq!((m.get(pen).unwrap().successes, m.get(pen).unwrap().failures), (2, 1));
        m.feedback(false).unwrap(); // saying it twice changes nothing more
        assert_eq!(m.get(pen).unwrap().failures, 1);
        assert!((m.calibration(10).unwrap() - 0.5625).abs() < 1e-4);
        // a button that turns out to do something else starts over
        m.learn(Rect::new(0, 0, 10, 10), p, Effect::Erases);
        assert!((m.get(pen).unwrap().belief() - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(m.get(pen).unwrap().failures, 0);
        let mut rng = Rng::new(3);
        let k = Known { id: 0, rect: Rect::new(0, 0, 1, 1), print: p, effect: Effect::Fills, successes: 30, failures: 2 };
        let mean = (0..400).map(|_| k.sample(&mut rng)).sum::<f32>() / 400.0;
        assert!((mean - k.belief()).abs() < 0.03, "{mean}");
    }
}
