//! Saying no, and staying safe while using your computer.
//!
//! Two layers:
//!   - goals: anything that could hurt you, someone else, your files or
//!     your accounts is refused before a single action, with the reason;
//!   - actions: every click, drag and key goes through `Hands`, which keeps
//!     them inside the app's window, refuses risky keys, stops after a set
//!     number of actions, and stops at once if you move the mouse yourself.

use super::desktop::{Desktop, Window};
use super::vision::{Image, Rect};
use anyhow::{bail, Result};

/// Why a goal is refused, if it is.
pub fn refuse_goal(goal: &str) -> Option<String> {
    let g = format!(" {} ", goal.to_lowercase().replace(|c: char| !c.is_alphanumeric() && c != '\'', " "));
    const RULES: &[(&[&str], &str)] = &[
        (&[" password", " passcode", " credential", " 2fa ", " one time code", " otp ", " login ", " log in ", " sign in "],
         "it involves passwords or signing in to accounts, which I won't handle for you"),
        (&[" bank", " payment", " credit card", " pay ", " purchase", " buy ", " checkout", " wire money", " transfer money", " crypto"],
         "it involves money or payments"),
        (&[" delete ", " erase all", " wipe ", " format ", " rm ", " uninstall", " remove all", " shred"],
         "it could destroy files or programs that can't be brought back"),
        (&[" send ", " email ", " e mail", " post ", " tweet", " message ", " dm ", " publish", " upload"],
         "it would send or publish something in your name"),
        (&[" hack", " exploit", " bypass", " crack", " keylog", " spy ", " steal", " phish", " malware", " virus", " ransom",
           " disable antivirus", " disable defender", " disable firewall"],
         "it could be used to break into or harm systems or people"),
        (&[" shutdown", " shut down", " restart ", " reboot", " registry", " regedit", " sudo ", " admin ", " administrator"],
         "it changes the system itself, outside any one app"),
        (&[" kill ", " hurt ", " weapon", " bomb", " suicide", " self harm"],
         "it could lead to someone getting hurt"),
    ];
    // "erase" in a paint app is fine: only "erase all"/"wipe" style goals are caught above
    RULES.iter().find(|(words, _)| words.iter().any(|w| g.contains(w))).map(|(_, why)| why.to_string())
}

/// How risky a key combination is.
pub fn key_allowed(combo: &str) -> Result<(), String> {
    let c = combo.to_lowercase();
    const SAFE: &[&str] = &["ctrl+z", "ctrl+y", "ctrl+shift+z", "escape", "enter", "tab", "left", "right", "up", "down"];
    const BLOCKED: &[(&str, &str)] = &[
        ("alt+f4", "closes the app"),
        ("ctrl+w", "closes a window or tab"),
        ("ctrl+q", "quits the app"),
        ("ctrl+s", "saves over a file"),
        ("ctrl+shift+s", "saves a file"),
        ("ctrl+p", "prints"),
        ("delete", "deletes"),
        ("super", "opens the system menu"),
        ("win", "opens the system menu"),
        ("ctrl+alt+delete", "is a system command"),
    ];
    if let Some((_, why)) = BLOCKED.iter().find(|(k, _)| *k == c || c.starts_with(&format!("{k}+"))) {
        return Err(format!("{combo} {why}"));
    }
    if SAFE.contains(&c.as_str()) || (c.len() == 1 && c.chars().all(|ch| ch.is_ascii_alphanumeric())) {
        return Ok(());
    }
    Err(format!("{combo} isn't on the list of keys I'll press"))
}

/// Why the agent stopped early.
#[derive(Debug)]
pub struct Stopped(pub String);

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Stopped {}

/// The only way the agent touches the desktop.
pub struct Hands<'a> {
    pub desk: &'a mut dyn Desktop,
    pub window: Window,
    /// Where actions may land (screen coordinates).
    pub allowed: Rect,
    pub max_actions: usize,
    pub actions: usize,
    pub delay_ms: u64,
    pub failsafe_px: i32,
    last_put: Option<(i32, i32)>,
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl<'a> Hands<'a> {
    pub fn new(desk: &'a mut dyn Desktop, window: Window, settings: &crate::config::AgentSettings) -> Hands<'a> {
        let allowed = window.rect;
        Hands { desk, window, allowed, max_actions: settings.max_actions, actions: 0, delay_ms: settings.action_delay_ms,
                failsafe_px: settings.failsafe_px, last_put: None, cancel: None }
    }

    fn check(&mut self) -> Result<()> {
        if let Some(c) = &self.cancel {
            if c.load(std::sync::atomic::Ordering::Relaxed) {
                bail!(Stopped("stopped: you asked me to".into()));
            }
        }
        if self.actions >= self.max_actions {
            bail!(Stopped(format!("stopped: used all {} actions this run allows (agent.max_actions)", self.max_actions)));
        }
        if let (Some(put), Some(now)) = (self.last_put, self.desk.cursor()) {
            if (put.0 - now.0).abs().max((put.1 - now.1).abs()) > self.failsafe_px {
                bail!(Stopped("stopped: you moved the mouse, so I let go".into()));
            }
        }
        self.actions += 1;
        Ok(())
    }

    fn inside(&self, x: i32, y: i32) -> Result<()> {
        if !self.allowed.contains(x, y) {
            bail!(Stopped(format!("refused: ({x}, {y}) is outside {:?}'s window", self.window.title)));
        }
        Ok(())
    }

    fn put(&mut self, x: i32, y: i32) -> Result<()> {
        self.desk.move_to(x, y)?;
        self.last_put = Some((x, y));
        Ok(())
    }

    pub fn click(&mut self, x: i32, y: i32) -> Result<()> {
        self.inside(x, y)?;
        self.check()?;
        self.put(x, y)?;
        self.desk.wait(20);
        self.desk.button(true)?;
        self.desk.wait(20);
        self.desk.button(false)?;
        self.desk.wait(self.delay_ms);
        Ok(())
    }

    /// Press at the first point, move through the rest (smoothly), release.
    pub fn drag(&mut self, path: &[(i32, i32)]) -> Result<()> {
        for &(x, y) in path {
            self.inside(x, y)?;
        }
        let Some(&first) = path.first() else { return Ok(()) };
        self.check()?;
        self.put(first.0, first.1)?;
        self.desk.wait(20);
        self.desk.button(true)?;
        let mut at = first;
        for &p in &path[1..] {
            let n = ((p.0 - at.0).abs().max((p.1 - at.1).abs()) / 8).max(1);
            for i in 1..=n {
                self.put(at.0 + (p.0 - at.0) * i / n, at.1 + (p.1 - at.1) * i / n)?;
                self.desk.wait(4);
            }
            at = p;
        }
        self.desk.wait(20);
        self.desk.button(false)?;
        self.desk.wait(self.delay_ms);
        Ok(())
    }

    pub fn key(&mut self, combo: &str) -> Result<()> {
        if let Err(why) = key_allowed(combo) {
            bail!(Stopped(format!("refused: {why}")));
        }
        self.check()?;
        self.desk.key(combo)?;
        self.desk.wait(self.delay_ms);
        Ok(())
    }

    /// Another window took over (a link opened a browser, a dialog popped
    /// up): its title. The agent stops rather than act in a window it
    /// wasn't given.
    pub fn strayed(&mut self) -> Option<String> {
        let now = self.desk.focused()?;
        (now.id != self.window.id && now.rect.intersect(&self.window.rect).area() > 0).then_some(now.title)
    }

    /// A screenshot, once the app has finished drawing: real apps take a
    /// moment after a click, so look until two shots in a row agree.
    pub fn see(&mut self) -> Result<Image> {
        let mut last = self.desk.capture()?;
        for _ in 0..30 {
            self.desk.wait(40);
            let now = self.desk.capture()?;
            if now == last {
                return Ok(now);
            }
            last = now;
        }
        Ok(last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::sim::SimPaint;

    #[test]
    fn harmful_goals_are_refused_and_harmless_ones_are_not() {
        for bad in ["type my bank password", "delete all my files", "send an email to my boss", "disable defender",
                    "buy a new game", "hack the wifi"] {
            assert!(refuse_goal(bad).is_some(), "{bad}");
        }
        for ok in ["draw a red circle", "erase the last line", "fill the canvas with blue", "learn paint"] {
            assert_eq!(refuse_goal(ok), None, "{ok}");
        }
        assert!(key_allowed("ctrl+z").is_ok() && key_allowed("alt+f4").is_err() && key_allowed("ctrl+s").is_err());
        assert!(key_allowed("ctrl+alt+t").is_err());
    }

    #[test]
    fn hands_stay_in_the_window_and_let_go_when_you_move_the_mouse() {
        let mut sim = SimPaint::new(0);
        let window = sim.focused().unwrap();
        let settings = crate::config::AgentSettings { max_actions: 3, ..Default::default() };
        let mut hands = Hands::new(&mut sim, window, &settings);
        assert!(hands.click(5, 5).unwrap_err().to_string().contains("outside"));
        hands.click(300, 300).unwrap();
        hands.desk.move_to(600, 500).unwrap(); // you grab the mouse
        assert!(hands.click(310, 300).unwrap_err().to_string().contains("moved the mouse"));
        hands.last_put = None;
        hands.key("ctrl+z").unwrap();
        hands.click(300, 300).unwrap();
        assert!(hands.click(300, 300).unwrap_err().to_string().contains("actions"));
        assert!(hands.key("alt+f4").is_err());
    }
}
