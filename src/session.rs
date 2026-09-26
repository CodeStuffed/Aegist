//! What a session shares between the screen (which only draws) and the
//! worker thread (which does the work): the model, the project, background
//! programs, and the messages the worker sends to be shown.

use crate::assist::Answer;
use crate::brain::Brain;
use crate::config::Settings;
use crate::ide::{Jobs, Server};
use crate::project::Project;
use crate::verify::Names;
use anyhow::{bail, Result};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub enum BrainState {
    Loading,
    Ready(Arc<Brain>),
    /// No model trained yet (the message says what to do).
    Missing(String),
    Failed(String),
}

pub struct Shared {
    pub brain: BrainState,
    pub project: Option<Project>,
    pub learned_names: Arc<HashSet<String>>,
    pub jobs: Jobs,
    pub servers: Vec<Server>,
    /// The full output of the last command run (lines shown were capped).
    pub last_output: Vec<String>,
    /// The last code Aegist wouldn't stand behind (for /show).
    pub last_refused: Option<Answer>,
    pub cwd: PathBuf,
}

/// From the worker to the screen.
pub enum Msg {
    /// Lines to add to the output, permanently.
    Print(Vec<String>),
    /// What the spinner says.
    Status(String),
    /// Start showing code as it's written.
    StreamStart { title: String, lang: Option<&'static str> },
    /// More code, with the probability of the token that wrote it.
    Stream(String, f32),
    StreamEnd,
    /// A question with one-key answers; the worker waits for the reply.
    Ask { question: String, choices: Vec<(char, String)>, reply: Sender<char> },
    /// The worker finished.
    Done,
}

/// Where a worker's output goes: the interactive screen, or plain stdout
/// (commands run from the shell).
#[derive(Clone)]
pub enum Out {
    Screen(Sender<Msg>),
    Plain { yes: bool },
}

#[derive(Clone)]
pub struct Ctx {
    pub settings: Arc<Settings>,
    pub shared: Arc<Mutex<Shared>>,
    pub out: Out,
    pub cancel: Arc<AtomicBool>,
    pub width: Arc<AtomicUsize>,
}

impl Ctx {
    pub fn width(&self) -> usize {
        self.width.load(Ordering::Relaxed).clamp(40, 160)
    }

    pub fn print(&self, lines: Vec<String>) {
        match &self.out {
            Out::Screen(tx) => {
                let _ = tx.send(Msg::Print(lines));
            }
            Out::Plain { .. } => {
                for l in lines {
                    println!("{l}");
                }
            }
        }
    }

    pub fn line(&self, s: impl Into<String>) {
        self.print(vec![s.into()]);
    }

    pub fn status(&self, s: impl Into<String>) {
        match &self.out {
            Out::Screen(tx) => {
                let _ = tx.send(Msg::Status(s.into()));
            }
            Out::Plain { .. } => {}
        }
    }

    pub fn send(&self, m: Msg) {
        if let Out::Screen(tx) = &self.out {
            let _ = tx.send(m);
        }
    }

    /// Ask a question; the first choice is the default when there's no one
    /// to ask (plain output answers yes only with --yes).
    pub fn ask(&self, question: &str, choices: &[(char, &str)]) -> char {
        match &self.out {
            Out::Screen(tx) => {
                let (rtx, rrx) = mpsc::channel();
                let _ = tx.send(Msg::Ask { question: question.to_string(), choices: choices.iter().map(|(k, l)| (*k, l.to_string())).collect(),
                                           reply: rtx });
                loop {
                    match rrx.recv_timeout(Duration::from_millis(100)) {
                        Ok(c) => return c,
                        Err(mpsc::RecvTimeoutError::Timeout) if self.cancelled() => return 'n',
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(_) => return 'n',
                    }
                }
            }
            Out::Plain { yes } => {
                let answer = if *yes { choices.first().map_or('y', |c| c.0) } else { 'n' };
                println!("{question} -> {}", if answer == 'n' { "no (pass --yes to accept)" } else { "yes (--yes)" });
                answer
            }
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// The model, once it's loaded (waiting for it if it's still loading).
    pub fn brain(&self) -> Result<Arc<Brain>> {
        let mut said = false;
        loop {
            match &self.shared.lock().expect("lock").brain {
                BrainState::Ready(b) => return Ok(b.clone()),
                BrainState::Missing(why) => bail!("{why}"),
                BrainState::Failed(why) => bail!("the model couldn't be loaded: {why}"),
                BrainState::Loading => {}
            }
            if !said {
                self.status("Loading the model…");
                said = true;
            }
            if self.cancelled() {
                bail!("cancelled");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Names Aegist can vouch for (learned code + this project).
    pub fn names(&self) -> Names {
        let s = self.shared.lock().expect("lock");
        Names { learned: s.learned_names.clone(), project: s.project.as_ref().map(|p| p.names.clone()).unwrap_or_default() }
    }

    pub fn cwd(&self) -> PathBuf {
        self.shared.lock().expect("lock").cwd.clone()
    }
}

/// Load the model in the background, into `shared`.
pub fn load_brain(settings: Arc<Settings>, shared: Arc<Mutex<Shared>>) {
    shared.lock().expect("lock").brain = BrainState::Loading;
    std::thread::spawn(move || {
        let state = match Brain::load(&settings) {
            Ok(b) => BrainState::Ready(Arc::new(b)),
            Err(e) if e.downcast_ref::<crate::brain::NoBrain>().is_some() => BrainState::Missing(e.to_string()),
            Err(e) => BrainState::Failed(e.to_string()),
        };
        shared.lock().expect("lock").brain = state;
    });
}

pub fn new_shared(settings: &Settings, cwd: PathBuf) -> Shared {
    Shared {
        brain: BrainState::Loading,
        project: None,
        learned_names: Arc::new(crate::learn::known_names(settings)),
        jobs: Jobs::default(),
        servers: Vec::new(),
        last_output: Vec::new(),
        last_refused: None,
        cwd,
    }
}
