//! Typed decisions: pick one of a fixed set of options, say whether
//! something is true, or give a score - read straight from the model's
//! probabilities, never by letting it write free text.
//!
//! This follows the idea behind "System One" decision models such as Jev:
//! a question with a closed set of answers can't be answered with something
//! invented, only with one of the options, and the answer comes with a
//! probability instead of a confident-sounding sentence. Here it is built
//! on Aegist's own from-scratch model (no other AI is involved):
//!
//!   - the question is read once, and every option continues the question's
//!     cached keys and values, in parallel (one shared "state", many answers);
//!   - each option's bias is cancelled by scoring it once with the question
//!     blanked out (contextual calibration), so the model can't win by
//!     always preferring the option it likes best;
//!   - when the best option isn't likely enough, or not clearly ahead of the
//!     next, the answer is "I don't know", never a guess;
//!   - every decision is logged, and when you say whether it was right
//!     (`good` / `bad`), `learn` refits how sure it lets itself be, so its
//!     confidence tracks how often it is actually right.

use crate::brain::Brain;
use crate::config::{DecideSettings, Settings};
use crate::corpus::FILE;
use crate::predict::Experience;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// One of the options.
    Choose,
    /// Yes or no: is the statement true?
    Truth,
    /// A number on a scale.
    Score,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Question {
    pub kind: Kind,
    pub text: String,
    /// What the question is about (the "state"): code, a situation, notes.
    #[serde(default)]
    pub context: String,
    pub options: Vec<String>,
}

impl Question {
    pub fn choose(text: &str, options: &[&str]) -> Question {
        Question { kind: Kind::Choose, text: text.into(), context: String::new(), options: options.iter().map(|o| o.to_string()).collect() }
    }

    pub fn truth(statement: &str) -> Question {
        Question { kind: Kind::Truth, text: statement.into(), context: String::new(), options: vec!["yes".into(), "no".into()] }
    }

    /// A score from `lo` to `hi` (at most 50 levels).
    pub fn score(text: &str, lo: i32, hi: i32) -> Question {
        let hi = hi.max(lo).min(lo + 49);
        Question { kind: Kind::Score, text: text.into(), context: String::new(), options: (lo..=hi).map(|v| v.to_string()).collect() }
    }

    /// What experience files this question under: its text and context.
    pub fn key(&self) -> String {
        format!("{} {}", self.text, self.context)
    }

    pub fn with_context(mut self, context: &str) -> Question {
        self.context = context.into();
        self
    }

    fn prompt(&self, blank: bool) -> String {
        let text = if blank { "N/A" } else { &self.text };
        let context = if blank || self.context.is_empty() { String::new() } else { format!("{}\n", self.context.trim_end()) };
        let ask = match self.kind {
            Kind::Choose => format!("# question: {text}\n# options: {}\n", self.options.join(" | ")),
            Kind::Truth => format!("# true or false: {text}\n# true (yes/no)?"),
            Kind::Score => format!("# {text}\n# score ({} to {}):", self.options[0], self.options[self.options.len() - 1]),
        };
        let lead = if self.kind == Kind::Choose { "# answer:" } else { "" };
        format!("{FILE}decision.txt\n{context}{ask}{lead}")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub id: u64,
    pub question: Question,
    /// Probability of each option, after calibration.
    pub probs: Vec<f32>,
    /// None = "I don't know".
    pub pick: Option<usize>,
    /// Probability of the best option (whether or not it was picked).
    pub confidence: f32,
    /// How far the best option is ahead of the runner-up.
    pub margin: f32,
    /// For scores: the probability-weighted value.
    pub expected: Option<f32>,
    /// Per-option scores before the temperature (what calibration refits):
    /// the model's part plus experience's part, below.
    pub scores: Vec<f32>,
    #[serde(default)]
    pub from_model: Vec<f32>,
    #[serde(default)]
    pub from_experience: Vec<f32>,
    pub reason: String,
    pub millis: f64,
}

impl Decision {
    pub fn answer(&self) -> &str {
        match self.pick {
            Some(i) => &self.question.options[i],
            None => "I don't know",
        }
    }

    /// For truth questions: the probability that the statement is true.
    pub fn p_true(&self) -> Option<f32> {
        (self.question.kind == Kind::Truth).then(|| self.probs[0])
    }
}

/// How sure the decider lets itself be: its scores are divided by this
/// temperature before they become probabilities. Refit from feedback.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub temperature: f32,
    /// Decisions with feedback it was fitted on.
    pub fitted_on: usize,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration { temperature: 1.0, fitted_on: 0 }
    }
}

pub fn softmax(scores: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-3);
    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = scores.iter().map(|s| ((s - m) / t).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

/// Turn per-option scores into a decision: probabilities, the pick or an
/// abstention, and why.
pub fn conclude(question: Question, scores: Vec<f32>, temperature: f32, settings: &DecideSettings) -> Decision {
    let probs = softmax(&scores, temperature);
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let best = order[0];
    let confidence = probs[best];
    let margin = confidence - order.get(1).map_or(0.0, |&i| probs[i]);
    let expected = (question.kind == Kind::Score)
        .then(|| question.options.iter().zip(&probs).map(|(o, p)| o.parse::<f32>().unwrap_or(0.0) * p).sum());
    let (pick, reason) = if confidence < settings.abstain_below {
        (None, format!("the best option ({:?}) is only {:.0}% likely", question.options[best], confidence * 100.0))
    } else if probs.len() > 1 && margin < settings.min_margin {
        let second = &question.options[order[1]];
        (None, format!("{:?} and {:?} are too close to call ({:.0}% vs {:.0}%)", question.options[best], second,
                       confidence * 100.0, (confidence - margin) * 100.0))
    } else {
        (Some(best), format!("{:.0}% likely, {:.0} points ahead of the next option", confidence * 100.0, margin * 100.0))
    };
    Decision { id: 0, question, probs, pick, confidence, margin, expected, scores, from_model: Vec::new(), from_experience: Vec::new(), reason,
               millis: 0.0 }
}

/// Decisions from the model and from experience.
pub struct Decider<'a> {
    /// The trained model (None: decide from experience alone).
    pub brain: Option<&'a Brain>,
    pub settings: DecideSettings,
    pub calibration: Calibration,
    pub experience: Experience,
    log: Option<Log>,
}

impl<'a> Decider<'a> {
    /// A decider that logs to, calibrates from and learns in Aegist's memory folder.
    pub fn new(brain: Option<&'a Brain>, settings: &Settings) -> Decider<'a> {
        let log = Log::new(settings);
        Decider { brain, settings: settings.decide.clone(), calibration: log.calibration(), experience: Experience::load(settings),
                  log: Some(log) }
    }

    /// A decider that remembers nothing (for tests and one-off checks).
    pub fn ephemeral(brain: Option<&'a Brain>, settings: &DecideSettings) -> Decider<'a> {
        Decider { brain, settings: settings.clone(), calibration: Calibration::default(), experience: Experience::default(), log: None }
    }

    /// What the model thinks of each option: mean log-probability per
    /// token, minus the same with the question blanked out.
    pub fn model_scores(&self, q: &Question) -> Vec<f32> {
        let Some(brain) = self.brain else { return vec![0.0; q.options.len()] };
        let options: Vec<String> = q.options.iter().map(|o| format!(" {o}\n")).collect();
        let mean = |v: Vec<(f32, usize)>| -> Vec<f32> { v.into_iter().map(|(t, n)| t / n.max(1) as f32).collect() };
        let mut s = mean(brain.score_continuations(&q.prompt(false), &options));
        if self.settings.contextual_calibration {
            let prior = mean(brain.score_continuations(&q.prompt(true), &options));
            for (x, p) in s.iter_mut().zip(prior) {
                *x -= p;
            }
        }
        s
    }

    /// What experience (your past feedback) says of each option.
    pub fn experience_scores(&self, q: &Question) -> Vec<f32> {
        let key = q.key();
        q.options.iter().map(|o| self.experience.score(&key, o)).collect()
    }

    pub fn decide(&self, q: Question) -> Decision {
        let start = Instant::now();
        if q.options.is_empty() {
            let mut d = conclude(Question { options: vec!["-".into()], ..q }, vec![0.0], 1.0, &self.settings);
            d.pick = None;
            d.reason = "there were no options to choose from".into();
            return d;
        }
        let (model, learned) = (self.model_scores(&q), self.experience_scores(&q));
        let scores = model.iter().zip(&learned).map(|(a, b)| a + b).collect();
        let mut d = conclude(q, scores, self.calibration.temperature, &self.settings);
        d.from_model = model;
        d.from_experience = learned;
        d.millis = start.elapsed().as_secs_f64() * 1e3;
        if let Some(log) = &self.log {
            d.id = log.record(&d);
        }
        d
    }
}

/// One logged decision, and whether it turned out right.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    #[serde(default)]
    decision: Option<Decision>,
    /// A later line: feedback on decision `id`.
    #[serde(default)]
    feedback: Option<u64>,
    #[serde(default)]
    correct: Option<bool>,
}

/// The decision log (data/memory/decisions.jsonl) and the calibration fitted on it.
pub struct Log {
    dir: PathBuf,
}

/// What `learn` found.
#[derive(Debug)]
pub struct Learned {
    pub judged: usize,
    pub accuracy: f32,
    pub before: Calibration,
    pub after: Calibration,
    /// Expected calibration error (how far stated confidence is from how
    /// often it's right), before and after.
    pub error_before: f32,
    pub error_after: f32,
}

impl Log {
    pub fn new(settings: &Settings) -> Log {
        Log { dir: settings.data_path("memory") }
    }

    fn path(&self) -> PathBuf {
        self.dir.join("decisions.jsonl")
    }

    fn calibration_path(&self) -> PathBuf {
        self.dir.join("calibration.json")
    }

    pub fn calibration(&self) -> Calibration {
        std::fs::read_to_string(self.calibration_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    fn append(&self, e: &Entry) {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(self.path()) {
            let _ = writeln!(f, "{}", serde_json::to_string(e).unwrap_or_default());
        }
    }

    fn entries(&self) -> Vec<Entry> {
        std::fs::read_to_string(self.path()).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    /// Log a decision; returns its id.
    pub fn record(&self, d: &Decision) -> u64 {
        let id = self.entries().iter().filter_map(|e| e.decision.as_ref().map(|d| d.id)).max().unwrap_or(0) + 1;
        let mut d = d.clone();
        d.id = id;
        self.append(&Entry { decision: Some(d), feedback: None, correct: None });
        id
    }

    /// The most recent decision.
    pub fn last(&self) -> Option<Decision> {
        self.entries().into_iter().rev().find_map(|e| e.decision)
    }

    /// Say whether decision `id` (or the latest) was right.
    pub fn feedback(&self, id: Option<u64>, correct: bool) -> Result<Decision> {
        let d = match id {
            Some(id) => self.entries().into_iter().find_map(|e| e.decision.filter(|d| d.id == id)),
            None => self.last(),
        };
        let Some(d) = d else { anyhow::bail!("there's no decision to give feedback on yet") };
        self.append(&Entry { decision: None, feedback: Some(d.id), correct: Some(correct) });
        // teach experience: the option it leaned to was right (so the others weren't), or wasn't
        if let Some(best) = (0..d.probs.len()).max_by(|&a, &b| d.probs[a].total_cmp(&d.probs[b])) {
            let mut x = Experience::load_from(&self.dir);
            let key = d.question.key();
            x.learn(&key, &d.question.options[best], correct);
            if correct && d.question.kind != Kind::Score {
                for (_, o) in d.question.options.iter().enumerate().filter(|(i, _)| *i != best) {
                    x.learn(&key, o, false);
                }
            }
            x.save_to(&self.dir)?;
        }
        Ok(d)
    }

    /// The right answer to the latest decision was `option`: learn it.
    pub fn answer(&self, option: &str) -> Result<Decision> {
        let Some(d) = self.last() else { anyhow::bail!("there's no decision to give the answer to yet") };
        let Some(right) = d.question.options.iter().position(|o| o.eq_ignore_ascii_case(option.trim())) else {
            anyhow::bail!("{:?} wasn't one of the options ({})", option.trim(), d.question.options.join(" | "));
        };
        let best = (0..d.probs.len()).max_by(|&a, &b| d.probs[a].total_cmp(&d.probs[b])).unwrap_or(0);
        self.append(&Entry { decision: None, feedback: Some(d.id), correct: Some(best == right) });
        let mut x = Experience::load_from(&self.dir);
        let key = d.question.key();
        for (i, o) in d.question.options.iter().enumerate() {
            x.learn(&key, o, i == right);
        }
        x.save_to(&self.dir)?;
        Ok(d)
    }

    /// Decisions with feedback: (scores, the option it would pick, whether that was right).
    fn judged(&self) -> Vec<(Vec<f32>, usize, bool)> {
        let entries = self.entries();
        let mut verdict = std::collections::HashMap::new();
        for e in &entries {
            if let (Some(id), Some(c)) = (e.feedback, e.correct) {
                verdict.insert(id, c);
            }
        }
        entries
            .into_iter()
            .filter_map(|e| e.decision)
            .filter_map(|d| {
                let c = *verdict.get(&d.id)?;
                let best = (0..d.probs.len()).max_by(|&a, &b| d.probs[a].total_cmp(&d.probs[b]))?;
                Some((d.scores, best, c))
            })
            .collect()
    }

    /// Refit the temperature on every decision with feedback, so the
    /// confidence it states matches how often it turned out right.
    pub fn learn(&self) -> Result<Learned> {
        let judged = self.judged();
        let before = self.calibration();
        let after = fit(&judged).unwrap_or_else(|| before.clone());
        let accuracy = if judged.is_empty() { 0.0 } else { judged.iter().filter(|j| j.2).count() as f32 / judged.len() as f32 };
        let learned = Learned { judged: judged.len(), accuracy, error_before: calibration_error(&judged, before.temperature),
                                error_after: calibration_error(&judged, after.temperature), before, after };
        std::fs::write(self.calibration_path(), serde_json::to_string_pretty(&learned.after)?)?;
        Ok(learned)
    }
}

fn log_loss(judged: &[(Vec<f32>, usize, bool)], t: f32) -> f32 {
    judged
        .iter()
        .map(|(s, best, correct)| {
            let p = softmax(s, t)[*best].clamp(1e-4, 1.0 - 1e-4);
            if *correct { -p.ln() } else { -(1.0 - p).ln() }
        })
        .sum()
}

/// The temperature that makes the stated confidence fit the outcomes best
/// (least log loss), by golden-section search over log-temperature.
pub fn fit(judged: &[(Vec<f32>, usize, bool)]) -> Option<Calibration> {
    if judged.len() < 3 {
        return None;
    }
    let f = |lt: f32| log_loss(judged, lt.exp());
    let (mut a, mut b) = ((0.02f32).ln(), (50f32).ln());
    let g = (5f32.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if f(c) < f(d) {
            b = d;
        } else {
            a = c;
        }
    }
    Some(Calibration { temperature: ((a + b) / 2.0).exp(), fitted_on: judged.len() })
}

/// Expected calibration error over 5 confidence bins.
pub fn calibration_error(judged: &[(Vec<f32>, usize, bool)], t: f32) -> f32 {
    if judged.is_empty() {
        return 0.0;
    }
    let mut bins = [(0f32, 0f32, 0usize); 5];
    for (s, best, correct) in judged {
        let p = softmax(s, t)[*best];
        let b = ((p * 5.0) as usize).min(4);
        bins[b].0 += p;
        bins[b].1 += *correct as u8 as f32;
        bins[b].2 += 1;
    }
    bins.iter().filter(|b| b.2 > 0).map(|b| (b.0 - b.1).abs()).sum::<f32>() / judged.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Precision;
    use crate::trainer::tests::trained;

    #[test]
    fn it_abstains_unless_one_option_clearly_wins() {
        let s = DecideSettings::default();
        let q = Question::choose("which?", &["a", "b", "c"]);
        let d = conclude(q.clone(), vec![3.0, 0.0, 0.0], 1.0, &s);
        assert_eq!((d.pick, d.answer()), (Some(0), "a"));
        let close = conclude(q.clone(), vec![2.0, 1.9, -5.0], 1.0, &s);
        assert_eq!(close.pick, None);
        assert!(close.reason.contains("too close") && close.answer() == "I don't know", "{}", close.reason);
        let flat = conclude(q, vec![0.1, 0.0, 0.05], 1.0, &s);
        assert!(flat.pick.is_none() && flat.reason.contains("only"), "{}", flat.reason);
        let score = conclude(Question::score("how good?", 1, 3), vec![0.0, 0.0, 9.0], 1.0, &s);
        assert!((score.expected.unwrap() - 3.0).abs() < 0.01);
        assert_eq!(Question::truth("2 > 1").options, vec!["yes", "no"]);
    }

    #[test]
    fn feedback_teaches_it_how_sure_to_be() {
        // an overconfident decider: big score gaps, but right only 60% of the time
        let judged: Vec<(Vec<f32>, usize, bool)> = (0..50).map(|i| (vec![4.0, 0.0], 0, i % 5 < 3)).collect();
        assert!(calibration_error(&judged, 1.0) > 0.3);
        let c = fit(&judged).unwrap();
        assert!(c.temperature > 5.0, "{c:?}");
        let p = softmax(&judged[0].0, c.temperature)[0];
        assert!((p - 0.6).abs() < 0.03, "p={p}");
        assert!(calibration_error(&judged, c.temperature) < 0.05);
        assert!(fit(&judged[..2]).is_none());
    }

    #[test]
    fn decisions_are_logged_judged_and_learned_from() {
        let (tmp, s) = trained(30);
        let brain = Brain::load_at(&s, Precision::F32).unwrap();
        let decider = Decider::new(Some(&brain), &s);
        let d = decider.decide(Question::choose("what does add return", &["a + b", "a - b"]).with_context("def add(a, b):"));
        assert_eq!(d.id, 1);
        assert_eq!(d.probs.len(), 2);
        assert!((d.probs.iter().sum::<f32>() - 1.0).abs() < 1e-4 && d.millis >= 0.0);
        let log = Log::new(&s);
        for _ in 0..3 {
            decider.decide(Question::truth("1 + 1 == 2"));
        }
        assert_eq!(log.last().unwrap().id, 4);
        assert_eq!(log.feedback(Some(1), true).unwrap().id, 1);
        log.feedback(None, false).unwrap();
        log.feedback(Some(2), true).unwrap();
        let learned = log.learn().unwrap();
        assert_eq!(learned.judged, 3);
        assert!(Log::new(&s).calibration().fitted_on == 3);
        assert!(Decider::ephemeral(Some(&brain), &s.decide).decide(Question::choose("?", &[])).pick.is_none());
        drop(tmp);
    }

    #[test]
    fn without_a_model_it_decides_from_what_you_taught_it() {
        let tmp = tempfile::tempdir().unwrap();
        let s = crate::config::testing::settings(tmp.path());
        let q = || Question::choose("which sort for nearly sorted data", &["insertion sort", "quick sort", "heap sort"]);
        let first = Decider::new(None, &s).decide(q());
        assert!(first.pick.is_none(), "no model, no experience: I don't know");
        let log = Log::new(&s);
        log.answer("insertion sort").unwrap();
        assert!(log.answer("bubble sort").is_err());
        let d = Decider::new(None, &s).decide(q());
        assert_eq!(d.answer(), "insertion sort", "{d:?}");
        log.feedback(None, true).unwrap();
        let again = Decider::new(None, &s).decide(q());
        assert!(again.confidence > d.confidence && again.from_experience[0] > 0.0);
    }
}
