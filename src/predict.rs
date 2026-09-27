//! Estimating and predicting, from scratch and honestly.
//!
//!   forecast   the next values of a series of numbers. Several simple
//!              models (flat, straight trend, growth, repeating pattern)
//!              are each tested on the series' own past - predicting every
//!              point from the ones before it - and the one that did best
//!              predicts. The range it gives is how far off it was on the
//!              past, widening further out. Too little data, or a past it
//!              couldn't predict: it says so instead of drawing a line.
//!   estimate   a quantity like ones you've told it before ("minutes to
//!              build the project = 4"): the values of the most similar
//!              things it knows, weighted by similarity, as a middle value
//!              and a range. Nothing similar: "I don't know".
//!   Experience fast, learned preferences for typed decisions (`decide`):
//!              every "good", "bad" or right answer you give nudges the
//!              weights of word-pair features, so the next similar question
//!              leans the way you taught it - in microseconds, no model run.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

// ------------------------------------------------------------------ forecast

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Stays at the recent level.
    Flat,
    /// Goes up or down by the same amount each step.
    Trend,
    /// Grows or shrinks by the same factor each step.
    Growth,
    /// Repeats a pattern of this many steps.
    Pattern(usize),
}

impl Method {
    pub fn describe(&self) -> String {
        match self {
            Method::Flat => "it stays about level".into(),
            Method::Trend => "it changes by a steady amount each step".into(),
            Method::Growth => "it grows by a steady factor each step".into(),
            Method::Pattern(p) => format!("it repeats every {p} steps"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Forecast {
    pub method: Method,
    /// (prediction, low, high) for each step ahead (an 80% range).
    pub steps: Vec<(f64, f64, f64)>,
    /// Typical one-step error on the series' own past.
    pub past_error: f64,
    /// How many past points it was tested on.
    pub tested: usize,
}

/// Predict x[n] (and further) from x[..n] with `m`, `h` steps ahead.
fn predict_with(m: Method, xs: &[f64], h: usize) -> Option<f64> {
    let n = xs.len();
    match m {
        Method::Flat => {
            let k = n.min(3);
            (k > 0).then(|| xs[n - k..].iter().sum::<f64>() / k as f64)
        }
        Method::Trend => {
            if n < 2 {
                return None;
            }
            // least squares over the recent part
            let k = n.min(8);
            let ys = &xs[n - k..];
            let mx = (k - 1) as f64 / 2.0;
            let my = ys.iter().sum::<f64>() / k as f64;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (i, y) in ys.iter().enumerate() {
                sxy += (i as f64 - mx) * (y - my);
                sxx += (i as f64 - mx).powi(2);
            }
            let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
            Some(my + slope * ((k - 1) as f64 - mx + h as f64))
        }
        Method::Growth => {
            if n < 2 || xs.iter().any(|&x| x <= 0.0) {
                return None;
            }
            let logs: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
            predict_with(Method::Trend, &logs, h).map(f64::exp)
        }
        Method::Pattern(p) => {
            if n < p * 2 {
                return None;
            }
            // last cycle, shifted by how much each cycle moved on average
            let base = xs[n - p + (h - 1) % p];
            let cycles = n / p;
            let drift = (xs[n - 1] - xs[n - 1 - p * (cycles - 1)]) / (cycles - 1).max(1) as f64;
            Some(base + drift * ((h - 1) / p + 1) as f64 * (cycles > 1) as u8 as f64)
        }
    }
}

/// Forecast `h` steps after `xs`, or say why not.
pub fn forecast(xs: &[f64], h: usize) -> Result<Forecast, String> {
    let n = xs.len();
    if n < 3 {
        return Err(format!("I need at least 3 numbers to see a pattern; I have {n}."));
    }
    let h = h.clamp(1, 50);
    let mut methods = vec![Method::Flat, Method::Trend, Method::Growth];
    methods.extend((2..=n / 2).take(12).map(Method::Pattern));
    // test each on the past: predict every point (after the first few) from the ones before it
    let first = (n / 3).max(2);
    // simpler methods win near-ties: a pattern must earn its keep
    let penalty = |m: Method| match m {
        Method::Flat => 1.0,
        Method::Trend | Method::Growth => 1.02,
        Method::Pattern(_) => 1.05,
    };
    let mut best: Option<(Method, Vec<f64>, f64)> = None;
    for m in methods {
        // every past point this method can predict from the ones before it
        let errs: Vec<f64> = (first..n).filter_map(|t| predict_with(m, &xs[..t], 1).map(|p| (p - xs[t]).abs())).collect();
        // a repeating pattern needs two cycles of evidence; the rest, two points
        let need = if let Method::Pattern(p) = m { 2 * p } else { 2.min(n - first) };
        if errs.len() < need {
            continue;
        }
        let score = errs.iter().sum::<f64>() / errs.len() as f64 * penalty(m) + 1e-12 * penalty(m);
        if best.as_ref().is_none_or(|b| score < b.2) {
            best = Some((m, errs, score));
        }
    }
    // anything cleverer than "stays level" has to beat it clearly
    let flat = (first..n).filter_map(|t| predict_with(Method::Flat, &xs[..t], 1).map(|p| (p - xs[t]).abs())).collect::<Vec<f64>>();
    let flat_err = flat.iter().sum::<f64>() / flat.len().max(1) as f64;
    let best = best.map(|(m, e, score)| {
        if m != Method::Flat && flat_err > 0.0 && score > flat_err * 0.7 { (Method::Flat, flat.clone()) } else { (m, e) }
    });
    let (method, mut errs) = best.ok_or("none of my methods could be tested on this series")?;
    let scale = xs.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-9);
    errs.sort_by(|a, b| a.total_cmp(b));
    let past = errs.iter().sum::<f64>() / errs.len() as f64;
    let level = xs.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
    if method == Method::Flat && past > level.max(1e-9) * 0.35 {
        return Err(format!(
            "I don't know: on this series' own past I was off by {} on average, a big part of the numbers themselves, \
             and nothing I tried (a trend, growth, a repeating pattern) did clearly better. It looks random to me, so any \
             prediction would be a guess.",
            fmt(past)
        ));
    }
    // 80% range: the 80th-percentile past error (at least a hair), growing with the distance ahead
    let q80 = errs[((errs.len() as f64 * 0.8).ceil() as usize).saturating_sub(1).min(errs.len() - 1)].max(scale * 1e-6);
    let steps = (1..=h)
        .map(|k| {
            let v = predict_with(method, xs, k).unwrap_or(f64::NAN);
            let w = q80 * (k as f64).sqrt();
            (v, v - w, v + w)
        })
        .collect();
    Ok(Forecast { method, steps, past_error: past, tested: errs.len() })
}

/// A number, without noise digits.
pub fn fmt(x: f64) -> String {
    if !x.is_finite() {
        return "?".into();
    }
    let a = x.abs();
    let s = if a >= 1000.0 || (a - a.round()).abs() < 1e-9 {
        format!("{:.0}", x)
    } else if a >= 10.0 {
        format!("{:.1}", x)
    } else {
        format!("{:.3}", x).trim_end_matches('0').trim_end_matches('.').to_string()
    };
    if s == "-0" { "0".into() } else { s }
}

/// The numbers in a piece of text: "3, 5.5 and -7" -> [3, 5.5, -7].
pub fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter_map(|w| w.trim_matches('.').parse::<f64>().ok())
        .filter(|x| x.is_finite())
        .collect()
}

// ------------------------------------------------------------------ features

fn words(text: &str) -> Vec<String> {
    const STOP: &[&str] = &["the", "a", "an", "of", "to", "in", "is", "it", "for", "on", "and", "or", "my", "how", "what", "which", "do",
                            "does", "be", "this", "that", "with", "much", "many", "long"];
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOP.contains(w))
        .map(|w| w.trim_end_matches('s').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn h(parts: &[&str]) -> u64 {
    crate::util::fnv1a(parts.join("\u{1f}").as_bytes())
}

/// Similarity of two texts by their words (Jaccard), 0..1.
pub fn similarity(a: &str, b: &str) -> f32 {
    let (a, b): (std::collections::HashSet<String>, std::collections::HashSet<String>) = (words(a).into_iter().collect(), words(b).into_iter().collect());
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.intersection(&b).count() as f32 / a.union(&b).count() as f32
}

// ------------------------------------------------------------------ estimate

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Known {
    pub what: String,
    pub value: f64,
    pub at: String,
}

#[derive(Clone, Debug)]
pub struct Estimate {
    pub middle: f64,
    pub low: f64,
    pub high: f64,
    /// The similar things it's based on, with their similarity.
    pub from: Vec<(Known, f32)>,
}

pub struct Estimates {
    path: PathBuf,
}

impl Estimates {
    pub fn new(settings: &crate::config::Settings) -> Estimates {
        Estimates { path: settings.data_path("memory").join("estimates.jsonl") }
    }

    pub fn all(&self) -> Vec<Known> {
        std::fs::read_to_string(&self.path).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    pub fn teach(&self, what: &str, value: f64) -> Result<()> {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(&Known { what: what.trim().into(), value, at: crate::util::iso_now() })?)?;
        Ok(())
    }

    /// Estimate `what` from the most similar things it knows, or say why it can't.
    pub fn estimate(&self, what: &str) -> Result<Estimate, String> {
        let all = self.all();
        if all.is_empty() {
            return Err("I don't know: I haven't been told any values yet. Teach me with \"estimate <thing> = <number>\", \
                        and the next similar question gets an answer."
                .into());
        }
        let mut near: Vec<(Known, f32)> = all.into_iter().map(|k| {
            let s = similarity(what, &k.what);
            (k, s)
        }).filter(|(_, s)| *s >= 0.34).collect();
        if near.is_empty() {
            return Err(format!("I don't know: nothing I've been told is like \"{}\".", what.trim()));
        }
        near.sort_by(|a, b| b.1.total_cmp(&a.1));
        near.truncate(25);
        // recent-ish and similar things count most: weight = similarity squared
        let mut vals: Vec<(f64, f32)> = near.iter().map(|(k, s)| (k.value, s * s)).collect();
        vals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f32 = vals.iter().map(|v| v.1).sum();
        // weighted quantiles, interpolating between values (4 and 6 -> 5 in the middle)
        let mut centers = Vec::with_capacity(vals.len());
        let mut acc = 0.0;
        for &(_, w) in &vals {
            centers.push((acc + w / 2.0) / total);
            acc += w;
        }
        let quantile = |q: f32| -> f64 {
            let i = centers.iter().position(|&c| c >= q).unwrap_or(vals.len() - 1);
            if i == 0 || centers[i] <= q {
                return vals[i].0;
            }
            let t = ((q - centers[i - 1]) / (centers[i] - centers[i - 1])) as f64;
            vals[i - 1].0 + (vals[i].0 - vals[i - 1].0) * t
        };
        Ok(Estimate { middle: quantile(0.5), low: quantile(0.1), high: quantile(0.9), from: near })
    }
}

// ---------------------------------------------------------------- experience

/// Learned leanings for decisions: a sparse logistic model over hashed
/// (question word, option word) pairs, option words and exact
/// (question, option) pairs, trained online from your feedback.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Experience {
    weights: HashMap<u64, f32>,
    pub lessons: usize,
}

impl Experience {
    pub fn load(settings: &crate::config::Settings) -> Experience {
        Self::load_from(&settings.data_path("memory"))
    }

    /// From a memory folder.
    pub fn load_from(dir: &std::path::Path) -> Experience {
        std::fs::read_to_string(dir.join("experience.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save_to(&self, dir: &std::path::Path) -> Result<()> {
        crate::util::write_atomic(&dir.join("experience.json"), serde_json::to_string(self)?.as_bytes())
    }

    fn features(question: &str, option: &str) -> Vec<u64> {
        let (qw, ow) = (words(question), words(option));
        let mut f = vec![h(&["x", &question.trim().to_lowercase(), &option.trim().to_lowercase()])];
        for o in &ow {
            f.push(h(&["o", o]));
            for q in &qw {
                f.push(h(&["p", q, o]));
            }
        }
        if ow.is_empty() {
            f.push(h(&["o", &option.trim().to_lowercase()]));
        }
        f
    }

    /// How much experience favors `option` for `question` (0 = no opinion).
    pub fn score(&self, question: &str, option: &str) -> f32 {
        let f = Self::features(question, option);
        f.iter().map(|k| self.weights.get(k).copied().unwrap_or(0.0)).sum::<f32>() / (f.len() as f32).sqrt()
    }

    /// Learn that `option` was (`right`) or wasn't the answer to `question`.
    pub fn learn(&mut self, question: &str, option: &str, right: bool) {
        let f = Self::features(question, option);
        let norm = (f.len() as f32).sqrt();
        let p = 1.0 / (1.0 + (-self.score(question, option)).exp());
        let g = (right as u8 as f32 - p) * 3.0 / norm;
        for k in f {
            let w = self.weights.entry(k).or_default();
            *w = (*w + g).clamp(-8.0, 8.0);
        }
        self.lessons += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecasts_pick_the_method_the_past_supports() {
        let f = forecast(&[3.0, 5.0, 7.0, 9.0, 11.0], 2).unwrap();
        assert_eq!(f.method, Method::Trend);
        assert!((f.steps[0].0 - 13.0).abs() < 1e-6 && (f.steps[1].0 - 15.0).abs() < 1e-6, "{f:?}");
        let g = forecast(&[2.0, 4.0, 8.0, 16.0, 32.0, 64.0], 1).unwrap();
        assert_eq!(g.method, Method::Growth);
        assert!((g.steps[0].0 - 128.0).abs() < 1.0, "{g:?}");
        let p = forecast(&[1.0, 5.0, 2.0, 1.0, 5.0, 2.0, 1.0, 5.0, 2.0, 1.0, 5.0, 2.0], 3).unwrap();
        assert_eq!(p.method, Method::Pattern(3));
        assert!((p.steps[0].0 - 1.0).abs() < 1e-6 && (p.steps[1].0 - 5.0).abs() < 1e-6);
        let noisy = forecast(&[10.0, 11.0, 9.5, 10.4, 10.1, 9.8, 10.6, 10.2], 1).unwrap();
        assert!(noisy.steps[0].1 < 10.2 && noisy.steps[0].2 > 10.0, "{noisy:?}");
        assert!(forecast(&[1.0, 2.0], 1).unwrap_err().contains("at least 3"));
        let random = [5.0, 91.0, 12.0, 70.0, 3.0, 55.0, 99.0, 1.0, 40.0, 77.0];
        assert!(forecast(&random, 1).unwrap_err().contains("random"), "{:?}", forecast(&random, 1));
        assert_eq!(numbers("3, 5.5 and -7."), vec![3.0, 5.5, -7.0]);
        assert_eq!(fmt(2.50), "2.5");
        assert_eq!(fmt(13.0), "13");
    }

    #[test]
    fn estimates_come_from_similar_things_it_was_told() {
        let tmp = tempfile::tempdir().unwrap();
        let s = crate::config::testing::settings(tmp.path());
        let e = Estimates::new(&s);
        assert!(e.estimate("minutes to build the project").unwrap_err().contains("don't know"));
        for v in [4.0, 5.0, 4.5, 6.0] {
            e.teach("minutes to build the project", v).unwrap();
        }
        e.teach("hours of sleep", 7.5).unwrap();
        let est = e.estimate("how many minutes to build project").unwrap();
        assert!(est.middle >= 4.5 && est.middle <= 5.0 && est.low <= est.middle && est.high >= est.middle, "{est:?}");
        assert_eq!(est.from.len(), 4);
        assert!(e.estimate("price of a house").unwrap_err().contains("nothing I've been told"));
    }

    #[test]
    fn experience_leans_the_way_it_was_taught_and_generalizes_a_little() {
        let mut x = Experience::default();
        assert_eq!(x.score("which sort for nearly sorted data", "insertion sort"), 0.0);
        for _ in 0..3 {
            x.learn("which sort for nearly sorted data", "insertion sort", true);
            x.learn("which sort for nearly sorted data", "quick sort", false);
        }
        assert!(x.score("which sort for nearly sorted data", "insertion sort") > 1.0);
        assert!(x.score("which sort for nearly sorted data", "quick sort") < -1.0);
        // a reworded question shares word pairs, so it leans the same way, less strongly
        let a = x.score("best sort for data that is nearly sorted", "insertion sort");
        let b = x.score("best sort for data that is nearly sorted", "quick sort");
        assert!(a > b, "{a} {b}");
    }
}
