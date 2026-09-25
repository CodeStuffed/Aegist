//! A line per council session in data/sessions.jsonl. The research loop
//! reads it back to find subjects that keep coming up.

use crate::config::Settings;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub at: String,
    pub claim: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub stance: String,
    #[serde(default)]
    pub confidence: String,
}

fn path(settings: &Settings) -> std::path::PathBuf {
    settings.data_path("").join("sessions.jsonl")
}

pub fn record(settings: &Settings, entry: &Session) -> Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path(settings))?;
    writeln!(f, "{}", serde_json::to_string(entry)?)?;
    Ok(())
}

pub fn sessions(settings: &Settings) -> Vec<Session> {
    std::fs::read_to_string(path(settings))
        .map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
        .unwrap_or_default()
}

/// Keywords across all past claims, most frequent first (ties: first seen first).
pub fn keyword_counts(settings: &Settings) -> Vec<(String, usize)> {
    let mut order: Vec<(String, usize)> = Vec::new();
    for s in sessions(settings) {
        for k in s.keywords {
            match order.iter_mut().find(|(w, _)| *w == k) {
                Some(e) => e.1 += 1,
                None => order.push((k, 1)),
            }
        }
    }
    order.sort_by_key(|e| std::cmp::Reverse(e.1)); // stable
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    #[test]
    fn counts_keywords_across_sessions() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        for kw in [vec!["pricing"], vec!["pricing", "churn"], vec!["orbit"]] {
            let entry = Session { at: "t".into(), claim: "c".into(), keywords: kw.into_iter().map(String::from).collect(),
                                  stance: "yes".into(), confidence: "Low".into() };
            record(&s, &entry).unwrap();
        }
        assert_eq!(keyword_counts(&s)[0], ("pricing".to_string(), 2));
        assert_eq!(sessions(&s).len(), 3);
    }
}
