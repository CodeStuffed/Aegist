//! The local knowledge base: passages of text with metadata, searchable with
//! BM25 (the classic ranking function behind keyword search engines),
//! implemented here from scratch - no embedding model, no outside AI.
//!
//! Stored one JSON object per line in data/knowledge_base/passages.jsonl and
//! indexed in memory on load.

use crate::config::Settings;
use crate::text::content_words;
use crate::util::fnv1a;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

const K1: f64 = 1.5;
const B: f64 = 0.75;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl Passage {
    pub fn meta(&self, key: &str) -> &str {
        self.metadata.get(key).and_then(|v| v.as_str()).unwrap_or("")
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    #[serde(flatten)]
    pub passage: Passage,
    pub score: f64,
    /// Score as a fraction of the best this query could get.
    pub relevance: f64,
}

pub struct KnowledgeStore {
    path: PathBuf,
    docs: Vec<Passage>,
    hashes: HashSet<u64>,
    postings: HashMap<String, Vec<(u32, u32)>>, // term -> (doc, count)
    lengths: Vec<u32>,
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl KnowledgeStore {
    pub fn path_for(settings: &Settings) -> PathBuf {
        settings.data_path("knowledge_base").join("passages.jsonl")
    }

    pub fn open(settings: &Settings) -> Result<Self> {
        let path = Self::path_for(settings);
        let mut kb = KnowledgeStore { path: path.clone(), docs: Vec::new(), hashes: HashSet::new(), postings: HashMap::new(), lengths: Vec::new() };
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(p) = serde_json::from_str::<Passage>(line) {
                    kb.index(p);
                }
            }
        }
        Ok(kb)
    }

    fn index(&mut self, p: Passage) {
        let doc = self.docs.len() as u32;
        self.hashes.insert(fnv1a(normalize(&p.text).as_bytes()));
        let mut counts: HashMap<String, u32> = HashMap::new();
        for w in content_words(&p.text) {
            *counts.entry(w).or_default() += 1;
        }
        self.lengths.push(counts.values().sum());
        for (term, n) in counts {
            self.postings.entry(term).or_default().push((doc, n));
        }
        self.docs.push(p);
    }

    /// Store a passage. Returns its id, or None if it was already stored.
    pub fn add(&mut self, text: &str, metadata: Map<String, Value>) -> Result<Option<String>> {
        let text = normalize(text);
        let hash = fnv1a(text.as_bytes());
        if text.is_empty() || self.hashes.contains(&hash) {
            return Ok(None);
        }
        let p = Passage { id: format!("{hash:016x}"), text, metadata };
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(&p)?)?;
        let id = p.id.clone();
        self.index(p);
        Ok(Some(id))
    }

    fn idf(&self, term: &str) -> f64 {
        let n = self.docs.len() as f64;
        let df = self.postings.get(term).map_or(0, |p| p.len()) as f64;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// idf, except a word no passage contains counts as if one did -
    /// otherwise one unheard-of word would swamp the rest of the query.
    fn reference_idf(&self, term: &str) -> f64 {
        if self.postings.contains_key(term) {
            self.idf(term)
        } else {
            (1.0 + (self.docs.len() as f64 - 0.5) / 1.5).ln()
        }
    }

    /// Top-k passages by BM25, keeping those whose relevance (score as a
    /// fraction of what a typical passage containing every query word once
    /// would get) is at least `min_relevance`.
    pub fn search(&self, query: &str, k: usize, min_relevance: f64) -> Vec<Hit> {
        let terms: HashSet<String> = content_words(query).into_iter().collect();
        if self.docs.is_empty() || terms.is_empty() {
            return Vec::new();
        }
        let avg = self.lengths.iter().map(|&l| l as f64).sum::<f64>() / self.lengths.len() as f64;
        let avg = if avg > 0.0 { avg } else { 1.0 };
        let mut scores: HashMap<u32, f64> = HashMap::new();
        for term in &terms {
            let idf = self.idf(term);
            for &(doc, tf) in self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[]) {
                let tf = tf as f64;
                let norm = tf + K1 * (1.0 - B + B * self.lengths[doc as usize] as f64 / avg);
                *scores.entry(doc).or_default() += idf * tf * (K1 + 1.0) / norm;
            }
        }
        let best: f64 = terms.iter().map(|t| self.reference_idf(t)).sum();
        let mut ranked: Vec<(u32, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(k)
            .map(|(doc, s)| (doc, s, (s / best).min(1.0)))
            .filter(|&(_, _, r)| r >= min_relevance)
            .map(|(doc, s, r)| Hit {
                passage: self.docs[doc as usize].clone(),
                score: (s * 1000.0).round() / 1000.0,
                relevance: (r * 1000.0).round() / 1000.0,
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    fn meta(title: &str) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("title".into(), Value::from(title));
        m
    }

    #[test]
    fn ranks_dedupes_and_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        assert!(kb.search("anything", 3, 0.0).is_empty());
        kb.add("Usage-based pricing charges customers per unit consumed.", meta("Pricing")).unwrap();
        kb.add("Glass refracts light because light slows down in it.", meta("Optics")).unwrap();
        kb.add("Subscription pricing charges a flat monthly fee.", meta("Subscriptions")).unwrap();
        assert!(kb.add("Glass refracts  light because\nlight slows down in it.", meta("dupe")).unwrap().is_none());

        let hits = kb.search("should we move to usage-based pricing?", 2, 0.0);
        assert_eq!(hits.iter().map(|h| h.passage.meta("title")).collect::<Vec<_>>(), vec!["Pricing", "Subscriptions"]);
        let top = kb.search("refracts light glass", 5, 0.35);
        assert_eq!(top.len(), 1);
        assert!(top[0].passage.meta("title") == "Optics" && top[0].relevance > 0.35 && top[0].relevance <= 1.0);
        assert!(kb.search("kangaroo", 5, 0.0).is_empty());

        let reopened = KnowledgeStore::open(&s).unwrap();
        assert_eq!(reopened.count(), 3);
    }

    #[test]
    fn a_weak_one_word_coincidence_is_not_relevant() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("Customers pay for products that are profitable to sell.", meta("Markets")).unwrap();
        kb.add("Houses in the suburbs are built from timber.", meta("Houses")).unwrap();
        assert_eq!(kb.search("Customers will pay for a profitable subscription", 3, 0.35)[0].passage.meta("title"), "Markets");
        assert!(kb.search("Kangaroos can jump over tall houses easily", 3, 0.35).is_empty());
    }

    #[test]
    fn reads_passages_written_by_the_python_version() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let line = r#"{"id": "f5b7487aa8459be9", "text": "Customers pay for products.", "metadata": {"title": "Markets", "url": "https://x"}}"#;
        std::fs::write(KnowledgeStore::path_for(&s), format!("{line}\n")).unwrap();
        let mut kb = KnowledgeStore::open(&s).unwrap();
        assert_eq!(kb.count(), 1);
        assert!(kb.add("Customers pay for products.", Map::new()).unwrap().is_none());
    }
}
