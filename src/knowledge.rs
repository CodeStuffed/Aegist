//! The local knowledge base: passages of text with metadata, searchable with
//! BM25 (the classic ranking function behind keyword search engines),
//! implemented here from scratch - no embedding model, no outside AI.
//!
//! Stored one JSON object per line in data/knowledge_base/passages.jsonl and
//! indexed in memory on load.
//!
//! Passages are also linked, like notes in Obsidian: to the paragraphs next
//! to them in the same article, and to passages sharing their rarest words.
//! `search_linked` finds the best matches, then follows those links to fill a
//! fixed text budget - the model's context window is small, so it gets the
//! relevant passage plus its closest context, and nothing else.

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
/// Links weaker than this (link strength x relevance of the source) aren't followed.
const LINK_MIN: f64 = 0.15;

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
    /// Score as a fraction of the best this query could get (for a linked
    /// passage: its link strength times the relevance of what linked to it).
    pub relevance: f64,
    /// How it was found: "match", or the link that led to it.
    pub via: String,
}

pub struct KnowledgeStore {
    path: PathBuf,
    docs: Vec<Passage>,
    hashes: HashSet<u64>,
    postings: HashMap<String, Vec<(u32, u32)>>, // term -> (doc, count)
    lengths: Vec<u32>,
    /// article (title, else url) -> its passages, in the order they were stored
    articles: HashMap<String, Vec<u32>>,
    /// each passage's position in its article's list
    position: Vec<Option<(String, usize)>>,
    /// each passage's distinct content words
    terms: Vec<Vec<String>>,
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
        let mut kb = KnowledgeStore {
            path: path.clone(),
            docs: Vec::new(),
            hashes: HashSet::new(),
            postings: HashMap::new(),
            lengths: Vec::new(),
            articles: HashMap::new(),
            position: Vec::new(),
            terms: Vec::new(),
        };
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
        let mut distinct: Vec<String> = counts.keys().cloned().collect();
        distinct.sort();
        self.terms.push(distinct);
        for (term, n) in counts {
            self.postings.entry(term).or_default().push((doc, n));
        }
        let article = [p.meta("title"), p.meta("url")].into_iter().find(|s| !s.is_empty()).map(str::to_string);
        self.position.push(article.map(|a| {
            let list = self.articles.entry(a.clone()).or_default();
            list.push(doc);
            (a, list.len() - 1)
        }));
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

    /// (doc, BM25 score, relevance) for the top-k passages with relevance
    /// (score as a fraction of what a typical passage containing every query
    /// word once would get) at least `min_relevance`.
    fn rank(&self, query: &str, k: usize, min_relevance: f64) -> Vec<(u32, f64, f64)> {
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
            .collect()
    }

    fn hit(&self, doc: u32, score: f64, relevance: f64, via: String) -> Hit {
        Hit {
            passage: self.docs[doc as usize].clone(),
            score: (score * 1000.0).round() / 1000.0,
            relevance: (relevance * 1000.0).round() / 1000.0,
            via,
        }
    }

    /// Top-k passages by BM25 relevance (no links followed).
    pub fn search(&self, query: &str, k: usize, min_relevance: f64) -> Vec<Hit> {
        self.rank(query, k, min_relevance)
            .into_iter()
            .map(|(doc, s, r)| self.hit(doc, s, r, "match".into()))
            .collect()
    }

    /// Passages linked to `doc`, with link strength in (0, 1]:
    /// - neighbors in the same article: 0.8 next door, 0.4 two away
    /// - passages sharing at least 2 of its 6 rarest words: up to 0.6
    fn links(&self, doc: u32) -> Vec<(u32, f64, String)> {
        let mut out: HashMap<u32, (f64, String)> = HashMap::new();
        let mut offer = |d: u32, w: f64, why: String| {
            if d != doc && out.get(&d).map_or(true, |(old, _)| w > *old) {
                out.insert(d, (w, why));
            }
        };
        if let Some((article, pos)) = &self.position[doc as usize] {
            let list = &self.articles[article];
            for (dist, w) in [(1usize, 0.8), (2, 0.4)] {
                for p in [pos.checked_sub(dist), Some(pos + dist)].into_iter().flatten() {
                    if let Some(&d) = list.get(p) {
                        offer(d, w, format!("linked: next to it in {article}"));
                    }
                }
            }
        }
        let mut rare: Vec<&String> = self.terms[doc as usize].iter().collect();
        rare.sort_by(|a, b| self.idf(b).total_cmp(&self.idf(a)).then(a.cmp(b)));
        rare.truncate(6);
        let mut shared: HashMap<u32, Vec<&str>> = HashMap::new();
        for t in &rare {
            for &(d, _) in self.postings.get(t.as_str()).map(|v| v.as_slice()).unwrap_or(&[]) {
                shared.entry(d).or_default().push(t.as_str());
            }
        }
        for (d, words) in shared {
            if words.len() >= 2 {
                let mut words = words;
                words.sort();
                let w = 0.6 * words.len() as f64 / rare.len().max(1) as f64;
                offer(d, w, format!("linked: shares {}", words.join(", ")));
            }
        }
        out.into_iter().map(|(d, (w, why))| (d, w, why)).collect()
    }

    /// Like Obsidian following links from a note: the best `k` matches, then
    /// the passages they link to (strongest links first), as long as the
    /// total text fits in `budget_chars`. Best matches come first.
    pub fn search_linked(&self, query: &str, k: usize, min_relevance: f64, budget_chars: usize) -> Vec<Hit> {
        let seeds = self.rank(query, k, min_relevance);
        let mut chosen: HashSet<u32> = HashSet::new();
        let mut out = Vec::new();
        let mut used = 0usize;
        let mut take = |doc: u32, score: f64, rel: f64, via: String, out: &mut Vec<Hit>, used: &mut usize| {
            let len = self.docs[doc as usize].text.len();
            // The best match is always used, even if it alone overflows (the
            // prompt is cut from the left, keeping the part nearest the claim).
            if chosen.contains(&doc) || (!chosen.is_empty() && *used + len > budget_chars) {
                return;
            }
            chosen.insert(doc);
            *used += len;
            out.push(self.hit(doc, score, rel, via));
        };
        for &(doc, s, r) in &seeds {
            take(doc, s, r, "match".into(), &mut out, &mut used);
        }
        let mut neighbors: Vec<(u32, f64, String)> = Vec::new();
        for &(doc, _, r) in &seeds {
            for (d, w, why) in self.links(doc) {
                neighbors.push((d, r * w, why));
            }
        }
        neighbors.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (d, strength, why) in neighbors {
            if strength >= LINK_MIN {
                take(d, 0.0, strength, why, &mut out, &mut used);
            }
        }
        out
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

    fn article(title: &str) -> Map<String, Value> {
        let mut m = meta(title);
        m.insert("url".into(), Value::from(format!("https://x/{title}")));
        m
    }

    #[test]
    fn linked_search_follows_article_neighbors_and_shared_words() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        // one article, three consecutive paragraphs; only the middle one matches the query
        kb.add("Newton studied how glass bends sunlight into a spectrum of colours.", article("Opticks")).unwrap();
        kb.add("Refraction through a prism separates white light into colours.", article("Opticks")).unwrap();
        kb.add("He measured the angles carefully with brass instruments.", article("Opticks")).unwrap();
        // another article sharing two rare words with the match, but none with the query
        kb.add("A prism splits candle light too.", article("Candles")).unwrap();
        // unrelated
        kb.add("Bees gather nectar from meadow flowers in spring.", article("Bees")).unwrap();

        let plain = kb.search("refraction separates white", 3, 0.2);
        assert_eq!(plain.len(), 1);

        let linked = kb.search_linked("refraction separates white", 3, 0.2, 10_000);
        let found: Vec<(&str, &str)> = linked.iter().map(|h| (h.passage.text.split(' ').next().unwrap(), h.via.as_str())).collect();
        assert_eq!(found[0], ("Refraction", "match"));
        assert!(found.iter().any(|(w, via)| *w == "Newton" && via.contains("next to it in Opticks")), "{found:?}");
        assert!(found.iter().any(|(w, via)| *w == "He" && via.contains("next to it")), "{found:?}");
        assert!(found.iter().any(|(w, via)| *w == "A" && via.contains("shares")), "{found:?}");
        assert!(!found.iter().any(|(w, _)| *w == "Bees"));
        // linked passages rank below the match and carry a link strength
        assert!(linked[1..].iter().all(|h| h.relevance < linked[0].relevance && h.relevance > 0.0));
    }

    #[test]
    fn linked_search_respects_the_budget_but_keeps_the_best_match() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("Refraction through a prism separates white light into colours.", article("Opticks")).unwrap();
        kb.add("Newton studied how glass bends sunlight into a spectrum of colours.", article("Opticks")).unwrap();
        let tight = kb.search_linked("refraction separates white", 3, 0.2, 70);
        assert_eq!(tight.len(), 1); // the neighbor doesn't fit
        let tiny = kb.search_linked("refraction separates white", 3, 0.2, 5);
        assert_eq!(tiny.len(), 1); // the best match is kept even over budget
        assert!(kb.search_linked("kangaroo", 3, 0.2, 1000).is_empty());
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
