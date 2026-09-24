//! `council eval`: how well the model tells true from false.
//!
//! Builds claim pairs from the first sentences of articles in the corpus
//! ("<Title> is a ...", "<Title> was a ..."): the true one, and the same
//! subject with another article's description. A pair is right when the true
//! claim gets the higher negation-test score. Reported for the raw and the
//! calibrated score, with and without knowledge-base evidence. Chance is 50%.

use crate::brain::Brain;
use crate::config::Settings;
use crate::corpus;
use crate::council::{context_budget_chars, evidence_then, negate, negation_test};
use crate::knowledge::KnowledgeStore;
use crate::rng::Rng;
use anyhow::Result;

/// (subject, verb, description) from "Title is a description, ..." sentences.
pub fn facts(settings: &Settings, max: usize) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for f in corpus::corpus_files(settings) {
        let text = std::fs::read_to_string(&f).unwrap_or_default();
        let paras: Vec<&str> = text.split("\n\n").collect();
        for w in paras.windows(2) {
            let (title, para) = (w[0].trim(), w[1]);
            if title.is_empty() || title.contains('.') || title.len() > 60 || title.split_whitespace().count() > 6 {
                continue;
            }
            for verb in ["is", "was"] {
                let Some(rest) = para.strip_prefix(&format!("{title} {verb} ")) else { continue };
                if !(rest.starts_with("a ") || rest.starts_with("an ") || rest.starts_with("the ")) {
                    continue;
                }
                let desc: Vec<&str> = rest.split([',', '.', ';', '(']).next().unwrap_or("").split_whitespace().take(8).collect();
                if desc.len() >= 3 {
                    out.push((title.to_string(), verb.to_string(), desc.join(" ")));
                }
            }
            if out.len() >= max {
                return out;
            }
        }
    }
    out
}

#[derive(Debug)]
pub struct EvalReport {
    pub pairs: usize,
    /// share right, [raw, calibrated] x [without evidence, with evidence]
    pub right: [[f64; 2]; 2],
}

pub fn negation_eval(brain: &Brain, settings: &Settings, max_pairs: usize) -> Result<EvalReport> {
    let kb = KnowledgeStore::open(settings)?;
    let facts = facts(settings, max_pairs);
    anyhow::ensure!(facts.len() >= 10,
        "found only {} usable article first sentences (\"<Title> is a ...\") in the corpus; `council import-wikipedia` provides plenty",
        facts.len());
    let mut rng = Rng::new(7);
    let mut right = [[0usize; 2]; 2];
    let mut n = 0;
    for (i, (subject, verb, desc)) in facts.iter().enumerate() {
        let Some(j) = (0..20).map(|_| rng.below(facts.len())).find(|&j| j != i && facts[j].1 == *verb && facts[j].2 != *desc) else { continue };
        let pair = [format!("{subject} {verb} {desc}"), format!("{subject} {verb} {}", facts[j].2)];
        let mut scores = [[[0f32; 2]; 2]; 2]; // [claim][evidence][raw, calibrated]
        for (c, claim) in pair.iter().enumerate() {
            let neg = negate(claim).expect("an is/was sentence has a verb to flip");
            for (e, with) in [false, true].into_iter().enumerate() {
                let hits = if with {
                    kb.search_linked(claim, settings.memory.top_k, settings.memory.min_relevance, context_budget_chars(brain, settings))
                } else {
                    Vec::new()
                };
                let refs: Vec<_> = hits.iter().collect();
                let evidence = brain.prefix(&evidence_then("", &refs), 64, None);
                let t = negation_test(brain, &neg, &evidence, settings, None);
                scores[c][e] = [t.raw_contrast, t.contrast];
            }
        }
        for (e, k) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            right[k][e] += (scores[0][e][k] > scores[1][e][k]) as usize;
        }
        n += 1;
    }
    let share = |x: usize| x as f64 / n.max(1) as f64;
    Ok(EvalReport { pairs: n, right: [[share(right[0][0]), share(right[0][1])], [share(right[1][0]), share(right[1][1])]] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    #[test]
    fn claims_come_from_article_first_sentences() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let dir = crate::corpus::corpus_dir(&s).join("wikipedia");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("part-00001.txt"), "Antimony\n\nAntimony is a chemical element with the symbol Sb, used in alloys.\n\n\
            It is found in nature.\n\nGold dollar\n\nThe gold dollar was a coin.\n\nZagreb Synagogue\n\nZagreb Synagogue was the main synagogue in Zagreb (Croatia).\n\n\
            Notes on things\n\nNotes on things is short.").unwrap();
        let f = facts(&s, 100);
        assert_eq!(f, vec![
            ("Antimony".into(), "is".into(), "a chemical element with the symbol Sb".into()),
            ("Zagreb Synagogue".into(), "was".into(), "the main synagogue in Zagreb".into()),
        ]);
        let tmp2 = tempfile::tempdir().unwrap();
        let s2 = testing::settings(tmp2.path());
        std::fs::create_dir_all(s2.data_path("brain")).unwrap();
        assert!(facts(&s2, 10).is_empty());
    }
}
