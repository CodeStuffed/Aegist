//! How well the negation test tells true from false on YOUR trained model:
//!     cargo run --release --example negation_eval
//! Builds claim pairs from the first sentence of articles in the corpus
//! ("<Title> is a ...", "<Title> was a ..."): the true one, and the same
//! subject with another article's description. A pair counts as right when
//! the true claim gets the higher negation-test score. Reports the raw
//! score and the calibrated one, with and without knowledge-base evidence.
use council::brain::Brain;
use council::config::Settings;
use council::corpus;
use council::council::{context_budget_chars, evidence_then, negate, negation_test};
use council::knowledge::KnowledgeStore;
use council::rng::Rng;

fn main() -> anyhow::Result<()> {
    rayon::ThreadPoolBuilder::new().build()?.install(run)
}

/// (subject, verb, description) from "Title is a description, ..." sentences.
fn facts(settings: &Settings) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for f in corpus::corpus_files(settings) {
        let text = std::fs::read_to_string(&f).unwrap_or_default();
        let paras: Vec<&str> = text.split("\n\n").collect();
        for w in paras.windows(2) {
            let (title, para) = (w[0].trim(), w[1]);
            if title.contains('.') || title.len() > 60 || title.split_whitespace().count() > 6 {
                continue;
            }
            for verb in ["is", "was"] {
                let start = format!("{title} {verb} ");
                let Some(rest) = para.strip_prefix(&start) else { continue };
                if !(rest.starts_with("a ") || rest.starts_with("an ") || rest.starts_with("the ")) {
                    continue;
                }
                let desc: Vec<&str> = rest.split(|c| c == ',' || c == '.' || c == ';' || c == '(').next().unwrap_or("").split_whitespace().take(8).collect();
                if desc.len() >= 3 {
                    out.push((title.to_string(), verb.to_string(), desc.join(" ")));
                }
            }
        }
    }
    out
}

fn run() -> anyhow::Result<()> {
    let settings = Settings::load()?;
    let brain = Brain::load(&settings)?;
    let kb = KnowledgeStore::open(&settings)?;
    let facts = facts(&settings);
    anyhow::ensure!(facts.len() >= 10, "found only {} usable first sentences", facts.len());
    let mut rng = Rng::new(7);
    // [raw, calibrated] x [no evidence, evidence]
    let mut right = [[0usize; 2]; 2];
    let mut n = 0;
    for (i, (subject, verb, desc)) in facts.iter().enumerate() {
        // another article's description, with the same verb
        let j = (0..20).map(|_| rng.below(facts.len())).find(|&j| j != i && facts[j].1 == *verb && facts[j].2 != *desc);
        let Some(j) = j else { continue };
        let truth = format!("{subject} {verb} {desc}");
        let lie = format!("{subject} {verb} {}", facts[j].2);
        let mut scores = [[[0f32; 2]; 2]; 2]; // [claim][evidence][raw, calibrated]
        for (c, claim) in [&truth, &lie].into_iter().enumerate() {
            let neg = negate(claim).expect("an is/was sentence always has a verb to flip");
            for (e, with) in [false, true].into_iter().enumerate() {
                let hits = if with {
                    kb.search_linked(claim, settings.memory.top_k, settings.memory.min_relevance, context_budget_chars(&brain, &settings))
                } else {
                    Vec::new()
                };
                let refs: Vec<_> = hits.iter().collect();
                let evidence = brain.prefix(&evidence_then("", &refs), 64, None);
                let t = negation_test(&brain, &neg, &evidence, &settings, None);
                scores[c][e] = [t.raw_contrast, t.contrast];
            }
        }
        for e in 0..2 {
            for k in 0..2 {
                right[k][e] += (scores[0][e][k] > scores[1][e][k]) as usize;
            }
        }
        n += 1;
    }
    println!("{n} true/false pairs from {} first sentences; right = the true claim scored higher (chance: 50%)", facts.len());
    for (k, name) in ["raw", "calibrated"].into_iter().enumerate() {
        println!("  {name:10}  without evidence {:5.1}%   with knowledge-base evidence {:5.1}%",
            100.0 * right[k][0] as f64 / n as f64, 100.0 * right[k][1] as f64 / n as f64);
    }
    Ok(())
}
