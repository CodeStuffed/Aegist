//! How the knowledge base scales, on made-up text:
//!     cargo run --release --example kb_bench -- 1000000
//! Imports N passages (default 1,000,000) into a scratch folder, builds the
//! on-disk index, then times opening it and searching it.
use council::config::Settings;
use council::knowledge::KnowledgeStore;
use council::rng::Rng;
use serde_json::{Map, Value};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(1_000_000);
    let tmp = tempfile::tempdir()?;
    let mut settings = Settings::load()?;
    settings.data_dir = tmp.path().to_path_buf();
    // Zipf-like vocabulary: word k is picked with probability ~ 1/k
    let vocab = 300_000f64;
    let word = |rng: &mut Rng| format!("w{}", (vocab.ln() * rng.uniform() as f64).exp() as usize);

    let start = Instant::now();
    let mut kb = KnowledgeStore::open(&settings)?;
    let mut w = kb.bulk()?;
    let mut rng = Rng::new(1);
    for i in 0..n {
        let len = 40 + rng.below(80);
        let text: Vec<String> = (0..len).map(|_| word(&mut rng)).collect();
        let mut meta = Map::new();
        meta.insert("title".into(), Value::from(format!("Article {}", i / 6)));
        w.add(&text.join(" "), meta)?;
    }
    w.finish()?;
    let bytes = std::fs::metadata(KnowledgeStore::path_for(&settings))?.len();
    println!("wrote {n} passages ({:.0} MB) in {:.1}s", bytes as f64 / 1e6, start.elapsed().as_secs_f64());

    let start = Instant::now();
    kb.commit()?;
    let (_, _, _, index_bytes) = kb.stats();
    println!("built the index ({:.0} MB) in {:.1}s", index_bytes as f64 / 1e6, start.elapsed().as_secs_f64());

    let start = Instant::now();
    let kb = KnowledgeStore::open(&settings)?;
    println!("opened {} passages in {:.1} ms", kb.count(), start.elapsed().as_secs_f64() * 1e3);

    let mut rng = Rng::new(9);
    let queries: Vec<String> = (0..50).map(|_| (0..8).map(|_| word(&mut rng)).collect::<Vec<_>>().join(" ")).collect();
    for (name, linked) in [("search", false), ("search_linked", true)] {
        let start = Instant::now();
        let mut found = 0;
        for q in &queries {
            found += if linked { kb.search_linked(q, 3, 0.2, 3000).len() } else { kb.search(q, 3, 0.2).len() };
        }
        println!("{name:14} {:.1} ms per query ({found} passages found for {} queries)",
                 start.elapsed().as_secs_f64() * 1e3 / queries.len() as f64, queries.len());
    }
    Ok(())
}
