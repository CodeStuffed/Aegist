//! What quantization costs and buys on YOUR trained model:
//!     cargo run --release --example quant_eval
//! For f32, int8 and int4: weight memory, held-out bits per byte (how well
//! it still predicts code it never trained on), and generation speed.
use aegist::brain::Brain;
use aegist::config::Settings;
use aegist::corpus;
use aegist::model::KvCache;
use aegist::quant::Precision;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Run inside the thread pool: parallel calls made from a pool thread are cheap.
    rayon::ThreadPoolBuilder::new().build()?.install(run)
}

fn run() -> anyhow::Result<()> {
    let mut settings = Settings::load()?;
    // Held-out text: the tail of the corpus, which training never sees.
    let chunks: Vec<String> = {
        let brain = Brain::load(&settings)?;
        let fim = aegist::trainer::fim_for(&brain.model.cfg, &settings);
        let tokens = corpus::corpus_tokens(&brain.tokenizer, &settings, fim)?;
        let n_val = ((tokens.len() as f64 * settings.training.val_fraction) as usize).max(brain.model.cfg.block_size + 2);
        let val: Vec<u32> = tokens[tokens.len() - n_val..].iter().map(|&t| t as u32).collect();
        val.chunks(brain.model.cfg.block_size - 2).take(40).map(|c| brain.tokenizer.decode(c)).collect()
    };
    println!("{:6} {:>10} {:>16} {:>18}", "", "weights", "held-out bits/B", "generation tok/s");
    for precision in [Precision::F32, Precision::Int8, Precision::Int4] {
        settings.inference.precision = precision;
        let brain = Brain::load(&settings)?;
        let (mut bits, mut bytes) = (0f64, 0usize);
        for text in &chunks {
            let n_tokens = brain.tokenizer.encode(text).len();
            bits += brain.text_nll(text) as f64 * n_tokens as f64 / std::f64::consts::LN_2;
            bytes += text.len();
        }
        // generation: prefill a prompt, then 64 single-token steps with the cache
        let m = &brain.model;
        let c = m.cfg.d_model;
        let prompt: Vec<u32> = brain.tokenizer.encode(&chunks[0]).into_iter().take(32).collect();
        let pre = m.infer(&KvCache::empty(&m.cfg), &prompt, 1, prompt.len(), None);
        let mut cache = KvCache::empty(&m.cfg);
        for r in 0..prompt.len() {
            cache.push_row(&pre, r, c);
        }
        let steps = 64.min(m.cfg.block_size - prompt.len() - 1);
        let start = Instant::now();
        for i in 0..steps {
            let out = m.infer(&cache, &[prompt[i % prompt.len()]], 1, 1, None);
            cache.push_row(&out, 0, c);
        }
        let tps = steps as f64 / start.elapsed().as_secs_f64();
        println!("{:6} {:>8.1}MB {:>16.4} {:>18.0}", precision.to_string(), m.weight_bytes() as f64 / 1e6, bits / bytes as f64, tps);
    }
    Ok(())
}
