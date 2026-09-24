//! Training speed for every tier in config/settings.yaml on this machine:
//!     cargo run --release --example bench
use council::config::Settings;
use council::model::{Acts, Model, ModelConfig};
use council::rng::Rng;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let settings = Settings::load()?;
    let mut tiers: Vec<_> = settings.model.tiers.iter().collect();
    tiers.sort_by(|a, b| a.1.max_ram_gb.total_cmp(&b.1.max_ram_gb));
    println!("{} threads", rayon::current_num_threads());
    for (name, t) in tiers {
        let cfg = ModelConfig {
            vocab_size: t.vocab_size, block_size: t.block_size, n_layer: t.n_layer, n_head: t.n_head,
            d_model: t.d_model, mlp_hidden: ModelConfig::default_mlp_hidden(t.d_model),
            dropout: settings.model.dropout, rope_base: settings.model.rope_base,
        };
        let b = (settings.training.tokens_per_step / t.block_size).max(1);
        let model = Model::new(cfg.clone(), 1);
        let mut acts = Acts::new(&cfg, b, t.block_size);
        let mut grads = vec![0f32; model.num_params()];
        let mut rng = Rng::new(0);
        let toks: Vec<u32> = (0..b * t.block_size).map(|_| rng.below(t.vocab_size) as u32).collect();
        model.forward_backward(&mut acts, &mut grads, &toks, &toks, Some(1)); // warm-up
        let steps = 4;
        let start = Instant::now();
        for i in 0..steps {
            model.forward_backward(&mut acts, &mut grads, &toks, &toks, Some(i));
        }
        let per_step = start.elapsed().as_secs_f64() / steps as f64;
        let tps = (b * t.block_size) as f64 / per_step;
        println!("{name:7} {}x{:<4} {:6.2}M params  {:>7.0} tokens/s  {:5.1}M tokens/hour",
                 t.n_layer, t.d_model, model.num_params() as f64 / 1e6, tps, tps * 3600.0 / 1e6);
    }
    Ok(())
}
