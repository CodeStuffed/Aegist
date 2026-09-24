//! The training loop: corpus -> tokens -> batches -> AdamW updates, with
//! held-out evaluation, checkpoints, and a clean stop on Ctrl-C. The first
//! run learns a vocabulary and creates a model sized for this machine;
//! every later run continues from the checkpoint.

use crate::checkpoint::{self, HistoryPoint, Meta, Stats};
use crate::config::Settings;
use crate::corpus;
use crate::hardware;
use crate::model::{Acts, Model, ModelConfig};
use crate::optim::{clip_grad_norm, AdamW};
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::util::iso_now;
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Training can't start yet because there isn't enough text.
#[derive(Debug)]
pub struct NotEnoughText(pub String);

impl std::fmt::Display for NotEnoughText {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NotEnoughText {}

#[derive(Clone, Copy, Debug)]
pub enum Budget {
    Minutes(f64),
    Steps(u64),
}

#[derive(Debug)]
pub struct Report {
    pub steps_this_session: u64,
    pub tokens_this_session: u64,
    pub interrupted: bool,
    pub stats: Stats,
}

pub fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Tokens of training for a model this size to be trained well (~20 per parameter).
pub const TOKENS_PER_PARAM: f64 = 20.0;

pub fn human_duration(seconds: f64) -> String {
    let (m, h, d, y) = (60.0, 3600.0, 86_400.0, 365.25 * 86_400.0);
    match seconds {
        s if s < 90.0 => "1 minute".to_string(),
        s if s < h => format!("{:.0} minutes", s / m),
        s if s < 2.0 * d => format!("{:.1} hours", s / h),
        s if s < 2.0 * y => format!("{:.0} days", s / d),
        s => format!("{} years", commas((s / y).round() as u64)),
    }
}

fn config_for(settings: &Settings, tier: &crate::config::Tier, vocab_size: usize) -> ModelConfig {
    ModelConfig {
        vocab_size,
        block_size: tier.block_size,
        n_layer: tier.n_layer,
        n_head: tier.n_head,
        d_model: tier.d_model,
        mlp_hidden: ModelConfig::default_mlp_hidden(tier.d_model),
        dropout: settings.model.dropout,
        rope_base: settings.model.rope_base,
    }
}

/// Weights + gradients + two AdamW moments, plus activations for one step.
pub fn training_bytes(cfg: &ModelConfig, settings: &Settings) -> usize {
    let b = (settings.training.tokens_per_step / cfg.block_size).max(1);
    16 * crate::model::Layout::new(cfg).total + Acts::estimate_bytes(cfg, b, cfg.block_size)
}

fn check_memory(cfg: &ModelConfig, settings: &Settings) -> Result<()> {
    let need = training_bytes(cfg, settings) as f64 / (1u64 << 30) as f64;
    let have = hardware::detect().ram_gb;
    if need > 0.85 * have {
        anyhow::bail!(
            "Training this {}x{} model needs about {need:.1} GB of memory, but this machine has {have:.1} GB. \
             Pick a smaller size (`council train --tier small`) or lower training.tokens_per_step.",
            cfg.n_layer, cfg.d_model);
    }
    Ok(())
}

fn new_model(settings: &Settings, seed: u64, tier: Option<&str>, log: &mut dyn FnMut(String)) -> Result<(Model, Tokenizer, Meta)> {
    let text = corpus::read_corpus(settings, settings.training.tokenizer_train_chars);
    let needed = settings.training.min_new_model_chars;
    if text.len() < needed {
        return Err(NotEnoughText(format!(
            "Only {} KB of text so far; a new model needs at least {} KB to learn its vocabulary from.",
            text.len() / 1000, needed / 1000)).into());
    }
    let hw = hardware::detect();
    let tier_name = match tier {
        Some(name) => name.to_string(),
        None => hardware::pick_tier(hw.ram_gb, &settings.model.tiers),
    };
    let tier = settings.model.tiers.get(&tier_name).ok_or_else(|| {
        anyhow::anyhow!("No tier named {tier_name:?}; settings.yaml has: {}",
            settings.model.tiers.keys().cloned().collect::<Vec<_>>().join(", "))
    })?;
    // check with the full vocabulary before spending time learning it
    let planned = config_for(settings, tier, tier.vocab_size);
    planned.validate()?;
    check_memory(&planned, settings)?;
    log(format!("Creating a new model for the '{tier_name}' tier ({:.1} GB RAM).", hw.ram_gb));
    log(format!("Learning a {}-token vocabulary from the corpus...", tier.vocab_size));
    let tokenizer = Tokenizer::train(&text, tier.vocab_size);
    let cfg = config_for(settings, tier, tokenizer.vocab_size());
    let model = Model::new(cfg.clone(), seed);
    log(format!(
        "Model: {} layers x {} wide, {} heads, {}-token context, {:.2}M parameters.",
        cfg.n_layer, cfg.d_model, cfg.n_head, cfg.block_size, model.num_params() as f64 / 1e6
    ));
    let now = iso_now();
    let meta = Meta {
        format: checkpoint::FORMAT.to_string(),
        config: cfg,
        merges: tokenizer.merges.clone(),
        stats: Stats { created_at: now.clone(), updated_at: now, ..Default::default() },
    };
    Ok((model, tokenizer, meta))
}

/// Train, or keep training, for the given budget. Saves as it goes; on a
/// stop request it saves and returns with `interrupted` set.
/// `tier`: size for a NEW model by name (default: picked from RAM). An
/// existing model keeps its size.
pub fn train(settings: &Settings, budget: Budget, tier: Option<&str>, log: &mut dyn FnMut(String), seed: Option<u64>, stop: &AtomicBool) -> Result<Report> {
    let tcfg = &settings.training;
    if corpus::corpus_files(settings).is_empty() {
        return Err(NotEnoughText("The corpus is empty. Add text with `council train --data <folder>`, \
                                  or let `council research` collect some.".into()).into());
    }
    let seed = seed.unwrap_or_else(|| Rng::from_time().next_u64());
    let (mut model, tokenizer, mut meta) = match checkpoint::load(settings)? {
        Some(loaded) => {
            if let Some(name) = tier {
                let want = settings.model.tiers.get(name);
                let c = &loaded.0.cfg;
                if want.map_or(true, |t| (t.n_layer, t.d_model, t.block_size) != (c.n_layer, c.d_model, c.block_size)) {
                    anyhow::bail!("A {}x{} model already exists in {}. Delete that folder to start a new one at the {name:?} size \
                                   (your text and knowledge base are kept).", c.n_layer, c.d_model, checkpoint::brain_dir(settings).display());
                }
            }
            check_memory(&loaded.0.cfg, settings)?;
            loaded
        }
        None => new_model(settings, seed, tier, log)?,
    };
    let cfg = model.cfg.clone();
    let tokens = corpus::corpus_tokens(&tokenizer, settings)?;
    let t = cfg.block_size;
    let b = (tcfg.tokens_per_step / t).max(1);
    if tokens.len() < tcfg.min_corpus_tokens.max(4 * (t + 2)) {
        return Err(NotEnoughText(format!(
            "The corpus is only {} tokens; add more text first (need at least {}).",
            commas(tokens.len() as u64), commas(tcfg.min_corpus_tokens as u64))).into());
    }
    let n_val = ((tokens.len() as f64 * tcfg.val_fraction) as usize).max(t + 2);
    let (train_tok, val_tok) = tokens.split_at(tokens.len() - n_val);
    let val_bytes: usize = val_tok.iter().map(|&i| tokenizer.token_len(i as u32)).sum();
    let bits_per_byte = |loss: f32| loss / std::f32::consts::LN_2 * (val_tok.len() as f32 / val_bytes.max(1) as f32);
    log(format!("Corpus: {} tokens ({} train / {} held out).",
        commas(tokens.len() as u64), commas(train_tok.len() as u64), commas(val_tok.len() as u64)));

    let mut acts = Acts::new(&cfg, b, t);
    let mut grads = vec![0f32; model.num_params()];
    let mut opt = AdamW::new(model.num_params(), &model.layout.matrices, tcfg.weight_decay);
    checkpoint::load_optimizer(settings, &mut opt);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let (mut x, mut y) = (vec![0u32; b * t], vec![0u32; b * t]);

    let fill = |src: &[u16], rng: &mut Rng, x: &mut [u32], y: &mut [u32]| {
        for row in 0..b {
            let s = rng.below(src.len() - t - 1);
            for i in 0..t {
                x[row * t + i] = src[s + i] as u32;
                y[row * t + i] = src[s + i + 1] as u32;
            }
        }
    };
    let evaluate = |model: &Model, acts: &mut Acts, x: &mut [u32], y: &mut [u32]| -> f32 {
        let mut r = Rng::new(1234); // the same held-out batches every time
        let total: f32 = (0..tcfg.eval_batches).map(|_| {
            fill(val_tok, &mut r, x, y);
            model.loss(acts, x, y)
        }).sum();
        total / tcfg.eval_batches as f32
    };

    let start = Instant::now();
    let (mut last_log, mut last_eval, mut last_save) = (0.0f64, 0.0f64, 0.0f64);
    let mut step_i = 0u64;
    let mut interrupted = false;
    let lr_max = tcfg.learning_rate;
    loop {
        let elapsed = start.elapsed().as_secs_f64();
        match budget {
            Budget::Minutes(m) if elapsed >= m * 60.0 => break,
            Budget::Steps(n) if step_i >= n => break,
            _ => {}
        }
        if stop.load(Ordering::Relaxed) {
            interrupted = true;
            break;
        }
        fill(train_tok, &mut rng, &mut x, &mut y);
        grads.iter_mut().for_each(|g| *g = 0.0);
        let loss = model.forward_backward(&mut acts, &mut grads, &x, &y, Some(rng.next_u64()));
        clip_grad_norm(&mut grads, tcfg.grad_clip);
        let warm = ((meta.stats.steps + 1) as f32 / tcfg.warmup_steps.max(1) as f32).min(1.0);
        let frac = match budget {
            Budget::Minutes(m) => (elapsed / (m * 60.0)).min(1.0) as f32,
            Budget::Steps(n) => step_i as f32 / n.max(1) as f32,
        };
        let floor = tcfg.min_lr_fraction;
        let lr = lr_max * warm * (floor + (1.0 - floor) * 0.5 * (1.0 + (std::f32::consts::PI * frac).cos()));
        opt.step(&mut model.params, &grads, lr);

        step_i += 1;
        let st = &mut meta.stats;
        st.steps += 1;
        st.tokens_seen += (b * t) as u64;
        st.train_loss = Some(match st.train_loss { Some(ema) => 0.95 * ema + 0.05 * loss, None => loss });

        let now = start.elapsed().as_secs_f64();
        if now - last_eval >= tcfg.eval_every_s {
            let v = evaluate(&model, &mut acts, &mut x, &mut y);
            meta.stats.val_loss = Some(v);
            meta.stats.val_bpb = Some(bits_per_byte(v));
            last_eval = now;
        }
        if now - last_log >= tcfg.log_every_s {
            let rate = (step_i * (b * t) as u64) as f64 / now.max(1e-9);
            let held = meta.stats.val_loss.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into());
            log(format!("step {:>6} | loss {:.3} | held-out {held} | {:>7} tok/s | lr {lr:.1e}",
                meta.stats.steps, meta.stats.train_loss.unwrap_or(f32::NAN), commas(rate as u64)));
            if last_log == 0.0 {
                // once per session: how long this model needs, at this machine's real speed
                let target = TOKENS_PER_PARAM * model.num_params() as f64;
                let left = target - meta.stats.tokens_seen as f64;
                log(if left > 0.0 {
                    format!("At {} tokens/s, this {:.1}M-parameter model needs about {} more training to read ~{:.0}M tokens \
                             (20 per parameter, roughly \"trained well\").",
                        commas(rate as u64), model.num_params() as f64 / 1e6, human_duration(left / rate.max(1.0)), target / 1e6)
                } else {
                    "This model has already read ~20 tokens per parameter; more new text now helps more than more time.".to_string()
                });
            }
            last_log = now;
        }
        if now - last_save >= tcfg.checkpoint_every_s {
            meta.stats.train_seconds += now - last_save;
            last_save = now;
            save_with_eval(settings, &model, &mut meta, &opt, &mut acts, &mut x, &mut y, &evaluate, &bits_per_byte)?;
        }
    }
    meta.stats.train_seconds += start.elapsed().as_secs_f64() - last_save;
    save_with_eval(settings, &model, &mut meta, &opt, &mut acts, &mut x, &mut y, &evaluate, &bits_per_byte)?;
    let st = &meta.stats;
    log(format!("Saved: {} steps total, {:.1}M tokens seen, held-out loss {:.3} ({:.3} bits per byte).",
        commas(st.steps), st.tokens_seen as f64 / 1e6, st.val_loss.unwrap_or(f32::NAN), st.val_bpb.unwrap_or(f32::NAN)));
    Ok(Report { steps_this_session: step_i, tokens_this_session: step_i * (b * t) as u64, interrupted, stats: meta.stats })
}

#[allow(clippy::too_many_arguments)]
fn save_with_eval(
    settings: &Settings, model: &Model, meta: &mut Meta, opt: &AdamW, acts: &mut Acts, x: &mut [u32], y: &mut [u32],
    evaluate: &dyn Fn(&Model, &mut Acts, &mut [u32], &mut [u32]) -> f32, bpb: &dyn Fn(f32) -> f32,
) -> Result<()> {
    let v = evaluate(model, acts, x, y);
    let st = &mut meta.stats;
    st.val_loss = Some(v);
    st.val_bpb = Some(bpb(v));
    st.history.push(HistoryPoint { step: st.steps, val_loss: v, val_bpb: bpb(v), at: iso_now() });
    if st.history.len() > 500 {
        st.history.remove(0);
    }
    checkpoint::save(settings, model, meta, Some(opt))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::config::testing;

    pub const CORPUS: &str = "Light is refracted when it passes from air into glass. The rays of light bend toward the \
perpendicular. This is true. White light is made of many colours, and a prism separates them.\n\n\
The market for a product depends on customers who will pay for it. A business that charges more than its costs \
is profitable. It will make money when customers pay for it.\n\n\
Heavy objects do not fall faster than light objects in a vacuum. This is false. That is not right, because \
gravity accelerates every body at the same rate.\n\n";

    /// A data dir with CORPUS imported and a model trained for `steps`.
    pub fn trained(steps: u64) -> (tempfile::TempDir, Settings) {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(&tmp.path().join("data"));
        let src = tmp.path().join("notes.txt");
        std::fs::write(&src, CORPUS.repeat(20)).unwrap();
        corpus::import_texts(&src, &s).unwrap();
        train(&s, Budget::Steps(steps), None, &mut |_| {}, Some(0), &AtomicBool::new(false)).unwrap();
        (tmp, s)
    }

    #[test]
    fn empty_and_tiny_corpora_are_clear_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(&tmp.path().join("data"));
        let stop = AtomicBool::new(false);
        let err = train(&s, Budget::Steps(1), None, &mut |_| {}, None, &stop).unwrap_err();
        assert!(err.downcast_ref::<NotEnoughText>().unwrap().0.contains("corpus is empty"));
        std::fs::write(tmp.path().join("t.txt"), "too short").unwrap();
        corpus::import_texts(&tmp.path().join("t.txt"), &s).unwrap();
        let err = train(&s, Budget::Steps(1), None, &mut |_| {}, None, &stop).unwrap_err();
        assert!(err.to_string().contains("KB of text so far"));
        s.training.min_new_model_chars = 1;
        let err = train(&s, Budget::Steps(1), None, &mut |_| {}, None, &stop).unwrap_err();
        assert!(err.to_string().contains("only"));
    }

    #[test]
    fn learns_resumes_and_stops_cleanly() {
        let (_tmp, s) = trained(40);
        let meta = checkpoint::load(&s).unwrap().unwrap().2;
        let vocab = meta.config.vocab_size as f32;
        assert!(meta.stats.val_loss.unwrap() < vocab.ln() - 1.0, "held-out loss {:?}", meta.stats.val_loss);
        assert!(meta.stats.val_bpb.unwrap() > 0.0);
        let again = train(&s, Budget::Steps(10), None, &mut |_| {}, Some(1), &AtomicBool::new(false)).unwrap();
        assert_eq!((again.stats.steps, again.steps_this_session), (50, 10));
        assert!(checkpoint::brain_dir(&s).join("optim.bin").is_file());
        let stopped = train(&s, Budget::Steps(1000), None, &mut |_| {}, Some(2), &AtomicBool::new(true)).unwrap();
        assert!(stopped.interrupted && stopped.steps_this_session == 0 && stopped.stats.steps == 50);
    }

    #[test]
    fn old_python_checkpoints_are_rejected_with_advice() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(&tmp.path().join("data"));
        std::fs::write(checkpoint::brain_dir(&s).join("brain.json"), r#"{"config": {}, "merges": [], "stats": {}}"#).unwrap();
        let err = checkpoint::load(&s).err().unwrap().to_string();
        assert!(err.contains("older version") && err.contains("Delete"));
    }

    #[test]
    fn tier_choice_memory_check_and_durations() {
        let (_tmp, mut s) = trained(5);
        // an existing model keeps its size
        s.model.tiers.insert("big".into(), crate::config::Tier { max_ram_gb: 1e9, n_layer: 2, n_head: 2, d_model: 64, block_size: 48, vocab_size: 320, manual: true });
        let err = train(&s, Budget::Steps(1), Some("big"), &mut |_| {}, None, &AtomicBool::new(false)).unwrap_err();
        assert!(err.to_string().contains("already exists"));
        // a model that can't fit in memory is refused before anything is allocated
        let huge = crate::config::Tier { max_ram_gb: 1e9, n_layer: 96, n_head: 96, d_model: 12288, block_size: 2048, vocab_size: 50000, manual: true };
        let cfg = config_for(&s, &huge, 50000);
        assert!(training_bytes(&cfg, &s) > 1 << 40); // a GPT-3-sized model needs terabytes
        assert!(check_memory(&cfg, &s).unwrap_err().to_string().contains("needs about"));
        assert_eq!(human_duration(90.0), "2 minutes");
        assert_eq!(human_duration(3.0 * 3600.0), "3.0 hours");
        assert_eq!(human_duration(6e8), "19 years");
    }

    #[test]
    fn commas_format() {
        assert_eq!((commas(0), commas(999), commas(1000), commas(1234567)),
                   ("0".into(), "999".into(), "1,000".into(), "1,234,567".into()));
    }
}
