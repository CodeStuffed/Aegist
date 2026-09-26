//! The training loop: corpus -> tokens -> batches -> AdamW updates, with
//! held-out evaluation, checkpoints, and a clean stop on Ctrl-C. The first
//! run learns a vocabulary and creates a model sized for this machine;
//! every later run continues from the checkpoint.

use crate::checkpoint::{self, HistoryPoint, Meta, Stats};
use crate::config::{Device, GpuPrecision, Settings};
use crate::gpu::{self, cuda::CudaBackend, Backend, GpuTrainer};
use crate::corpus;
use crate::hardware;
use crate::model::{Acts, Model, ModelConfig};
use crate::optim::{clip_grad_norm, AdamW};
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::util::iso_now;
use anyhow::{Context, Result};
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

/// How to size a NEW model (an existing one keeps its size).
#[derive(Clone, Copy, Debug)]
pub enum Size<'a> {
    /// A tier from settings.yaml, by name.
    Tier(&'a str),
    /// Whatever will be smartest after this many hours of training in total.
    ForHours(f64),
    /// The automatic tier for this machine's RAM.
    FromRam,
}

/// Training cost of one token, forward and backward: ~6 operations per
/// weight, plus attention across the context.
pub fn flops_per_token(cfg: &ModelConfig) -> f64 {
    6.0 * crate::model::Layout::new(cfg).total as f64 + 12.0 * (cfg.n_layer * cfg.block_size * cfg.d_model) as f64
}

/// Useful training FLOP/s on this machine's CPU: a matrix multiply timed on
/// every core, times ~0.6 (about what the whole training loop reaches).
pub fn measure_cpu_flops() -> f64 {
    let measure = || {
        let n = 768;
        let (a, b) = (vec![0.5f32; n * n], vec![0.25f32; n * n]);
        let mut c = vec![0f32; n * n];
        crate::kernels::matmul(&mut c, &a, &b, n, n, n, false, false, false);
        // best of three, so a busy moment doesn't change the answer
        let best = (0..3)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..8 {
                    crate::kernels::matmul(&mut c, &a, &b, n, n, n, false, false, false);
                }
                start.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min);
        0.6 * 8.0 * 2.0 * (n * n * n) as f64 / best
    };
    rayon::ThreadPoolBuilder::new().build().map(|p| p.install(measure)).unwrap_or_else(|_| measure())
}

/// Where training runs.
pub enum Compute {
    Cpu { flops: f64, ram_gb: f64 },
    Gpu { backend: Box<dyn Backend>, flops: f64, bf16: bool },
}

/// The share of a GPU's measured TF32 matrix-multiply speed that training
/// reaches end to end (attention's small matrices, the other kernels,
/// memory traffic). A rough figure: the real rate is shown once training runs.
pub const GPU_EFFICIENCY: f64 = 0.5;

impl Compute {
    pub fn cpu() -> Compute {
        Compute::Cpu { flops: measure_cpu_flops(), ram_gb: hardware::detect().ram_gb }
    }

    /// Useful training FLOP/s.
    pub fn training_flops(&self) -> f64 {
        match self {
            Compute::Cpu { flops, .. } => *flops,
            Compute::Gpu { flops, .. } => *flops,
        }
    }

    /// Operations per token trained (the GPU recomputes each layer's
    /// forward pass during the backward pass: a third more).
    pub fn flops_per_token(&self, cfg: &ModelConfig) -> f64 {
        match self {
            Compute::Cpu { .. } => flops_per_token(cfg),
            Compute::Gpu { .. } => flops_per_token(cfg) * 4.0 / 3.0,
        }
    }

    /// Whether a model this size can train here.
    pub fn fits(&self, cfg: &ModelConfig, settings: &Settings) -> bool {
        self.check_fits(cfg, settings).is_ok()
    }

    fn check_fits(&self, cfg: &ModelConfig, settings: &Settings) -> Result<()> {
        match self {
            Compute::Cpu { ram_gb, .. } => check_memory_in(cfg, settings, *ram_gb),
            Compute::Gpu { backend, bf16, .. } => {
                // the CPU keeps a copy of the weights and optimizer state for checkpoints
                let host = 12.0 * crate::model::Layout::new(cfg).total as f64 / (1u64 << 30) as f64;
                let ram = hardware::detect().ram_gb;
                if host > 0.85 * ram {
                    anyhow::bail!("Training this {}x{} model on the GPU needs about {host:.1} GB of RAM for its checkpoints, \
                                   but this machine has {ram:.1} GB. Pick a smaller size with --tier.", cfg.n_layer, cfg.d_model);
                }
                let (free, _) = backend.memory()?;
                if gpu::micro_batch_for(cfg, free, 1, *bf16).is_none() {
                    anyhow::bail!("This {}x{} model needs about {:.1} GB of GPU memory to train, but {:.1} GB is free. \
                                   Pick a smaller size with --tier, or train on the CPU with --device cpu.",
                        cfg.n_layer, cfg.d_model, (gpu::training_bytes(cfg, 1, *bf16) + gpu::OVERHEAD_BYTES) as f64 / (1u64 << 30) as f64,
                        free as f64 / (1u64 << 30) as f64);
                }
                Ok(())
            }
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Compute::Cpu { .. } => "the CPU".into(),
            Compute::Gpu { backend, .. } => backend.describe(),
        }
    }
}

/// Open the NVIDIA GPU, compile the kernels, and check its results against
/// the CPU's before trusting it with training.
pub fn open_checked_gpu(settings: &Settings, log: &mut dyn FnMut(String)) -> Result<Compute> {
    let backend = CudaBackend::open()?;
    let bf16 = match settings.gpu.precision {
        GpuPrecision::Auto => backend.supports_bf16(),
        GpuPrecision::Tf32 => false,
        GpuPrecision::Bf16 if backend.supports_bf16() => true,
        GpuPrecision::Bf16 => anyhow::bail!("gpu.precision is bf16, but this GPU has no bf16 tensor cores (RTX 30xx or newer); use tf32"),
    };
    let r = gpu::self_check(&backend)?;
    if !r.passed {
        anyhow::bail!("the GPU's results don't match the CPU's (loss {} vs {}, worst gradient error {:.1e} in {}, weights after a step {:.1e}). \
                       Update the NVIDIA driver and CUDA Toolkit, then run `aegist gpu-check`.",
            r.loss_gpu, r.loss_cpu, r.worst_grad.1, r.worst_grad.0, r.step_error);
    }
    let flops = gpu::gemm_flops(&backend, bf16)? * GPU_EFFICIENCY;
    let b16 = r.bf16_grad.as_ref().map_or(String::new(), |b| format!("; {:.0e} with bf16", b.1));
    log(format!("GPU: {} - self-check passed (every gradient matches the CPU's to {:.0e}{b16}); matrix multiplies in {}.",
        backend.describe(), r.worst_grad.1.max(1e-9), if bf16 { "bf16" } else { "tf32" }));
    Ok(Compute::Gpu { backend: Box::new(backend), flops, bf16 })
}

/// Where to train, from settings.gpu.device (auto: the GPU if one works).
pub fn choose_compute(settings: &Settings, log: &mut dyn FnMut(String)) -> Result<Compute> {
    match settings.gpu.device {
        Device::Cpu => Ok(Compute::cpu()),
        Device::Gpu => open_checked_gpu(settings, log).context("training on the GPU was asked for (--device gpu)"),
        Device::Auto => match open_checked_gpu(settings, log) {
            Ok(c) => Ok(c),
            Err(e) => {
                log(format!("Training on the CPU ({e}).", e = e.to_string().trim_end_matches('.')));
                Ok(Compute::cpu())
            }
        },
    }
}

/// Repeating text more than about this many times stops helping.
pub const MAX_EPOCHS: f64 = 4.0;

/// The size that ends up smartest after `hours` of training at `flops_per_s`:
/// the biggest tier that fits in memory and can still read ~20 tokens per
/// parameter in that time (bigger would be cut off half-trained; smaller
/// would stop improving early) without repeating the corpus more than
/// MAX_EPOCHS times. Returns the tier's name and why.
pub fn pick_for_hours(settings: &Settings, hours: f64, compute: &Compute, corpus_tokens: f64) -> (String, String) {
    let flops_per_s = compute.training_flops();
    let mut sized: Vec<(&String, f64, f64)> = settings
        .model
        .tiers
        .iter()
        .map(|(name, t)| {
            let cfg = config_for(settings, t, t.vocab_size);
            let tps = if compute.fits(&cfg, settings) { flops_per_s / compute.flops_per_token(&cfg) } else { 0.0 };
            (name, crate::model::Layout::new(&cfg).total as f64, tps)
        })
        .filter(|&(_, _, tps)| tps > 0.0)
        .collect();
    sized.sort_by(|a, b| a.1.total_cmp(&b.1));
    let Some(&smallest) = sized.first() else {
        return (hardware::pick_tier(hardware::detect().ram_gb, &settings.model.tiers), "nothing fits in memory; using the smallest automatic size".into());
    };
    let readable = |tps: f64| (tps * hours * 3600.0).min(MAX_EPOCHS * corpus_tokens);
    let ok = |&(_, params, tps): &(&String, f64, f64)| readable(tps) >= TOKENS_PER_PARAM * params;
    let pick = sized.iter().rev().find(|t| ok(t)).copied().unwrap_or(smallest);
    let (name, params, tps) = pick;
    let per_param = readable(tps) / params;
    let tokens = |n: f64| if n >= 1e9 { format!("{:.1}B", n / 1e9) } else if n >= 1e7 { format!("{:.0}M", n / 1e6) } else { format!("{:.1}M", n / 1e6) };
    let ratio = |r: f64| if r >= 10.0 { format!("{r:.0}") } else { format!("{r:.1}") };
    let mut why = format!("{name} ({:.1}M parameters): {} h at ~{} tokens/s reads ~{} tokens, {} per parameter",
        params / 1e6, hours, commas(tps as u64), tokens(readable(tps)), ratio(per_param));
    if let Some(&(next, next_params, next_tps)) = sized.iter().find(|t| t.1 > params) {
        let limit = if tps * hours * 3600.0 > MAX_EPOCHS * corpus_tokens { " (the corpus is the limit: add more code)" } else { "" };
        why += &format!("; the next size up ({next}, {:.1}M) would get only {} per parameter - about 20 is needed to train well{limit}",
            next_params / 1e6, ratio(readable(next_tps) / next_params));
    }
    if !ok(&pick) {
        why += ". Even this, the smallest size, won't be trained well in that time";
    }
    (name.clone(), why)
}

/// Weights + gradients + two AdamW moments, plus activations for one step.
pub fn training_bytes(cfg: &ModelConfig, settings: &Settings) -> usize {
    let b = (settings.training.tokens_per_step / cfg.block_size).max(1);
    16 * crate::model::Layout::new(cfg).total + Acts::estimate_bytes(cfg, b, cfg.block_size)
}

fn check_memory_in(cfg: &ModelConfig, settings: &Settings, have: f64) -> Result<()> {
    let need = training_bytes(cfg, settings) as f64 / (1u64 << 30) as f64;
    if need > 0.85 * have {
        anyhow::bail!(
            "Training this {}x{} model needs about {need:.1} GB of memory, but this machine has {have:.1} GB. \
             Pick a smaller size (`aegist train --tier small`) or lower training.tokens_per_step.",
            cfg.n_layer, cfg.d_model);
    }
    Ok(())
}

fn new_model(settings: &Settings, seed: u64, size: Size, compute: &Compute, log: &mut dyn FnMut(String)) -> Result<(Model, Tokenizer, Meta)> {
    let text = corpus::read_corpus(settings, settings.training.tokenizer_train_chars);
    let needed = settings.training.min_new_model_chars;
    if text.len() < needed {
        return Err(NotEnoughText(format!(
            "Only {} KB of code so far; a new model needs at least {} KB to learn its vocabulary from. Add more with `aegist learn`.",
            text.len() / 1000, needed / 1000)).into());
    }
    let hw = hardware::detect();
    let tier_name = match size {
        Size::Tier(name) => name.to_string(),
        Size::FromRam => hardware::pick_tier(hw.ram_gb, &settings.model.tiers),
        Size::ForHours(hours) => {
            let bytes: u64 = corpus::corpus_files(settings).iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
            let (name, why) = pick_for_hours(settings, hours, compute, bytes as f64 / 4.0);
            log(format!("Size for {hours} hour(s) of training on {}: {why}. (Training longer in later sessions? \
                         Start with --plan-hours <total>, or pick one with --tier.)", compute.describe()));
            name
        }
    };
    let tier = settings.model.tiers.get(&tier_name).ok_or_else(|| {
        anyhow::anyhow!("No tier named {tier_name:?}; settings.yaml has: {}",
            settings.model.tiers.keys().cloned().collect::<Vec<_>>().join(", "))
    })?;
    // check with the full vocabulary before spending time learning it
    let planned = config_for(settings, tier, tier.vocab_size);
    planned.validate()?;
    compute.check_fits(&planned, settings)?;
    log(format!("Creating a new model at the '{tier_name}' size."));
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
        stats: Stats {
            created_at: now.clone(),
            updated_at: now,
            planned_seconds: match size {
                Size::ForHours(h) => Some(h * 3600.0),
                _ => None,
            },
            ..Default::default()
        },
    };
    Ok((model, tokenizer, meta))
}

/// Train, or keep training, for the given budget. Saves as it goes; on a
/// stop request it saves and returns with `interrupted` set.
/// `size`: how to size a NEW model; an existing model keeps its size.
pub fn train(settings: &Settings, budget: Budget, size: Size, log: &mut dyn FnMut(String), seed: Option<u64>, stop: &AtomicBool) -> Result<Report> {
    check_corpus(settings)?; // before spending time on the GPU's self-check
    let compute = choose_compute(settings, log)?;
    train_on(settings, budget, size, &compute, log, seed, stop)
}

/// `train`, on the given CPU or GPU.
pub fn train_on(settings: &Settings, budget: Budget, size: Size, compute: &Compute, log: &mut dyn FnMut(String), seed: Option<u64>,
                stop: &AtomicBool) -> Result<Report> {
    let tcfg = &settings.training;
    check_corpus(settings)?;
    let seed = seed.unwrap_or_else(|| Rng::from_time().next_u64());
    let (mut model, tokenizer, mut meta) = match checkpoint::load(settings)? {
        Some(loaded) => {
            if let Size::Tier(name) = size {
                let want = settings.model.tiers.get(name);
                let c = &loaded.0.cfg;
                if want.map_or(true, |t| (t.n_layer, t.d_model, t.block_size) != (c.n_layer, c.d_model, c.block_size)) {
                    anyhow::bail!("A {}x{} model already exists in {}. Delete that folder to start a new one at the {name:?} size \
                                   (your code corpus is kept).", c.n_layer, c.d_model, checkpoint::brain_dir(settings).display());
                }
            }
            compute.check_fits(&loaded.0.cfg, settings)?;
            loaded
        }
        None => new_model(settings, seed, size, compute, log)?,
    };
    let cfg = model.cfg.clone();
    let tokens = corpus::corpus_tokens(&tokenizer, settings, fim_for(&cfg, settings))?;
    let t = cfg.block_size;
    if tokens.len() < tcfg.min_corpus_tokens.max(4 * (t + 2)) {
        return Err(NotEnoughText(format!(
            "The corpus is only {} tokens; add more code first with `aegist learn` (need at least {}).",
            commas(tokens.len() as u64), commas(tcfg.min_corpus_tokens as u64))).into());
    }
    let n_val = ((tokens.len() as f64 * tcfg.val_fraction) as usize).min(tcfg.max_val_tokens).max(t + 2);
    let (train_tok, val_tok) = tokens.split_at(tokens.len() - n_val);
    let val_bytes: usize = val_tok.iter().map(|&i| tokenizer.token_len(i as u32)).sum();
    let bits_per_byte = |loss: f32| loss / std::f32::consts::LN_2 * (val_tok.len() as f32 / val_bytes.max(1) as f32);
    log(format!("Corpus: {} tokens ({} train / {} held out).",
        commas(tokens.len() as u64), commas(train_tok.len() as u64), commas(val_tok.len() as u64)));

    let mut opt = AdamW::new(model.num_params(), &model.layout.matrices, tcfg.weight_decay);
    checkpoint::load_optimizer(settings, &mut opt);
    // b sequences per step; eval_rows per held-out batch
    let (mut engine, b, eval_rows, lr_max, warmup, save_every) = match compute {
        Compute::Cpu { .. } => {
            let b = (tcfg.tokens_per_step / t).max(1);
            let engine = Engine::Cpu { acts: Box::new(Acts::new(&cfg, b, t)), grads: vec![0f32; model.num_params()] };
            (engine, b, b, tcfg.learning_rate, tcfg.warmup_steps, tcfg.checkpoint_every_s)
        }
        Compute::Gpu { backend, bf16, .. } => {
            let (free, _) = backend.memory()?;
            let per_step = (settings.gpu.tokens_per_step / t).max(1);
            let seqs = gpu::micro_batch_for(&cfg, free, per_step.min(64), *bf16).context("the model doesn't fit in the GPU's free memory")?;
            let b = per_step.div_ceil(seqs) * seqs;
            let engine = Engine::Gpu(Box::new(GpuTrainer::new(&**backend, &model, &opt, seqs, *bf16)?));
            log(format!("Training on the GPU: {} tokens per step, in micro-batches of {seqs} sequences.", commas((b * t) as u64)));
            (engine, b, seqs, settings.gpu.learning_rate, settings.gpu.warmup_steps, settings.gpu.checkpoint_every_s)
        }
    };
    let mut rng = Rng::new(seed ^ 0x5eed);
    let (mut x, mut y) = (vec![0u32; b * t], vec![0u32; b * t]);
    let (mut ex, mut ey) = (vec![0u32; eval_rows * t], vec![0u32; eval_rows * t]);
    let evaluate = |engine: &mut Engine, model: &Model, ex: &mut [u32], ey: &mut [u32]| -> Result<f32> {
        let mut r = Rng::new(1234); // the same held-out batches every time
        let mut total = 0.0;
        for _ in 0..tcfg.eval_batches {
            fill(val_tok, &mut r, t, ex, ey);
            total += engine.loss(model, ex, ey)?;
        }
        Ok(total / tcfg.eval_batches as f32)
    };

    let start = Instant::now();
    let (mut last_log, mut last_eval, mut last_save) = (0.0f64, 0.0f64, 0.0f64);
    let mut step_i = 0u64;
    let mut interrupted = false;
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
        fill(train_tok, &mut rng, t, &mut x, &mut y);
        let warm = ((meta.stats.steps + 1) as f32 / warmup.max(1) as f32).min(1.0);
        let trained_before = meta.stats.train_seconds + (elapsed - last_save);
        let frac = schedule_fraction(budget, elapsed, step_i, meta.stats.planned_seconds, trained_before);
        let lr = learning_rate(lr_max, tcfg.min_lr_fraction, warm, frac);
        let loss = engine.step(&mut model, &mut opt, &x, &y, rng.next_u64(), tcfg.grad_clip, lr)?;

        step_i += 1;
        let st = &mut meta.stats;
        st.steps += 1;
        st.tokens_seen += (b * t) as u64;
        st.train_loss = Some(match st.train_loss { Some(ema) => 0.95 * ema + 0.05 * loss, None => loss });

        let now = start.elapsed().as_secs_f64();
        if now - last_eval >= tcfg.eval_every_s {
            let v = evaluate(&mut engine, &model, &mut ex, &mut ey)?;
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
                    "This model has already read ~20 tokens per parameter; more new code now helps more than more time.".to_string()
                });
            }
            last_log = now;
        }
        if now - last_save >= save_every {
            meta.stats.train_seconds += now - last_save;
            last_save = now;
            let v = evaluate(&mut engine, &model, &mut ex, &mut ey)?;
            save(settings, &mut engine, &mut model, &mut opt, &mut meta, v, bits_per_byte(v))?;
        }
    }
    meta.stats.train_seconds += start.elapsed().as_secs_f64() - last_save;
    let v = evaluate(&mut engine, &model, &mut ex, &mut ey)?;
    save(settings, &mut engine, &mut model, &mut opt, &mut meta, v, bits_per_byte(v))?;
    let st = &meta.stats;
    log(format!("Saved: {} steps total, {:.1}M tokens seen, held-out loss {:.3} ({:.3} bits per byte).",
        commas(st.steps), st.tokens_seen as f64 / 1e6, st.val_loss.unwrap_or(f32::NAN), st.val_bpb.unwrap_or(f32::NAN)));
    Ok(Report { steps_this_session: step_i, tokens_this_session: step_i * (b * t) as u64, interrupted, stats: meta.stats })
}

/// How far through its learning-rate schedule training is (0..1): through
/// the model's planned total time if it has one (so several sessions share
/// one schedule), else through this session.
fn schedule_fraction(budget: Budget, elapsed: f64, step_i: u64, planned: Option<f64>, trained_total: f64) -> f32 {
    match (budget, planned) {
        (Budget::Minutes(_), Some(p)) if p > 0.0 => (trained_total / p).clamp(0.0, 1.0) as f32,
        (Budget::Minutes(m), _) => (elapsed / (m * 60.0)).min(1.0) as f32,
        (Budget::Steps(n), _) => step_i as f32 / n.max(1) as f32,
    }
}

/// Warmup, then cosine decay from lr_max down to floor * lr_max.
fn learning_rate(lr_max: f32, floor: f32, warm: f32, frac: f32) -> f32 {
    lr_max * warm * (floor + (1.0 - floor) * 0.5 * (1.0 + (std::f32::consts::PI * frac).cos()))
}

fn check_corpus(settings: &Settings) -> Result<()> {
    if corpus::corpus_files(settings).is_empty() {
        return Err(NotEnoughText("There's no code to learn from yet. Add some with `aegist learn <folder or git URL>`, \
                                  or `aegist learn --pack python` for a ready-made set.".into()).into());
    }
    Ok(())
}

/// How the corpus becomes fill-in-the-middle examples for a model this size:
/// each example fits in its context (~3 characters of code per token).
pub fn fim_for(cfg: &ModelConfig, settings: &Settings) -> corpus::Fim {
    corpus::Fim { rate: settings.code.fim_rate, window_chars: cfg.block_size * 3 }
}

/// `x`/`y`: random windows of `src` (rows of t tokens) and the tokens after them.
fn fill(src: &[u16], rng: &mut Rng, t: usize, x: &mut [u32], y: &mut [u32]) {
    for row in 0..x.len() / t {
        let s = rng.below(src.len() - t - 1);
        for i in 0..t {
            x[row * t + i] = src[s + i] as u32;
            y[row * t + i] = src[s + i + 1] as u32;
        }
    }
}

/// The training step itself, on the CPU or the GPU.
enum Engine<'a> {
    Cpu { acts: Box<Acts>, grads: Vec<f32> },
    Gpu(Box<GpuTrainer<&'a dyn Backend>>),
}

impl Engine<'_> {
    /// Forward, backward, clip and AdamW on one batch; returns its loss.
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, model: &mut Model, opt: &mut AdamW, x: &[u32], y: &[u32], dropout_seed: u64, clip: f32, lr: f32) -> Result<f32> {
        match self {
            Engine::Cpu { acts, grads } => {
                grads.iter_mut().for_each(|g| *g = 0.0);
                let loss = model.forward_backward(acts, grads, x, y, Some(dropout_seed));
                clip_grad_norm(grads, clip);
                opt.step(&mut model.params, grads, lr);
                Ok(loss)
            }
            Engine::Gpu(g) => {
                let loss = g.forward_backward(x, y, Some(dropout_seed))?;
                g.clip(clip)?;
                g.adamw(lr)?;
                Ok(loss)
            }
        }
    }

    fn loss(&mut self, model: &Model, x: &[u32], y: &[u32]) -> Result<f32> {
        match self {
            Engine::Cpu { acts, .. } => Ok(model.loss(acts, x, y)),
            Engine::Gpu(g) => g.loss(x, y),
        }
    }
}

/// Record the held-out loss and write the checkpoint (bringing the GPU's
/// weights and optimizer state back first).
fn save(settings: &Settings, engine: &mut Engine, model: &mut Model, opt: &mut AdamW, meta: &mut Meta, val_loss: f32, bpb: f32) -> Result<()> {
    if let Engine::Gpu(g) = engine {
        g.download_into(model, opt)?;
    }
    let st = &mut meta.stats;
    st.val_loss = Some(val_loss);
    st.val_bpb = Some(bpb);
    st.history.push(HistoryPoint { step: st.steps, val_loss, val_bpb: bpb, at: iso_now() });
    if st.history.len() > 500 {
        st.history.remove(0);
    }
    checkpoint::save(settings, model, meta, Some(opt))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::config::testing;

    pub const CORPUS: &str = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n\n\n\
class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n\
    def pop(self):\n        return self.items.pop()\n\n\ndef is_even(n):\n    return n % 2 == 0\n\n\n\
for i in range(10):\n    if is_even(i):\n        print(add(i, 1))\n";

    /// Put `code` into the corpus as data/corpus/test/<name>.
    pub fn add_code(s: &Settings, name: &str, code: &str) {
        let path = corpus::corpus_dir(s).join("test").join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, code).unwrap();
    }

    /// A data dir with CORPUS in it and a model trained for `steps`.
    pub fn trained(steps: u64) -> (tempfile::TempDir, Settings) {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(&tmp.path().join("data"));
        for i in 0..20 {
            add_code(&s, &format!("m{i}.py"), CORPUS);
        }
        train(&s, Budget::Steps(steps), Size::FromRam, &mut |_| {}, Some(0), &AtomicBool::new(false)).unwrap();
        (tmp, s)
    }

    #[test]
    fn empty_and_tiny_corpora_are_clear_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(&tmp.path().join("data"));
        let stop = AtomicBool::new(false);
        let err = train(&s, Budget::Steps(1), Size::FromRam, &mut |_| {}, None, &stop).unwrap_err();
        assert!(err.downcast_ref::<NotEnoughText>().unwrap().0.contains("no code to learn from"));
        add_code(&s, "t.py", "x = 1\n");
        let err = train(&s, Budget::Steps(1), Size::FromRam, &mut |_| {}, None, &stop).unwrap_err();
        assert!(err.to_string().contains("KB of code so far"));
        s.training.min_new_model_chars = 1;
        let err = train(&s, Budget::Steps(1), Size::FromRam, &mut |_| {}, None, &stop).unwrap_err();
        assert!(err.to_string().contains("only"));
    }

    #[test]
    fn learns_resumes_and_stops_cleanly() {
        let (_tmp, s) = trained(40);
        let meta = checkpoint::load(&s).unwrap().unwrap().2;
        let vocab = meta.config.vocab_size as f32;
        assert!(meta.stats.val_loss.unwrap() < vocab.ln() - 1.0, "held-out loss {:?}", meta.stats.val_loss);
        assert!(meta.stats.val_bpb.unwrap() > 0.0);
        let again = train(&s, Budget::Steps(10), Size::FromRam, &mut |_| {}, Some(1), &AtomicBool::new(false)).unwrap();
        assert_eq!((again.stats.steps, again.steps_this_session), (50, 10));
        assert!(checkpoint::brain_dir(&s).join("optim.bin").is_file());
        let stopped = train(&s, Budget::Steps(1000), Size::FromRam, &mut |_| {}, Some(2), &AtomicBool::new(true)).unwrap();
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
        let err = train(&s, Budget::Steps(1), Size::Tier("big"), &mut |_| {}, None, &AtomicBool::new(false)).unwrap_err();
        assert!(err.to_string().contains("already exists"));
        // a model that can't fit in memory is refused before anything is allocated
        let huge = crate::config::Tier { max_ram_gb: 1e9, n_layer: 96, n_head: 96, d_model: 12288, block_size: 2048, vocab_size: 50000, manual: true };
        let cfg = config_for(&s, &huge, 50000);
        assert!(training_bytes(&cfg, &s) > 1 << 40); // a GPT-3-sized model needs terabytes
        assert!(check_memory_in(&cfg, &s, hardware::detect().ram_gb).unwrap_err().to_string().contains("needs about"));
        assert_eq!(human_duration(90.0), "2 minutes");
        assert_eq!(human_duration(3.0 * 3600.0), "3.0 hours");
        assert_eq!(human_duration(6e8), "19 years");
    }

    #[test]
    fn the_size_follows_the_time_and_text_available() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let s = Settings::from_file(&root.join("config/aegist.yaml"), &root).unwrap();
        let (gflops, ram) = (200e9, 1024.0);
        let cpu = |ram_gb: f64| Compute::Cpu { flops: gflops, ram_gb };
        let pick = |hours: f64, tokens: f64| pick_for_hours(&s, hours, &cpu(ram), tokens).0;
        let lots = 1e12;
        // more time, bigger model - exactly the biggest the time can train well
        let mut needed: Vec<(f64, &String)> = s.model.tiers.iter().map(|(name, t)| {
            let cfg = config_for(&s, t, t.vocab_size);
            (TOKENS_PER_PARAM * crate::model::Layout::new(&cfg).total as f64 * flops_per_token(&cfg) / gflops / 3600.0, name)
        }).collect();
        needed.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(needed.iter().map(|n| n.1.as_str()).collect::<Vec<_>>(), ["tiny", "small", "medium", "large", "110m", "xl", "235m", "xxl", "730m", "1b"]);
        for w in needed.windows(2) {
            let ((h, name), (next_h, _)) = (w[0], w[1]);
            assert_eq!(pick(h * 1.01, lots), *name, "{name} needs {h:.1} h");
            assert_eq!(pick(next_h * 0.99, lots), *name);
        }
        assert_eq!(pick(0.01, lots), "tiny");
        // plenty of time but little text: a small model (big ones would just memorise it)
        assert_eq!(pick(1e6, 3e6), "tiny");
        let (_, why) = pick_for_hours(&s, 1e6, &cpu(ram), 3e6);
        assert!(why.contains("the corpus is the limit"), "{why}");
        // too little memory for the big ones: the biggest that fits
        let small_ram = pick_for_hours(&s, 1e6, &cpu(2.0), lots).0;
        let fits = |name: &str| training_bytes(&config_for(&s, &s.model.tiers[name], s.model.tiers[name].vocab_size), &s) as f64 <= 0.85 * 2.0 * (1u64 << 30) as f64;
        assert!(fits(&small_ram) && small_ram != "1b");
        assert!(s.model.tiers.keys().filter(|n| fits(n)).all(|n| s.model.tiers[n].d_model <= s.model.tiers[&small_ram].d_model));
    }

    #[test]
    fn a_planned_run_follows_one_schedule_across_sessions() {
        let plan = Some(96.0 * 3600.0);
        // session 2 of 4 (24 h each) starts where session 1 ended
        let start2 = schedule_fraction(Budget::Minutes(24.0 * 60.0), 0.0, 0, plan, 24.0 * 3600.0);
        assert!((start2 - 0.25).abs() < 1e-6);
        let end4 = schedule_fraction(Budget::Minutes(24.0 * 60.0), 24.0 * 3600.0, 99, plan, 96.0 * 3600.0);
        assert_eq!(end4, 1.0);
        assert_eq!(schedule_fraction(Budget::Minutes(60.0), 1800.0, 5, None, 1e9), 0.5); // no plan: this session
        assert_eq!(schedule_fraction(Budget::Steps(10), 1e9, 5, plan, 0.0), 0.5); // steps: this session
        assert!((learning_rate(1e-3, 0.1, 1.0, 0.0) - 1e-3).abs() < 1e-9);
        assert!((learning_rate(1e-3, 0.1, 1.0, 1.0) - 1e-4).abs() < 1e-9);
        assert!((learning_rate(1e-3, 0.1, 0.5, 0.5) - 0.5 * 5.5e-4).abs() < 1e-9);
    }

    #[test]
    fn commas_format() {
        assert_eq!((commas(0), commas(999), commas(1000), commas(1234567)),
                   ("0".into(), "999".into(), "1,000".into(), "1,234,567".into()));
    }
}
