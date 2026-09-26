//! `aegist` - the command-line program. With no arguments it opens the
//! interactive session; the subcommands do one thing and exit.

use aegist::actions;
use aegist::app;
use aegist::bench;
use aegist::brain::{Brain, NoBrain};
use aegist::checkpoint;
use aegist::config::{Device, GpuPrecision, Settings};
use aegist::corpus;
use aegist::gpu::{self, Backend};
use aegist::hardware;
use aegist::learn;
use aegist::model::{Layout, ModelConfig};
use aegist::quant::Precision;
use aegist::session::{self, BrainState, Ctx, Out};
use aegist::trainer::{self, commas, human_duration, Budget, NotEnoughText, Size};
use aegist::ui::style::{self, pal, Style};
use aegist::ui::{text, widgets};
use aegist::verify::Verdict;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static STOP: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(name = "aegist", version, about = "A coding AI written and trained from scratch in Rust - its own transformer, tokenizer and training, no outside models.",
          long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Worker threads (default: every core).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// No colors.
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Add code for the model to learn from: a folder, a git URL, or a ready-made pack.
    Learn {
        /// A folder (or file) of code, or a git repository URL.
        source: Option<String>,
        /// A ready-made pack of well-known projects (see --list).
        #[arg(long)]
        pack: Option<String>,
        /// Show the packs and what's been learned so far.
        #[arg(long)]
        list: bool,
        /// Forget a source (its code and its names).
        #[arg(long)]
        forget: Option<String>,
    },
    /// Train (or keep training) the model on the code it has learned.
    Train {
        /// How long to train, in hours (decimals work).
        #[arg(long, default_value_t = 1.0, conflicts_with = "steps")]
        hours: f64,
        /// Train for this many steps instead of by time.
        #[arg(long)]
        steps: Option<u64>,
        /// Size for a NEW model, by tier name (see `aegist doctor`).
        /// Default: the size that will be smartest after --plan-hours of training.
        #[arg(long)]
        tier: Option<String>,
        /// Total hours you plan to train a NEW model, across sessions (default: --hours). Picks its size.
        #[arg(long)]
        plan_hours: Option<f64>,
        /// Where to train: auto (an NVIDIA GPU if one passes its self-check, else the CPU), cpu or gpu.
        #[arg(long, value_enum)]
        device: Option<Device>,
    },
    /// Write a new file (or add to an existing one), checked before it's saved.
    Write {
        /// The file to write.
        file: String,
        /// What it should do.
        description: String,
        /// Save it without asking, if it passes the checks.
        #[arg(long)]
        yes: bool,
    },
    /// Fill in code in a file at a line (a TODO line, or the end of the file).
    Complete {
        /// file[:line]
        target: String,
        #[arg(long)]
        yes: bool,
    },
    /// Make a failing command (default: the project's tests) pass.
    Fix {
        /// The command to make pass.
        command: Option<String>,
    },
    /// Measure the model on coding problems with tests (pass@1, pass@k, and
    /// how well its own verdicts predict the tests).
    Eval {
        /// Candidates per problem.
        #[arg(long, default_value_t = 4)]
        k: usize,
        /// Weights to use: f32, int8 or int4.
        #[arg(long, value_enum)]
        precision: Option<Precision>,
    },
    /// Hardware, model sizes, what's learned, and the model's state.
    Doctor,
    /// Test an NVIDIA GPU: compile the kernels, check them against the CPU, measure speed.
    GpuCheck,
}

fn main() -> ExitCode {
    #[cfg(unix)]
    // SAFETY: restoring the default SIGPIPE action before any threads start,
    // so `aegist doctor | head` ends quietly.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    if cli.no_color {
        style::set_depth(style::Depth::None);
    }
    let threads = cli.threads.or_else(|| std::env::var("AEGIST_THREADS").ok().and_then(|v| v.parse().ok())).unwrap_or(hardware::detect().threads);
    let pool = match rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build() {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("error: couldn't start worker threads: {e}");
            return ExitCode::from(2);
        }
    };
    match pool.install(|| run(cli.command)) {
        Ok(code) => code,
        Err(e) => {
            let plain = e.downcast_ref::<NoBrain>().is_some() || e.downcast_ref::<NotEnoughText>().is_some();
            eprintln!("{} {}", style::fg(pal::RED, "✗"), if plain { e.to_string() } else { format!("{e:#}") });
            ExitCode::from(2)
        }
    }
}

fn install_ctrl_c() {
    let _ = ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!("\n{} stopping after this step and saving (ctrl+c again quits at once)…", style::fg(pal::YELLOW, "■"));
    });
}

/// A worker context that prints straight to stdout.
fn plain_ctx(settings: Settings, yes: bool) -> Result<Ctx> {
    let cwd = std::env::current_dir()?;
    let mut shared = session::new_shared(&settings, cwd.clone());
    shared.project = Some(aegist::project::Project::open(&cwd));
    shared.brain = match Brain::load(&settings) {
        Ok(b) => BrainState::Ready(Arc::new(b)),
        Err(e) => return Err(e),
    };
    let width = crossterm::terminal::size().map(|(w, _)| w as usize).unwrap_or(100);
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let _ = ctrlc::set_handler(move || c2.store(true, Ordering::SeqCst));
    Ok(Ctx { settings: Arc::new(settings), shared: Arc::new(Mutex::new(shared)), out: Out::Plain { yes }, cancel, width: Arc::new(AtomicUsize::new(width)) })
}

fn heading(title: &str) {
    println!("\n  {}", style::gradient_styled(title, 0.15, true));
}

fn row(key: &str, value: impl std::fmt::Display) {
    println!("    {}  {value}", Style::new().fg(pal::VIOLET).paint(&text::pad(key, 11)));
}

fn run(command: Option<Command>) -> Result<ExitCode> {
    let settings = Settings::load()?;
    let Some(command) = command else {
        app::run(settings)?;
        return Ok(ExitCode::SUCCESS);
    };
    match command {
        Command::Learn { source, pack, list, forget } => {
            if let Some(f) = forget {
                let gone = learn::forget(&settings, &f)?;
                println!("  {} {}", if gone { style::fg(pal::GREEN, "✓") } else { style::fg(pal::YELLOW, "!") },
                         if gone { format!("forgot {f}") } else { format!("nothing learned under {f}") });
                return Ok(ExitCode::SUCCESS);
            }
            let mut log = |m: String| println!("  {} {}", style::faint("·"), style::dim(&m));
            let reports = match (source, pack) {
                (_, Some(name)) => {
                    let Some(p) = learn::pack(&name) else {
                        anyhow::bail!("no pack named {name:?}; `aegist learn --list` shows them");
                    };
                    install_ctrl_c();
                    let mut out = Vec::new();
                    for repo in p.repos {
                        if STOP.load(Ordering::Relaxed) {
                            break;
                        }
                        match learn::learn_git(&settings, repo, &mut log) {
                            Ok(r) => out.push(r),
                            Err(e) => println!("  {} {repo}: {e}", style::fg(pal::YELLOW, "!")),
                        }
                    }
                    out
                }
                (Some(s), None) if learn::is_git_url(&s) => vec![learn::learn_git(&settings, &s, &mut log)?],
                (Some(s), None) => vec![learn::learn_path(&settings, &PathBuf::from(s), None, &mut log)?],
                (None, None) => {
                    let _ = list;
                    heading("Packs");
                    for p in learn::PACKS {
                        println!("    {}  {}", Style::new().fg(pal::VIOLET).bold().paint(&text::pad(p.name, 12)), style::dim(p.about));
                    }
                    println!("    {}", style::faint("aegist learn --pack <name>   ·   aegist learn <folder or git URL>"));
                    let learned = learn::sources(&settings);
                    heading("Learned");
                    if learned.is_empty() {
                        println!("    {}", style::dim("nothing yet"));
                    }
                    for (src, files, bytes) in learned {
                        println!("    {}  {}", text::pad(&src, 28), style::faint(&format!("{files} files · {:.1} MB", bytes as f64 / 1e6)));
                    }
                    return Ok(ExitCode::SUCCESS);
                }
            };
            for r in &reports {
                println!("  {} {} {}", style::fg(pal::GREEN, "✓"), Style::new().bold().paint(&r.source),
                    style::dim(&format!("{} files · {:.1} MB · {} names · {} skipped · {} duplicates", r.files, r.bytes as f64 / 1e6,
                                        commas(r.names as u64), r.skipped, r.duplicates)));
            }
            println!("  {}", style::faint("next: aegist train --hours <how long>"));
        }
        Command::Train { hours, steps, tier, plan_hours, device } => {
            let mut settings = settings;
            if let Some(d) = device {
                settings.gpu.device = d;
            }
            install_ctrl_c();
            let budget = steps.map(Budget::Steps).unwrap_or(Budget::Minutes(hours * 60.0));
            if let (Some(h), true) = (plan_hours, checkpoint::exists(&settings)) {
                checkpoint::set_planned_seconds(&settings, h * 3600.0)?;
                println!("  {}", style::dim(&format!("This model's training is now planned at {h} hours in total; the learning rate follows that.")));
            }
            let size = match (&tier, steps) {
                (Some(t), _) => Size::Tier(t),
                (None, Some(_)) if plan_hours.is_none() => Size::FromRam,
                (None, _) => Size::ForHours(plan_hours.unwrap_or(hours)),
            };
            heading(&format!("Training · {}", steps.map_or(human_duration(hours * 3600.0), |s| format!("{s} steps"))));
            let mut losses: Vec<f32> = Vec::new();
            let mut log = |m: String| {
                if m.starts_with("step") {
                    if let Some(l) = m.split("loss ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|v| v.parse::<f32>().ok()) {
                        losses.push(l);
                    }
                    println!("    {}  {}", style::dim(&m.replace(" | ", " · ")), widgets::sparkline(&losses, 20));
                } else {
                    for l in text::wrap(&m, 96) {
                        println!("    {}", style::dim(&l));
                    }
                }
            };
            let report = trainer::train(&settings, budget, size, &mut log, None, &STOP)?;
            println!("  {} {}", style::fg(pal::GREEN, "✓"), if report.interrupted { "stopped; the checkpoint was saved" } else { "done" });
        }
        Command::Write { file, description, yes } => {
            let ctx = plain_ctx(settings, yes)?;
            actions::write(&ctx, Some(&file), None, &description)?;
        }
        Command::Complete { target, yes } => {
            let ctx = plain_ctx(settings, yes)?;
            actions::complete(&ctx, &target)?;
        }
        Command::Fix { command } => {
            let ctx = plain_ctx(settings, true)?;
            actions::fix(&ctx, command.as_deref())?;
        }
        Command::Eval { k, precision } => {
            let brain = Brain::load_at(&settings, precision.unwrap_or(settings.inference.precision))?;
            let names = aegist::verify::Names { learned: Arc::new(learn::known_names(&settings)), project: Arc::default() };
            heading("Evaluation");
            println!("    {}", style::dim(&format!("{} · {} problems · {k} candidates each · tests decide", brain.describe(), bench::PROBLEMS.len())));
            let outcomes = bench::run(&brain, &settings, k, &names, &mut |o, _| {
                let marks: String = o.candidates.iter().map(|(p, v)| {
                    let ch = if *p { "●" } else { "○" };
                    let c = match (p, *v > Verdict::Refused) { (true, true) => pal::GREEN, (false, false) => pal::FAINT, (true, false) => pal::CYAN, (false, true) => pal::RED };
                    style::fg(c, ch)
                }).collect();
                println!("    {}  {marks}", text::pad(o.name, 24));
            })?;
            let s = bench::summarize(&outcomes, k);
            let pct = |a: usize, b: usize| if b == 0 { "-".to_string() } else { format!("{:.0}%", 100.0 * a as f64 / b as f64) };
            heading("Results");
            row("pass@1", format!("{} / {}  {}", s.pass_at_1, s.problems, pct(s.pass_at_1, s.problems)));
            row(&format!("pass@{k}"), format!("{} / {}  {}", s.pass_at_k, s.problems, pct(s.pass_at_k, s.problems)));
            row("stood by", format!("{} candidates, {} of which passed their tests ({})", s.accepted.1, s.accepted.0, pct(s.accepted.0, s.accepted.1)));
            row("refused", format!("{} candidates, {} of which really did fail ({})", s.refused.1, s.refused.0, pct(s.refused.0, s.refused.1)));
            println!("    {}", style::faint("● passed · ○ failed · green/grey: its verdict agreed with the tests · red: it stood by code that failed · cyan: it refused code that passed"));
        }
        Command::Doctor => doctor(&settings)?,
        Command::GpuCheck => return gpu_check(&settings),
    }
    Ok(ExitCode::SUCCESS)
}

fn human_count(n: f64) -> String {
    match n {
        n if n >= 1e12 => format!("{:.1}T", n / 1e12),
        n if n >= 1e9 => format!("{:.2}B", n / 1e9),
        n if n >= 1e6 => format!("{:.1}M", n / 1e6),
        n => format!("{:.0}K", n / 1e3),
    }
}

fn human_bytes(b: f64) -> String {
    match b {
        b if b >= 1e9 => format!("{:.1} GB", b / 1e9),
        b => format!("{:.0} MB", b / 1e6),
    }
}

fn tier_config(settings: &Settings, t: &aegist::config::Tier) -> ModelConfig {
    ModelConfig {
        vocab_size: t.vocab_size, block_size: t.block_size, n_layer: t.n_layer, n_head: t.n_head, d_model: t.d_model,
        mlp_hidden: ModelConfig::default_mlp_hidden(t.d_model), dropout: settings.model.dropout, rope_base: settings.model.rope_base,
    }
}

fn doctor(settings: &Settings) -> Result<()> {
    for l in widgets::banner(100) {
        println!("{l}");
    }
    let hw = hardware::detect();
    heading("Files");
    row("settings", settings.settings_path().display());
    row("data", settings.data_dir.display());
    heading("Hardware");
    row("memory", format!("{:.1} GB", hw.ram_gb));
    row("cpu", format!("{} cores, {} threads, {}", hw.cpu_cores, hw.threads, hw.simd));
    match gpu::cuda::probe() {
        Ok(g) => row("gpu", format!("{} ({:.1} GB, CUDA driver {}.{}) - `aegist gpu-check` tests it", g.name, g.total_memory as f64 / (1u64 << 30) as f64,
                                    g.driver_version / 1000, g.driver_version % 1000 / 10)),
        Err(e) => row("gpu", style::dim(&format!("none usable ({e}); training uses the CPU"))),
    }
    let browser = aegist::ide::find_browser();
    row("browser", browser.map_or(style::dim("none found (for /preview; set AEGIST_BROWSER)"), |b| b.display().to_string()));

    let cpu = trainer::Compute::cpu();
    let flops_per_s = cpu.training_flops();
    heading(&format!("Sizes · times assume {:.0} GFLOP/s", flops_per_s / 1e9));
    let mut tiers: Vec<_> = settings.model.tiers.iter().collect();
    tiers.sort_by_key(|(_, t)| (t.n_layer * t.d_model * t.d_model, t.vocab_size));
    for (name, t) in tiers {
        let cfg = tier_config(settings, t);
        let params = Layout::new(&cfg).total as f64;
        let mem = trainer::training_bytes(&cfg, settings) as f64 / (1u64 << 30) as f64;
        let fits = if mem > 0.85 * hw.ram_gb { style::fg(pal::YELLOW, " too big for this machine") } else { String::new() };
        let well = trainer::flops_per_token(&cfg) * trainer::TOKENS_PER_PARAM * params / flops_per_s;
        println!("    {}  {:>7} params · {:>5} context · {:>6.1} GB to train · ~{:>11} to train well · int4 {:>7}{fits}",
            Style::new().fg(pal::VIOLET).bold().paint(&text::pad(name, 7)), human_count(params), t.block_size, mem, human_duration(well),
            human_bytes(0.5 * params + params / 8.0));
    }
    let files = corpus::corpus_files(settings);
    let bytes: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    let picks: Vec<String> = [1.0, 8.0, 24.0, 96.0]
        .iter()
        .map(|&h| format!("{h} h → {}", trainer::pick_for_hours(settings, h, &cpu, bytes as f64 / 4.0).0))
        .collect();
    println!("    {}", style::dim(&format!("a new model gets the size that ends up smartest in its training time: {}", picks.join(" · "))));

    heading("Learned code");
    let sources = learn::sources(settings);
    row("corpus", format!("{} files · {:.1} MB from {} sources", files.len(), bytes as f64 / 1e6, sources.len()));
    for (src, n, b) in sources.iter().take(12) {
        println!("      {}  {}", text::pad(src, 26), style::faint(&format!("{n} files · {:.1} MB", *b as f64 / 1e6)));
    }

    heading("Model");
    match checkpoint::load_meta(settings) {
        Some(meta) => {
            let c = &meta.config;
            let n = Layout::new(c).total as f64;
            let st = &meta.stats;
            row("shape", format!("{}×{} · {} heads · {:.2}M params · vocabulary {}", c.n_layer, c.d_model, c.n_head, n / 1e6, c.vocab_size));
            row("context", format!("{} tokens trained · reads up to {}", c.block_size, c.block_size * settings.inference.context_scale));
            row("weights", format!("writes with {} · f32 {} · int8 ~{} · int4 ~{}", settings.inference.precision, human_bytes(4.0 * n),
                                   human_bytes(n + 4.0 * n / c.d_model as f64), human_bytes(0.5 * n + n / 8.0)));
            row("trained", format!("{} steps · {:.1}M tokens · {} · held-out loss {} ({} bits per byte)", commas(st.steps), st.tokens_seen as f64 / 1e6,
                human_duration(st.train_seconds), st.val_loss.map_or("-".into(), |v| format!("{v:.3}")), st.val_bpb.map_or("-".into(), |v| format!("{v:.3}"))));
            let target = trainer::TOKENS_PER_PARAM * n;
            row("progress", format!("{} {:.0}% of the ~{:.0}M tokens a model this size should read", widgets::bar((st.tokens_seen as f64 / target) as f32, 16),
                100.0 * st.tokens_seen as f64 / target, target / 1e6));
        }
        None => row("model", style::fg(pal::YELLOW, "none yet - `aegist learn --pack python`, then `aegist train --hours 1`")),
    }
    println!();
    Ok(())
}

fn gpu_check(settings: &Settings) -> Result<ExitCode> {
    heading("GPU check");
    let be = match gpu::cuda::CudaBackend::open() {
        Ok(be) => be,
        Err(e) => {
            println!("    {} no usable GPU: {e:#}", style::fg(pal::YELLOW, "!"));
            println!("    {}", style::dim("Training uses the CPU. For an NVIDIA GPU, install its driver and the CUDA Toolkit 12.8+ (docs/GPU.md)."));
            return Ok(ExitCode::from(1));
        }
    };
    row("gpu", be.describe());
    let r = gpu::self_check(&be)?;
    row("check", format!("{} · loss {:.5} (CPU {:.5}) · worst gradient error {:.1e} ({}) · weights after one step {:.1e}{}",
        if r.passed { style::fg(pal::GREEN, "passed") } else { style::fg(pal::RED, "FAILED") }, r.loss_gpu, r.loss_cpu, r.worst_grad.1, r.worst_grad.0, r.step_error,
        r.bf16_grad.as_ref().map_or(String::new(), |b| format!(" · bf16: {:.1e} ({})", b.1, b.0))));
    if !r.passed {
        println!("    {}", style::dim("The GPU's results don't match the CPU's, so training won't use it. Update the NVIDIA driver and CUDA Toolkit."));
        return Ok(ExitCode::from(1));
    }
    let tf32 = gpu::gemm_flops(&be, false)?;
    let bf16 = match settings.gpu.precision {
        GpuPrecision::Auto => be.supports_bf16(),
        GpuPrecision::Bf16 => true,
        GpuPrecision::Tf32 => false,
    };
    let flops = if bf16 { gpu::gemm_flops(&be, true)? } else { tf32 };
    let (free, total) = be.memory()?;
    row("speed", format!("{:.1} TFLOP/s TF32{}", tf32 / 1e12, if bf16 { format!(" · {:.1} bf16 (used for training)", flops / 1e12) } else { String::new() }));
    row("memory", format!("{:.1} GB free of {:.1} GB", free as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64));
    let compute = trainer::Compute::Gpu { backend: Box::new(be), flops: flops * trainer::GPU_EFFICIENCY, bf16 };
    let mut tiers: Vec<_> = settings.model.tiers.iter().collect();
    tiers.sort_by_key(|(_, t)| (t.n_layer * t.d_model * t.d_model, t.vocab_size));
    heading("Sizes on this GPU");
    for (name, t) in tiers {
        let cfg = tier_config(settings, t);
        let params = Layout::new(&cfg).total as f64;
        let well = compute.flops_per_token(&cfg) * trainer::TOKENS_PER_PARAM * params / compute.training_flops();
        let micro = gpu::micro_batch_for(&cfg, free, 64, bf16);
        println!("    {}  {:>7} params · {} · ~{:>11} to train well", Style::new().fg(pal::VIOLET).bold().paint(&text::pad(name, 7)), human_count(params),
            micro.map_or(style::fg(pal::YELLOW, "too big for this GPU"), |s| format!("{s:>2} sequences per micro-batch")), human_duration(well));
    }
    Ok(ExitCode::SUCCESS)
}
