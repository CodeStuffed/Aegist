//! `council` - the command-line program: ask, train, research, doctor.

use anyhow::Result;
use clap::{Parser, Subcommand};
use council::brain::{Brain, NoBrain};
use council::checkpoint;
use council::config::{Device, Settings};
use council::council::{evaluate, Evaluation, Judgement, Options, PersonaResult};
use council::corpus;
use council::hardware;
use council::knowledge::KnowledgeStore;
use council::model::{Layout, ModelConfig};
use council::quant::Precision;
use council::research::{self, SystemClock};
use council::session_log;
use council::trainer::{self, commas, Budget, NotEnoughText, Size};
use council::util::wrap;
use council::wikipedia;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

const WIDTH: usize = 88;

static STOP: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(name = "council", version, about = "A council of personas on a transformer written and trained from scratch.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Worker threads (default: every core for train/research, up to 2 for ask).
    #[arg(long, global = true)]
    threads: Option<usize>,
}

#[derive(Subcommand)]
enum Command {
    /// Put a claim in front of the council.
    Ask {
        /// The statement to judge (use quotes).
        claim: String,
        /// Skip the repeat run that checks the council agrees with itself.
        #[arg(long)]
        no_recheck: bool,
        /// Print the full result as JSON.
        #[arg(long)]
        json: bool,
        /// Weights to answer with: f32 (exact), int8 or int4 (smaller, faster).
        /// Default: inference.precision in settings.yaml.
        #[arg(long, value_enum)]
        precision: Option<Precision>,
    },
    /// Train (or keep training) the model on the corpus.
    Train {
        /// Folder (or file) of .txt/.md text to add to the corpus first.
        #[arg(long)]
        data: Option<PathBuf>,
        /// How long to train, in hours (decimals work).
        #[arg(long, default_value_t = 1.0, conflicts_with = "steps")]
        hours: f64,
        /// Train for this many steps instead of by time.
        #[arg(long)]
        steps: Option<u64>,
        /// Size for a NEW model, by tier name in settings.yaml (see `council doctor`).
        /// Default: the size that will be smartest after --plan-hours of training.
        #[arg(long)]
        tier: Option<String>,
        /// Total hours you plan to train a NEW model, across all sessions
        /// (default: --hours). Picks its size.
        #[arg(long)]
        plan_hours: Option<f64>,
        /// Where to train: auto (an NVIDIA GPU if one passes its self-check, else the CPU), cpu or gpu.
        #[arg(long, value_enum)]
        device: Option<Device>,
    },
    /// Read Wikipedia on its own and keep training on what it finds.
    Research {
        /// How long to run, in hours.
        #[arg(long, default_value_t = 1.0)]
        hours: f64,
        /// Where to train: auto, cpu or gpu (as for train).
        #[arg(long, value_enum)]
        device: Option<Device>,
    },
    /// Test the NVIDIA GPU: compile the kernels, check its results against the
    /// CPU's, measure its speed, and show which model sizes it can train.
    GpuCheck,
    /// Import a Wikipedia dump (.xml.bz2 or .xml) into the corpus and the knowledge base.
    ImportWikipedia {
        /// The dump file, e.g. enwiki-latest-pages-articles-multistream.xml.bz2.
        dump: PathBuf,
        /// Stop after this many articles (running again continues from there).
        #[arg(long)]
        max_articles: Option<u64>,
        /// Only add text to the training corpus, not to the knowledge base.
        #[arg(long)]
        no_knowledge: bool,
    },
    /// Show hardware, model, and memory status.
    Doctor,
}

fn main() -> ExitCode {
    // Like other command-line tools, stop quietly when the reader of our
    // output goes away (e.g. `council doctor | head`) instead of panicking.
    #[cfg(unix)]
    // SAFETY: restoring the default SIGPIPE action before any threads start.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    let default_threads = hardware::detect().threads;
    let threads = cli
        .threads
        .or_else(|| std::env::var("COUNCIL_THREADS").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(match cli.command {
            // Answering one question is small work; leave cores for a training run.
            Command::Ask { .. } | Command::Doctor => default_threads.min(2),
            _ => default_threads,
        });
    let pool = match rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build() {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("error: couldn't start worker threads: {e}");
            return ExitCode::from(2);
        }
    };
    // Everything runs inside the pool: parallel work started from a pool
    // thread is cheap, which matters for the many small steps of generation.
    match pool.install(|| run(cli.command)) {
        Ok(code) => code,
        Err(e) => {
            if e.downcast_ref::<NoBrain>().is_some() || e.downcast_ref::<NotEnoughText>().is_some() {
                eprintln!("{e}");
            } else {
                eprintln!("error: {e:#}");
            }
            ExitCode::from(2)
        }
    }
}

fn install_ctrl_c() {
    let _ = ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!("\nStopping after this step and saving (press Ctrl-C again to quit immediately)...");
    });
}

fn run(command: Command) -> Result<ExitCode> {
    let settings = Settings::load()?;
    match command {
        Command::Ask { claim, no_recheck, json, precision } => {
            let mut settings = settings;
            if let Some(p) = precision {
                settings.inference.precision = p;
            }
            let brain = Brain::load(&settings)?;
            let opts = Options { recheck: no_recheck.then_some(false), ..Default::default() };
            let result = evaluate(&brain, &claim, &settings, &opts, &mut |m| eprintln!("  ... {m}"))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("{}", render(&result));
            }
        }
        Command::Train { data, hours, steps, tier, plan_hours, device } => {
            let mut settings = settings;
            if let Some(d) = device {
                settings.gpu.device = d;
            }
            if let Some(src) = data {
                let n = corpus::import_texts(&src, &settings)?;
                println!("Imported {n} file(s) into {}.", corpus::corpus_dir(&settings).display());
            }
            install_ctrl_c();
            let budget = steps.map(Budget::Steps).unwrap_or(Budget::Minutes(hours * 60.0));
            let size = match (&tier, steps) {
                (Some(t), _) => Size::Tier(t),
                (None, Some(_)) if plan_hours.is_none() => Size::FromRam,
                (None, _) => Size::ForHours(plan_hours.unwrap_or(hours)),
            };
            let report = trainer::train(&settings, budget, size, &mut |m| println!("{m}"), None, &STOP)?;
            if report.interrupted {
                println!("Stopped; the checkpoint was saved.");
            }
        }
        Command::Research { hours, device } => {
            let mut settings = settings;
            if let Some(d) = device {
                settings.gpu.device = d;
            }
            install_ctrl_c();
            println!("Researching and self-training for {hours} hour(s). Ctrl-C stops it; progress is saved as it goes.");
            let s = &settings;
            let summary = research::run(
                hours,
                s,
                &mut |m| println!("{m}"),
                &SystemClock,
                &mut |topic| research::fetch_wikipedia(topic, s),
                &mut |minutes, log| Ok(trainer::train(s, Budget::Minutes(minutes), Size::ForHours(hours), log, None, &STOP)?.interrupted),
                &STOP,
            )?;
            if summary.interrupted {
                println!("Stopped; everything collected so far is saved.");
            }
            println!("Done: {} topic(s), {} new passages, {:.0} min of training.",
                summary.topics.len(), summary.passages_stored, summary.training_minutes);
        }
        Command::ImportWikipedia { dump, max_articles, no_knowledge } => {
            install_ctrl_c();
            println!("Importing {} (Ctrl-C stops it; everything imported so far is kept).", dump.display());
            let opts = wikipedia::ImportOptions { max_articles, knowledge: !no_knowledge };
            let r = wikipedia::import(&dump, &settings, &opts, &mut |m| println!("{m}"), &STOP)?;
            println!("{}: {} articles, {:.1} MB of text for training, {} passages for the knowledge base.",
                if r.interrupted { "Stopped" } else { "Done" }, commas(r.articles), r.corpus_bytes as f64 / 1e6, commas(r.passages));
            if r.articles > 0 {
                println!("Next: `council train --hours <how long>` trains on it (see `council doctor` for sizes).");
            }
        }
        Command::Doctor => doctor(&settings)?,
        Command::GpuCheck => return gpu_check(&settings),
    }
    Ok(ExitCode::SUCCESS)
}

fn gpu_check(settings: &Settings) -> Result<ExitCode> {
    use council::gpu::{self, Backend};
    println!("Opening the GPU and compiling the kernels...");
    let be = match gpu::cuda::CudaBackend::open() {
        Ok(be) => be,
        Err(e) => {
            println!("No usable GPU: {e:#}");
            println!("Training uses the CPU. For an NVIDIA GPU, install its driver and the CUDA Toolkit 12.8+ (see HOW_TO_RUN.md).");
            return Ok(ExitCode::from(1));
        }
    };
    println!("GPU       {}", be.describe());
    let r = gpu::self_check(&be)?;
    println!("Check     {} - loss {:.5} (CPU {:.5}); worst gradient error {:.1e} ({}); weights after one step {:.1e}{}",
        if r.passed { "passed" } else { "FAILED" }, r.loss_gpu, r.loss_cpu, r.worst_grad.1, r.worst_grad.0, r.step_error,
        r.bf16_grad.as_ref().map_or(String::new(), |b| format!("; with bf16: {:.1e} ({})", b.1, b.0)));
    if !r.passed {
        println!("The GPU's results don't match the CPU's, so training won't use it. Update the NVIDIA driver and CUDA Toolkit and try again.");
        return Ok(ExitCode::from(1));
    }
    let tf32 = gpu::gemm_flops(&be, false)?;
    let bf16 = match settings.gpu.precision {
        council::config::GpuPrecision::Auto => be.supports_bf16(),
        council::config::GpuPrecision::Bf16 => true,
        council::config::GpuPrecision::Tf32 => false,
    };
    let flops = if bf16 { gpu::gemm_flops(&be, true)? } else { tf32 };
    let (free, total) = be.memory()?;
    println!("Speed     {:.1} TFLOP/s on TF32 matrix multiplies{}", tf32 / 1e12,
        if bf16 { format!(", {:.1} on bf16 (used for training)", flops / 1e12) } else { String::new() });
    println!("Memory    {:.1} GB free of {:.1} GB", free as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);
    let compute = trainer::Compute::Gpu { backend: Box::new(be), flops: flops * trainer::GPU_EFFICIENCY, bf16 };
    let mut tiers: Vec<_> = settings.model.tiers.iter().collect();
    tiers.sort_by_key(|(_, t)| (t.n_layer * t.d_model * t.d_model, t.vocab_size));
    println!("Sizes on this GPU (times are rough: the real speed shows once training runs)");
    for (name, t) in tiers {
        let cfg = ModelConfig {
            vocab_size: t.vocab_size, block_size: t.block_size, n_layer: t.n_layer, n_head: t.n_head, d_model: t.d_model,
            mlp_hidden: ModelConfig::default_mlp_hidden(t.d_model), dropout: settings.model.dropout, rope_base: settings.model.rope_base,
        };
        let params = Layout::new(&cfg).total as f64;
        let well = compute.flops_per_token(&cfg) * trainer::TOKENS_PER_PARAM * params / compute.training_flops();
        let micro = gpu::micro_batch_for(&cfg, free, 64, bf16);
        println!("  {name:7} {:>7} params | {} | ~{:>11} to train well",
            human_count(params), micro.map_or("too big for this GPU".to_string(), |s| format!("{s:>2} sequences per micro-batch")),
            trainer::human_duration(well));
    }
    let mb: u64 = corpus::corpus_files(settings).iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    let picks: Vec<String> = [8.0, 24.0, 96.0]
        .iter()
        .map(|&h| format!("{h} h -> {}", trainer::pick_for_hours(settings, h, &compute, mb as f64 / 4.0).0))
        .collect();
    println!("A new model sized for its training time on this GPU{}: {}",
        if mb == 0 { " (no text yet, so the smallest)" } else { ", with this corpus" }, picks.join(", "));
    Ok(ExitCode::SUCCESS)
}

fn doctor(settings: &Settings) -> Result<()> {
    let hw = hardware::detect();
    println!("Files");
    println!("  Settings   {}", settings.root.join("config").join("settings.yaml").display());
    println!("  Data       {}", settings.data_dir.display());
    println!("Hardware");
    println!("  RAM        {:.1} GB", hw.ram_gb);
    println!("  CPU        {} cores, {} threads, {}", hw.cpu_cores, hw.threads, hw.simd);
    match council::gpu::cuda::probe() {
        Ok(g) => println!("  GPU        {} ({:.1} GB, CUDA driver {}.{}) - `council gpu-check` tests it and shows its sizes",
            g.name, g.total_memory as f64 / (1u64 << 30) as f64, g.driver_version / 1000, g.driver_version % 1000 / 10),
        Err(e) => println!("  GPU        none usable ({e}); training uses the CPU"),
    }

    // Rough training speed: a matrix multiply timed on every core
    let cpu = trainer::Compute::cpu();
    let flops_per_s = cpu.training_flops();
    println!("Sizes (to choose one: council train --tier NAME; times assume {:.0} GFLOP/s)", flops_per_s / 1e9);
    let mut tiers: Vec<_> = settings.model.tiers.iter().collect();
    tiers.sort_by_key(|(_, t)| (t.n_layer * t.d_model * t.d_model, t.vocab_size));
    for (name, t) in tiers {
        let cfg = ModelConfig {
            vocab_size: t.vocab_size, block_size: t.block_size, n_layer: t.n_layer, n_head: t.n_head, d_model: t.d_model,
            mlp_hidden: ModelConfig::default_mlp_hidden(t.d_model), dropout: settings.model.dropout, rope_base: settings.model.rope_base,
        };
        let params = Layout::new(&cfg).total as f64;
        let mem = trainer::training_bytes(&cfg, settings) as f64 / (1u64 << 30) as f64;
        let fits = if mem > 0.85 * hw.ram_gb { "  (too big for this machine)" } else { "" };
        let well = trainer::flops_per_token(&cfg) * trainer::TOKENS_PER_PARAM * params / flops_per_s;
        println!("  {name:7} {:>7} params | {:>6.1} GB to train | ~{:>11} to train well | int4 file {:>7}{fits}",
            human_count(params), mem, trainer::human_duration(well),
            human_bytes(0.5 * params + params / 8.0));
    }

    let files = corpus::corpus_files(settings);
    let mb: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    let picks: Vec<String> = [1.0, 8.0, 24.0, 96.0]
        .iter()
        .map(|&h| format!("{h} h -> {}", trainer::pick_for_hours(settings, h, &cpu, mb as f64 / 4.0).0))
        .collect();
    println!("  Otherwise a new model gets the size that will be smartest after its training time");
    println!("  (--hours, or --plan-hours for several sessions){}: {}",
        if mb == 0 { " - with no text yet, the smallest" } else { ", with this corpus" }, picks.join(", "));
    println!("Corpus");
    println!("  {} file(s), {:.1} MB in {}", files.len(), mb as f64 / 1e6, corpus::corpus_dir(settings).display());

    println!("Model");
    match checkpoint::load(settings) {
        Ok(Some((model, tok, meta))) => {
            let st = &meta.stats;
            let c = &model.cfg;
            println!("  {}x{} transformer, {:.2}M params, vocabulary {}", c.n_layer, c.d_model, model.num_params() as f64 / 1e6, tok.vocab_size());
            let n = model.num_params() as f64;
            println!("  Answers with {} weights (inference.precision): f32 {} | int8 ~{} | int4 ~{}",
                settings.inference.precision, human_bytes(4.0 * n), human_bytes(n + 4.0 * n / c.d_model as f64), human_bytes(0.5 * n + n / 8.0));
            println!("  {} steps, {:.1}M tokens seen, {:.1} h trained, held-out loss {} ({} bits per byte)",
                commas(st.steps), st.tokens_seen as f64 / 1e6, st.train_seconds / 3600.0,
                st.val_loss.map_or("-".into(), |v| format!("{v:.3}")), st.val_bpb.map_or("-".into(), |v| format!("{v:.3}")));
            let target = trainer::TOKENS_PER_PARAM * n;
            println!("  Trained on {:.0}% of the ~{:.0}M tokens (20 per parameter) a model this size should see",
                100.0 * st.tokens_seen as f64 / target, target / 1e6);
        }
        Ok(None) => println!("  none yet - run `council train --data <folder> --hours 1`"),
        Err(e) => println!("  {e}"),
    }

    println!("Memory");
    let kb = KnowledgeStore::open(settings)?;
    let (segments, on_disk, in_memory, bytes) = kb.stats();
    println!("  {} passage(s) in the knowledge base", commas(kb.count() as u64));
    if segments > 0 {
        println!("  Search index: {} passage(s) on disk in {segments} segment(s), {:.1} MB, memory-mapped; {} in memory",
                 commas(on_disk as u64), bytes as f64 / 1e6, commas(in_memory as u64));
    }
    println!("  {} past council session(s)", session_log::sessions(settings).len());
    Ok(())
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

fn conf_label(conf: impl std::fmt::Display, before: Option<impl std::fmt::Display>) -> String {
    match before {
        Some(b) => format!("{conf}, capped from {b}"),
        None => conf.to_string(),
    }
}

fn snippet(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let cut: String = text.chars().take(limit).collect();
    format!("{}...", cut.rsplit_once(' ').map_or(cut.as_str(), |(a, _)| a))
}

fn render_persona(p: &PersonaResult) -> Vec<String> {
    let mut lines = vec![
        format!("{}  [{}]  signal {:+.2}", p.persona.to_uppercase(), conf_label(p.confidence, p.confidence_before_cap), p.signal),
        wrap(&format!("\"{}\"", p.position), WIDTH, "  ", ""),
    ];
    for k in &p.key_points {
        let source = if k.source.is_empty() { String::new() } else { format!(" ({})", k.source) };
        lines.push(wrap(&format!("{:+.2} {}{source}", k.delta, snippet(&k.text, 160)), WIDTH, "  ", "- "));
    }
    lines
}

fn render_judge(j: &Judgement, title: &str) -> Vec<String> {
    let mut lines = vec![
        format!("{title}  [{}]  relied on: {}", conf_label(j.confidence, j.confidence_before_cap), j.relied_on.join(", ")),
        wrap(&format!("Verdict: {}", j.verdict), WIDTH, "  ", ""),
        wrap(&format!("Why: {}", j.reasoning), WIDTH, "  ", ""),
        wrap(&format!("In its own words: \"{}\"", j.in_its_own_words), WIDTH, "  ", ""),
    ];
    if let Some(u) = &j.unresolved {
        lines.push(wrap(&format!("Unresolved: {u}"), WIDTH, "  ", ""));
    }
    lines
}

fn render(r: &Evaluation) -> String {
    let m = &r.model;
    let f = &r.familiarity;
    let investor = if r.routing.is_money_idea { "Investor joins" } else { "no Investor" };
    let mut out = vec![
        wrap(&format!("CLAIM: {}", r.claim), WIDTH, "", ""),
        format!("Model: {}, {:.1}M tokens trained, held-out loss {}", m.description, m.tokens_seen as f64 / 1e6,
            m.val_loss.map_or("-".into(), |v| format!("{v:.2}"))),
        wrap(&format!("Router: {investor} - {}", r.routing.reason), WIDTH, "", ""),
        format!("Familiarity: claim loss {:.2} vs. typical {:.2} -> confidence ceiling {}", f.claim_loss, f.typical_loss, f.cap),
        format!("Knowledge base: {} relevant passage(s), {} of them by following links",
            r.memory.len(), r.memory.iter().filter(|h| h.via != "match").count()),
        String::new(),
    ];
    for p in &r.panel {
        out.extend(render_persona(p));
        out.push(String::new());
    }
    if let Some(n) = &r.negation_test {
        out.push(format!("NEGATION TEST  [{}]  {:+.2}", n.confidence, n.contrast));
        out.push(wrap(&format!("Against: \"{}\"", n.flipped), WIDTH, "  ", ""));
        out.push(wrap(&format!("The rest of the claim is {} likely as stated ({:.2} vs. {:.2} nats/token).",
            if n.contrast >= 0.0 { "more" } else { "less" }, n.stated_logp, n.flipped_logp), WIDTH, "  ", ""));
        out.push(String::new());
    }
    out.extend(render_judge(&r.judge, "JUDGE"));
    out.push(String::new());
    match &r.consistency {
        Some(c) if !c.agreed => {
            out.extend(render_judge(&c.rerun_judge, "JUDGE, REPEAT RUN"));
            out.push(String::new());
            out.push(format!("{} ({}).", r.confidence_note.as_deref().unwrap_or(""), c.reasons.join("; ")));
        }
        Some(_) => out.push("Repeat run (dropout on) agreed with the first.".into()),
        None => {}
    }
    out.push(format!("OVERALL CONFIDENCE: {}", r.overall_confidence));
    out.join("\n")
}
