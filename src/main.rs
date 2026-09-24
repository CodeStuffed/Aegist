//! `council` - the command-line program: ask, train, research, doctor.

use anyhow::Result;
use clap::{Parser, Subcommand};
use council::brain::{Brain, NoBrain};
use council::checkpoint;
use council::config::Settings;
use council::council::{evaluate, Evaluation, Judgement, Options, PersonaResult};
use council::corpus;
use council::hardware;
use council::knowledge::KnowledgeStore;
use council::model::{Layout, ModelConfig};
use council::research::{self, SystemClock};
use council::session_log;
use council::trainer::{self, commas, Budget, NotEnoughText};
use council::util::wrap;
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
    /// Worker threads (default: every core for train/research, up to 4 for ask).
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
    },
    /// Read Wikipedia on its own and keep training on what it finds.
    Research {
        /// How long to run, in hours.
        #[arg(long, default_value_t = 1.0)]
        hours: f64,
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
            Command::Ask { .. } | Command::Doctor => default_threads.min(4),
            _ => default_threads,
        });
    let _ = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build_global();

    match run(cli.command) {
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
        Command::Ask { claim, no_recheck, json } => {
            let brain = Brain::load(&settings)?;
            let opts = Options { recheck: no_recheck.then_some(false), ..Default::default() };
            let result = evaluate(&brain, &claim, &settings, &opts, &mut |m| eprintln!("  ... {m}"))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("{}", render(&result));
            }
        }
        Command::Train { data, hours, steps } => {
            if let Some(src) = data {
                let n = corpus::import_texts(&src, &settings)?;
                println!("Imported {n} file(s) into {}.", corpus::corpus_dir(&settings).display());
            }
            install_ctrl_c();
            let budget = steps.map(Budget::Steps).unwrap_or(Budget::Minutes(hours * 60.0));
            let report = trainer::train(&settings, budget, &mut |m| println!("{m}"), None, &STOP)?;
            if report.interrupted {
                println!("Stopped; the checkpoint was saved.");
            }
        }
        Command::Research { hours } => {
            install_ctrl_c();
            println!("Researching and self-training for {hours} hour(s). Ctrl-C stops it; progress is saved as it goes.");
            let s = &settings;
            let summary = research::run(
                hours,
                s,
                &mut |m| println!("{m}"),
                &SystemClock,
                &mut |topic| research::fetch_wikipedia(topic, s),
                &mut |minutes, log| Ok(trainer::train(s, Budget::Minutes(minutes), log, None, &STOP)?.interrupted),
                &STOP,
            )?;
            if summary.interrupted {
                println!("Stopped; everything collected so far is saved.");
            }
            println!("Done: {} topic(s), {} new passages, {:.0} min of training.",
                summary.topics.len(), summary.passages_stored, summary.training_minutes);
        }
        Command::Doctor => doctor(&settings)?,
    }
    Ok(ExitCode::SUCCESS)
}

fn doctor(settings: &Settings) -> Result<()> {
    let hw = hardware::detect();
    let tier_name = hardware::pick_tier(hw.ram_gb, &settings.model.tiers);
    let t = &settings.model.tiers[&tier_name];
    let est = ModelConfig {
        vocab_size: t.vocab_size, block_size: t.block_size, n_layer: t.n_layer, n_head: t.n_head, d_model: t.d_model,
        mlp_hidden: ModelConfig::default_mlp_hidden(t.d_model), dropout: 0.0, rope_base: settings.model.rope_base,
    };
    println!("Files");
    println!("  Settings   {}", settings.root.join("config").join("settings.yaml").display());
    println!("  Data       {}", settings.data_dir.display());
    println!("Hardware");
    println!("  RAM        {:.1} GB", hw.ram_gb);
    println!("  CPU        {} cores, {} threads, {}", hw.cpu_cores, hw.threads, hw.simd);
    println!("  Tier       {tier_name} -> a new model would be {} layers x {} wide, {}-token context (~{:.1}M params)",
        t.n_layer, t.d_model, t.block_size, Layout::new(&est).total as f64 / 1e6);

    let files = corpus::corpus_files(settings);
    let mb: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    println!("Corpus");
    println!("  {} file(s), {:.1} MB in {}", files.len(), mb as f64 / 1e6, corpus::corpus_dir(settings).display());

    println!("Model");
    match checkpoint::load(settings) {
        Ok(Some((model, tok, meta))) => {
            let st = &meta.stats;
            let c = &model.cfg;
            println!("  {}x{} transformer, {:.2}M params, vocabulary {}", c.n_layer, c.d_model, model.num_params() as f64 / 1e6, tok.vocab_size());
            println!("  {} steps, {:.1}M tokens seen, {:.1} h trained, held-out loss {} ({} bits per byte)",
                commas(st.steps), st.tokens_seen as f64 / 1e6, st.train_seconds / 3600.0,
                st.val_loss.map_or("-".into(), |v| format!("{v:.3}")), st.val_bpb.map_or("-".into(), |v| format!("{v:.3}")));
        }
        Ok(None) => println!("  none yet - run `council train --data <folder> --hours 1`"),
        Err(e) => println!("  {e}"),
    }

    println!("Memory");
    println!("  {} passage(s) in the knowledge base", KnowledgeStore::open(settings)?.count());
    println!("  {} past council session(s)", session_log::sessions(settings).len());
    Ok(())
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
        format!("Knowledge base: {} relevant passage(s)", r.memory.len()),
        String::new(),
    ];
    for p in &r.panel {
        out.extend(render_persona(p));
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
