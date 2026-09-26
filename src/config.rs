//! Settings (config/aegist.yaml).
//!
//! Aegist's home - where its settings and trained model live - is found by
//! walking up from the executable (a build inside this repo uses the repo),
//! else ~/.aegist, where the built-in default settings are written on first
//! run so a downloaded binary works on its own. The code you work on is
//! wherever you run `aegist`; it is never mixed up with Aegist's home.
//! Overrides:
//!   AEGIST_HOME      Aegist's home folder
//!   AEGIST_SETTINGS  an alternate aegist.yaml
//!   AEGIST_DATA_DIR  where the corpus and model are written

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
pub struct Tier {
    pub max_ram_gb: f64,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,
    pub block_size: usize,
    pub vocab_size: usize,
    /// Only used when asked for by name (`aegist train --tier NAME`).
    #[serde(default)]
    pub manual: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelSettings {
    pub tiers: BTreeMap<String, Tier>,
    pub dropout: f32,
    pub rope_base: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainingSettings {
    pub tokens_per_step: usize,
    pub learning_rate: f32,
    pub min_lr_fraction: f32,
    pub warmup_steps: u64,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub val_fraction: f64,
    #[serde(default = "default_max_val")]
    pub max_val_tokens: usize,
    pub eval_batches: usize,
    pub eval_every_s: f64,
    pub checkpoint_every_s: f64,
    pub log_every_s: f64,
    pub tokenizer_train_chars: usize,
    pub min_corpus_tokens: usize,
    pub min_new_model_chars: usize,
}

fn default_max_val() -> usize {
    20_000_000
}

#[derive(Clone, Debug, Deserialize)]
pub struct CodeSettings {
    pub fim_rate: f64,
    pub max_file_kb: u64,
    pub max_line_chars: usize,
}

impl Default for CodeSettings {
    fn default() -> Self {
        CodeSettings { fim_rate: 0.5, max_file_kb: 512, max_line_chars: 1000 }
    }
}

fn default_precision() -> crate::quant::Precision {
    crate::quant::Precision::Int8
}

#[derive(Clone, Debug, Deserialize)]
pub struct InferenceSettings {
    /// Weights used to write code: f32 (exact), int8 or int4 (smaller, faster).
    #[serde(default = "default_precision")]
    pub precision: crate::quant::Precision,
    #[serde(default = "default_context_scale")]
    pub context_scale: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_new")]
    pub max_new_tokens: usize,
    #[serde(default = "yes")]
    pub speculative: bool,
}

fn default_context_scale() -> usize {
    4
}
fn default_temperature() -> f32 {
    0.2
}
fn default_top_p() -> f32 {
    0.95
}
fn default_max_new() -> usize {
    400
}
fn yes() -> bool {
    true
}

impl Default for InferenceSettings {
    fn default() -> Self {
        InferenceSettings {
            precision: default_precision(),
            context_scale: default_context_scale(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_new_tokens: default_max_new(),
            speculative: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct HonestySettings {
    pub samples: usize,
    pub min_confidence: f32,
    pub uncertain_below: f32,
    pub max_invented_names: usize,
    pub require_syntax_check: bool,
    pub check_timeout_s: u64,
}

impl Default for HonestySettings {
    fn default() -> Self {
        HonestySettings { samples: 4, min_confidence: 0.35, uncertain_below: 0.3, max_invented_names: 0, require_syntax_check: true,
                          check_timeout_s: 20 }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct DecideSettings {
    /// Below this probability for its best option, a decision is "I don't know".
    pub abstain_below: f32,
    /// ...or when the best option isn't ahead of the next by at least this much.
    pub min_margin: f32,
    /// Subtract each option's bias (its score with the question blanked out).
    pub contextual_calibration: bool,
}

impl Default for DecideSettings {
    fn default() -> Self {
        DecideSettings { abstain_below: 0.5, min_margin: 0.1, contextual_calibration: true }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct AgentSettings {
    /// Most actions (clicks, drags, keys) in one run.
    pub max_actions: usize,
    /// Pause after each action, so the app can draw.
    pub action_delay_ms: u64,
    /// Ask before touching the real desktop.
    pub confirm: bool,
    /// Stop if you move the mouse this far from where Aegist put it.
    pub failsafe_px: i32,
}

impl Default for AgentSettings {
    fn default() -> Self {
        AgentSettings { max_actions: 300, action_delay_ms: 120, confirm: true, failsafe_px: 40 }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct VoiceSettings {
    /// Speak replies aloud with the system's own text-to-speech.
    pub enabled: bool,
}

/// Where training runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    /// An NVIDIA GPU if one works (after a self-check), else the CPU.
    #[default]
    Auto,
    Cpu,
    Gpu,
}

/// Number format of the GPU's matrix multiplies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuPrecision {
    /// bf16 on GPUs with bf16 tensor cores (RTX 30xx and newer), else tf32
    #[default]
    Auto,
    Bf16,
    Tf32,
}

fn default_gpu_tokens() -> usize {
    131_072
}
fn default_gpu_lr() -> f32 {
    6e-4
}
fn default_gpu_warmup() -> u64 {
    500
}
fn default_gpu_checkpoint() -> f64 {
    1800.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct GpuSettings {
    #[serde(default)]
    pub device: Device,
    #[serde(default)]
    pub precision: GpuPrecision,
    /// Tokens per optimizer step on the GPU (split into micro-batches that fit its memory).
    #[serde(default = "default_gpu_tokens")]
    pub tokens_per_step: usize,
    #[serde(default = "default_gpu_lr")]
    pub learning_rate: f32,
    #[serde(default = "default_gpu_warmup")]
    pub warmup_steps: u64,
    /// Seconds between checkpoints on the GPU (big models: GBs per save).
    #[serde(default = "default_gpu_checkpoint")]
    pub checkpoint_every_s: f64,
}

impl Default for GpuSettings {
    fn default() -> Self {
        GpuSettings { device: Device::Auto, precision: GpuPrecision::Auto, tokens_per_step: default_gpu_tokens(), learning_rate: default_gpu_lr(),
                      warmup_steps: default_gpu_warmup(), checkpoint_every_s: default_gpu_checkpoint() }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PathSettings {
    pub data_dir: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    pub model: ModelSettings,
    pub training: TrainingSettings,
    #[serde(default)]
    pub code: CodeSettings,
    #[serde(default)]
    pub inference: InferenceSettings,
    #[serde(default)]
    pub honesty: HonestySettings,
    #[serde(default)]
    pub gpu: GpuSettings,
    #[serde(default)]
    pub decide: DecideSettings,
    #[serde(default)]
    pub agent: AgentSettings,
    #[serde(default)]
    pub voice: VoiceSettings,
    pub paths: PathSettings,
    /// Aegist's home (holds config/).
    #[serde(skip)]
    pub root: PathBuf,
    /// Resolved data directory (paths.data_dir, or AEGIST_DATA_DIR).
    #[serde(skip)]
    pub data_dir: PathBuf,
}

const SETTINGS_FILE: &str = "aegist.yaml";

fn has_config(dir: &Path) -> bool {
    dir.join("config").join(SETTINGS_FILE).is_file()
}

/// The default settings, built into the binary.
pub const DEFAULT_SETTINGS: &str = include_str!("../config/aegist.yaml");

/// Write the built-in settings under `root/config/` unless some are already there.
pub fn ensure_default_config(root: &Path) -> Result<()> {
    if has_config(root) {
        return Ok(());
    }
    let path = root.join("config").join(SETTINGS_FILE);
    std::fs::create_dir_all(path.parent().expect("has a parent"))?;
    std::fs::write(&path, DEFAULT_SETTINGS).with_context(|| format!("writing {}", path.display()))
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

pub fn find_root() -> Result<PathBuf> {
    if let Ok(home) = std::env::var("AEGIST_HOME") {
        let root = PathBuf::from(home);
        ensure_default_config(&root)?;
        return Ok(root);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.ancestors().find(|d| has_config(d)) {
            return Ok(dir.to_path_buf());
        }
    }
    let Some(home) = home_dir() else {
        bail!("Couldn't find a home folder. Set AEGIST_HOME to a folder for Aegist to keep its model in.");
    };
    let root = home.join(".aegist");
    ensure_default_config(&root)?;
    Ok(root)
}

impl Settings {
    /// Load from Aegist's home (`find_root`), honoring the env overrides.
    pub fn load() -> Result<Settings> {
        let root = find_root()?;
        let path = std::env::var("AEGIST_SETTINGS").map(PathBuf::from).unwrap_or_else(|_| root.join("config").join(SETTINGS_FILE));
        let mut s = Settings::from_file(&path, &root)?;
        if let Ok(dir) = std::env::var("AEGIST_DATA_DIR") {
            s.data_dir = PathBuf::from(dir);
        }
        Ok(s)
    }

    pub fn from_file(path: &Path, root: &Path) -> Result<Settings> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Settings::from_str(&text, root).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_str(text: &str, root: &Path) -> Result<Settings> {
        let mut s: Settings = serde_yaml::from_str(text)?;
        s.root = root.to_path_buf();
        s.data_dir = root.join(&s.paths.data_dir);
        Ok(s)
    }

    /// Creates the directory if needed.
    pub fn data_path(&self, sub: &str) -> PathBuf {
        let p = if sub.is_empty() { self.data_dir.clone() } else { self.data_dir.join(sub) };
        let _ = std::fs::create_dir_all(&p);
        p
    }

    pub fn settings_path(&self) -> PathBuf {
        self.root.join("config").join(SETTINGS_FILE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_parse_and_are_written_once() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_default_config(tmp.path()).unwrap();
        let s = Settings::from_file(&tmp.path().join("config/aegist.yaml"), tmp.path()).unwrap();
        assert!(s.model.tiers.contains_key("tiny") && s.honesty.samples >= 1 && s.inference.context_scale >= 1);
        std::fs::write(tmp.path().join("config/aegist.yaml"), "edited").unwrap();
        ensure_default_config(tmp.path()).unwrap(); // never overwrites your edits
        assert_eq!(std::fs::read_to_string(tmp.path().join("config/aegist.yaml")).unwrap(), "edited");
    }
}

#[cfg(test)]
pub mod testing {
    //! Settings for tests: the real aegist.yaml, shrunk so a model trains
    //! in a moment, writing into a throwaway directory.
    use super::*;

    pub fn settings(data_dir: &Path) -> Settings {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut s = Settings::from_str(DEFAULT_SETTINGS, &root).unwrap();
        s.data_dir = data_dir.to_path_buf();
        s.model.tiers = BTreeMap::from([(
            "test".to_string(),
            Tier { max_ram_gb: 1e9, n_layer: 1, n_head: 2, d_model: 32, block_size: 48, vocab_size: 320, manual: false },
        )]);
        let t = &mut s.training;
        t.tokens_per_step = 8 * 48;
        t.min_corpus_tokens = 100;
        t.min_new_model_chars = 1000;
        t.warmup_steps = 5;
        t.learning_rate = 3e-3;
        t.eval_every_s = 3600.0;
        t.log_every_s = 3600.0;
        t.checkpoint_every_s = 3600.0;
        s.gpu.device = Device::Cpu; // tests are the same on machines with and without a GPU
        s
    }
}
