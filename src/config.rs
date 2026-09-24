//! Settings (config/settings.yaml) and personas (config/personas/*.yaml).
//!
//! The project root is found by walking up from the current directory (then
//! from the executable) until config/settings.yaml turns up, so `council`
//! works from anywhere inside the project. Run anywhere else, it uses
//! ~/.council, writing the built-in default config there on first run so
//! a downloaded binary works on its own. Overrides:
//!   COUNCIL_HOME      project root
//!   COUNCIL_SETTINGS  an alternate settings.yaml
//!   COUNCIL_DATA_DIR  where runtime data is written

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
    pub eval_batches: usize,
    pub eval_every_s: f64,
    pub checkpoint_every_s: f64,
    pub log_every_s: f64,
    pub tokenizer_train_chars: usize,
    pub min_corpus_tokens: usize,
    pub min_new_model_chars: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GenerateSettings {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Thresholds {
    pub high: f32,
    pub medium: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Familiarity {
    pub medium_above: f32,
    pub low_above: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CouncilSettings {
    pub generate: GenerateSettings,
    pub neutral_prompt: String,
    pub confidence_thresholds: Thresholds,
    pub mixed_margin: f32,
    pub familiarity: Familiarity,
    pub min_tokens_trained: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RouterSettings {
    pub money_keywords: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MemorySettings {
    pub top_k: usize,
    pub min_relevance: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UncertaintySettings {
    pub enabled: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResearchSettings {
    pub wikipedia_api: String,
    pub articles_per_topic: usize,
    pub max_topics_per_hour: usize,
    pub refresh_hours: f64,
    pub train_minutes_per_topic: f64,
    pub min_passage_chars: usize,
    pub seed_topics: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PathSettings {
    pub data_dir: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    pub model: ModelSettings,
    pub training: TrainingSettings,
    pub council: CouncilSettings,
    pub router: RouterSettings,
    pub memory: MemorySettings,
    pub uncertainty: UncertaintySettings,
    pub research: ResearchSettings,
    pub paths: PathSettings,
    /// Project root (holds config/).
    #[serde(skip)]
    pub root: PathBuf,
    /// Resolved data directory (paths.data_dir, or COUNCIL_DATA_DIR).
    #[serde(skip)]
    pub data_dir: PathBuf,
}

fn has_config(dir: &Path) -> bool {
    dir.join("config").join("settings.yaml").is_file()
}

/// The default config, built into the binary.
const DEFAULT_FILES: &[(&str, &str)] = &[
    ("settings.yaml", include_str!("../config/settings.yaml")),
    ("personas/README.md", include_str!("../config/personas/README.md")),
    ("personas/believer.yaml", include_str!("../config/personas/believer.yaml")),
    ("personas/skeptic.yaml", include_str!("../config/personas/skeptic.yaml")),
    ("personas/investor.yaml", include_str!("../config/personas/investor.yaml")),
    ("personas/judge.yaml", include_str!("../config/personas/judge.yaml")),
];

/// Write the built-in config under `root/config/` unless one is already there.
pub fn ensure_default_config(root: &Path) -> Result<()> {
    if has_config(root) {
        return Ok(());
    }
    for (name, text) in DEFAULT_FILES {
        let path = root.join("config").join(name);
        std::fs::create_dir_all(path.parent().expect("has a parent"))?;
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

pub fn find_root() -> Result<PathBuf> {
    if let Ok(home) = std::env::var("COUNCIL_HOME") {
        let root = PathBuf::from(home);
        ensure_default_config(&root)?;
        return Ok(root);
    }
    let mut starts = vec![std::env::current_dir()?];
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe);
    }
    for start in starts {
        if let Some(dir) = start.ancestors().find(|d| has_config(d)) {
            return Ok(dir.to_path_buf());
        }
    }
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        bail!("Couldn't find config/settings.yaml or a home folder. Set COUNCIL_HOME to a folder for council to use.");
    };
    let root = PathBuf::from(home).join(".council");
    ensure_default_config(&root)?;
    Ok(root)
}

impl Settings {
    /// Load from the project found by `find_root`, honoring the env overrides.
    pub fn load() -> Result<Settings> {
        let root = find_root()?;
        let path = std::env::var("COUNCIL_SETTINGS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("config").join("settings.yaml"));
        let mut s = Settings::from_file(&path, &root)?;
        if let Ok(dir) = std::env::var("COUNCIL_DATA_DIR") {
            s.data_dir = PathBuf::from(dir);
        }
        Ok(s)
    }

    pub fn from_file(path: &Path, root: &Path) -> Result<Settings> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut s: Settings = serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
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

    pub fn persona_dir(&self) -> PathBuf {
        self.root.join("config").join("personas")
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Persona {
    pub name: String,
    pub role: String,
    pub lead_in: String,
    #[serde(default)]
    pub probes: Vec<String>,
    #[serde(default)]
    pub counter_probes: Vec<String>,
}

pub fn load_persona(settings: &Settings, name: &str) -> Result<Persona> {
    let dir = settings.persona_dir();
    let path = dir.join(format!("{name}.yaml"));
    if !path.is_file() {
        let mut available: Vec<String> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
                    .filter(|s| s != "README")
                    .collect()
            })
            .unwrap_or_default();
        available.sort();
        bail!("No persona {name:?}; available: {available:?}");
    }
    let text = std::fs::read_to_string(&path)?;
    serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_config_is_written_once_and_loads() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_default_config(tmp.path()).unwrap();
        let s = Settings::from_file(&tmp.path().join("config/settings.yaml"), tmp.path()).unwrap();
        assert!(!s.model.tiers.is_empty());
        assert_eq!(load_persona(&s, "investor").unwrap().counter_probes.len(), 3);
        std::fs::write(tmp.path().join("config/settings.yaml"), "edited").unwrap();
        ensure_default_config(tmp.path()).unwrap(); // never overwrites your edits
        assert_eq!(std::fs::read_to_string(tmp.path().join("config/settings.yaml")).unwrap(), "edited");
        assert!(load_persona(&s, "oracle").unwrap_err().to_string().contains("believer"));
    }
}

#[cfg(test)]
pub mod testing {
    //! Settings for tests: the real settings.yaml, shrunk so a model trains
    //! in a moment, writing into a throwaway directory.
    use super::*;

    pub fn settings(data_dir: &Path) -> Settings {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut s = Settings::from_file(&root.join("config/settings.yaml"), &root).unwrap();
        s.data_dir = data_dir.to_path_buf();
        s.model.tiers = BTreeMap::from([(
            "test".to_string(),
            Tier { max_ram_gb: 1e9, n_layer: 1, n_head: 2, d_model: 32, block_size: 48, vocab_size: 320 },
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
        s.council.generate.max_new_tokens = 12;
        s.council.min_tokens_trained = 0;
        s
    }
}
