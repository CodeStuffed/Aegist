//! Saving and loading the trained model. Layout of data/brain/:
//!   brain.json   format tag, model config, tokenizer merges, training stats
//!   model.bin    every weight, little-endian f32, in Layout order
//!   optim.bin    AdamW step count and moments, so training resumes smoothly

use crate::config::Settings;
use crate::model::{Model, ModelConfig};
use crate::optim::AdamW;
use crate::tokenizer::Tokenizer;
use crate::util::{bytes_to_f32s, f32s_to_bytes, iso_now, write_atomic};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const FORMAT: &str = "council-rust-1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryPoint {
    pub step: u64,
    pub val_loss: f32,
    pub val_bpb: f32,
    pub at: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    pub steps: u64,
    pub tokens_seen: u64,
    pub train_seconds: f64,
    pub train_loss: Option<f32>,
    pub val_loss: Option<f32>,
    /// Held-out bits per byte of text: comparable across tokenizers and models.
    pub val_bpb: Option<f32>,
    pub created_at: String,
    pub updated_at: String,
    pub history: Vec<HistoryPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub format: String,
    pub config: ModelConfig,
    pub merges: Vec<(u32, u32)>,
    pub stats: Stats,
}

pub fn brain_dir(settings: &Settings) -> PathBuf {
    settings.data_path("brain")
}

pub fn exists(settings: &Settings) -> bool {
    brain_dir(settings).join("brain.json").is_file()
}

pub fn save(settings: &Settings, model: &Model, meta: &mut Meta, optim: Option<&AdamW>) -> Result<()> {
    let dir = brain_dir(settings);
    write_atomic(&dir.join("model.bin"), &f32s_to_bytes(&model.params))?;
    if let Some(o) = optim {
        let mut bytes = o.t.to_le_bytes().to_vec();
        bytes.extend(f32s_to_bytes(&o.m));
        bytes.extend(f32s_to_bytes(&o.v));
        write_atomic(&dir.join("optim.bin"), &bytes)?;
    }
    meta.stats.updated_at = iso_now();
    write_atomic(&dir.join("brain.json"), &serde_json::to_vec(meta)?)
}

/// The saved model's size, without reading its weights (None if there's no model).
pub fn saved_config(settings: &Settings) -> Option<crate::model::ModelConfig> {
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(brain_dir(settings).join("brain.json")).ok()?).ok()?;
    serde_json::from_value(raw.get("config")?.clone()).ok()
}

/// The saved model, or None if there isn't one yet.
pub fn load(settings: &Settings) -> Result<Option<(Model, Tokenizer, Meta)>> {
    let dir = brain_dir(settings);
    let meta_path = dir.join("brain.json");
    if !meta_path.is_file() {
        return Ok(None);
    }
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta_path)?)?;
    if raw.get("format").and_then(|f| f.as_str()) != Some(FORMAT) {
        bail!("The model in {} was made by an older version of council-engine and can't be loaded. \
               Delete that folder and train again - your corpus and knowledge base are kept.", dir.display());
    }
    let meta: Meta = serde_json::from_value(raw).context("reading brain.json")?;
    meta.config.validate()?;
    let params = bytes_to_f32s(&std::fs::read(dir.join("model.bin")).context("reading model.bin")?);
    let expected = crate::model::Layout::new(&meta.config).total;
    if params.len() != expected {
        bail!("model.bin has {} weights but the config needs {expected}; the checkpoint is damaged.", params.len());
    }
    let tokenizer = Tokenizer::new(meta.merges.clone());
    Ok(Some((Model::from_params(meta.config.clone(), params), tokenizer, meta)))
}

/// Restore optimizer state if it matches this model; otherwise start fresh.
pub fn load_optimizer(settings: &Settings, opt: &mut AdamW) {
    let Ok(bytes) = std::fs::read(brain_dir(settings).join("optim.bin")) else { return };
    let n = opt.m.len();
    if bytes.len() != 8 + 8 * n {
        return;
    }
    opt.t = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    opt.m = bytes_to_f32s(&bytes[8..8 + 4 * n]);
    opt.v = bytes_to_f32s(&bytes[8 + 4 * n..]);
}
