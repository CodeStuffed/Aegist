//! The trained model at inference time: what the council actually talks to.
//!
//!   score(prompt, continuations)  how likely each continuation is after the prompt
//!   text_nll(text)                how surprising text is overall (familiarity)
//!   generate(prompt)              sample a continuation, token by token
//!
//! The prompt is run through the model once and its keys/values cached;
//! every continuation then attends to that cache, all of them in one batch.
//! Passing an rng switches dropout on (Monte Carlo dropout): the same
//! question gets a slightly different network each time, which is how the
//! uncertainty check tells a firm answer from a coin flip.

use crate::checkpoint::{self, Meta};
use crate::config::Settings;
use crate::kernels::log_softmax;
use crate::model::{InferModel, KvCache, Layout};
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use anyhow::Result;

#[derive(Debug)]
pub struct NoBrain(pub String);

impl std::fmt::Display for NoBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NoBrain {}

/// A prompt already run through the model: score or continue after it
/// without recomputing it.
#[derive(Clone)]
pub struct Prefix {
    ids: Vec<u32>,
    cache: KvCache,
    /// log-probabilities of the next token
    logp: Vec<f32>,
    /// positions kept free after it for what follows
    reserve: usize,
}

pub struct Brain {
    pub model: InferModel,
    pub tokenizer: Tokenizer,
    pub meta: Meta,
    pub num_params: usize,
    lead: Vec<u32>,
}

impl Brain {
    /// Load the trained model at the precision in settings (inference.precision).
    pub fn load(settings: &Settings) -> Result<Brain> {
        match checkpoint::load(settings)? {
            Some((model, tokenizer, meta)) => {
                let lead = tokenizer.encode("\n");
                let num_params = Layout::new(&model.cfg).total;
                // the full-precision copy is dropped once converted
                let model = model.to_inference(settings.inference.precision);
                Ok(Brain { model, tokenizer, meta, num_params, lead })
            }
            None => Err(NoBrain(format!(
                "No trained model in {}. Train one first: `council train --data <folder of .txt/.md files> --hours 1`, \
                 or `council research --hours 1` to collect text and train on it.",
                checkpoint::brain_dir(settings).display()
            ))
            .into()),
        }
    }

    pub fn describe(&self) -> String {
        let c = &self.model.cfg;
        format!("{}x{} transformer, {:.2}M params, {} weights ({:.1} MB)", c.n_layer, c.d_model,
                self.num_params as f64 / 1e6, self.model.precision, self.model.weight_bytes() as f64 / 1e6)
    }

    fn block(&self) -> usize {
        self.model.cfg.block_size
    }

    /// Lead-in + prompt, cut from the left to leave `reserve` positions free.
    fn fit_prompt(&self, prompt: &str, reserve: usize) -> Vec<u32> {
        let mut ids = self.lead.clone();
        ids.extend(self.tokenizer.encode(prompt));
        let keep = self.block().saturating_sub(reserve).max(1);
        if ids.len() > keep {
            ids.drain(..ids.len() - keep);
        }
        ids
    }

    /// Run the prompt through the model: its cache and the log-probs of the next token.
    fn prefill(&self, ids: &[u32], rng: Option<&mut Rng>) -> (KvCache, Vec<f32>) {
        let c = self.model.cfg.d_model;
        let v = self.model.cfg.vocab_size;
        let out = self.model.infer(&KvCache::empty(&self.model.cfg), ids, 1, ids.len(), rng);
        let mut cache = KvCache::empty(&self.model.cfg);
        for r in 0..ids.len() {
            cache.push_row(&out, r, c);
        }
        let mut last = out.logits[(ids.len() - 1) * v..ids.len() * v].to_vec();
        log_softmax(&mut last);
        (cache, last)
    }

    fn encode_continuations(&self, continuations: &[String], room: usize) -> Vec<Vec<u32>> {
        continuations
            .iter()
            .map(|s| {
                let mut ids = self.tokenizer.encode(s);
                ids.truncate((self.block() / 2).min(room).max(1));
                if ids.is_empty() {
                    ids.push(self.lead[0]);
                }
                ids
            })
            .collect()
    }

    /// Run `prompt` through the model (cut from the left to leave `reserve`
    /// positions free), to score or continue after it.
    pub fn prefix(&self, prompt: &str, reserve: usize, rng: Option<&mut Rng>) -> Prefix {
        let ids = self.fit_prompt(prompt, reserve);
        let (cache, logp) = self.prefill(&ids, rng);
        Prefix { ids, cache, logp, reserve }
    }

    /// `p` followed by `text`, run on top of p's cache.
    pub fn extend(&self, p: &Prefix, text: &str, rng: Option<&mut Rng>) -> Prefix {
        let new = self.tokenizer.encode(text);
        if new.is_empty() {
            return p.clone();
        }
        if p.ids.len() + new.len() + p.reserve > self.block() {
            // doesn't fit on top: start again from the text, cut from the left
            let own = p.ids.strip_prefix(self.lead.as_slice()).unwrap_or(&p.ids);
            return self.prefix(&format!("{}{text}", self.tokenizer.decode(own)), p.reserve, rng);
        }
        let (c, v) = (self.model.cfg.d_model, self.model.cfg.vocab_size);
        let out = self.model.infer(&p.cache, &new, 1, new.len(), rng);
        let mut cache = p.cache.clone();
        for r in 0..new.len() {
            cache.push_row(&out, r, c);
        }
        let mut logp = out.logits[(new.len() - 1) * v..new.len() * v].to_vec();
        log_softmax(&mut logp);
        let mut ids = p.ids.clone();
        ids.extend(new);
        Prefix { ids, cache, logp, reserve: p.reserve }
    }

    /// Mean log-probability per token of each continuation after `prompt`.
    pub fn score(&self, prompt: &str, continuations: &[String], mut rng: Option<&mut Rng>) -> Vec<f32> {
        let longest = self.encode_continuations(continuations, self.block()).iter().map(Vec::len).max().unwrap_or(1);
        let p = self.prefix(prompt, longest, rng.as_deref_mut());
        self.score_after(&p, continuations, rng)
    }

    /// Mean log-probability per token of each continuation after `p`.
    pub fn score_after(&self, p: &Prefix, continuations: &[String], rng: Option<&mut Rng>) -> Vec<f32> {
        let v = self.model.cfg.vocab_size;
        let conts = self.encode_continuations(continuations, self.block() - p.ids.len());
        let longest = conts.iter().map(Vec::len).max().unwrap_or(1);
        let (cache, first) = (&p.cache, &p.logp);

        // Every continuation minus its last token, padded to the same length;
        // the padding sits after the real tokens, so causal attention never sees it.
        let ext_len = longest - 1;
        let ext = if ext_len > 0 {
            let mut toks = vec![0u32; conts.len() * ext_len];
            for (i, c) in conts.iter().enumerate() {
                toks[i * ext_len..i * ext_len + c.len() - 1].copy_from_slice(&c[..c.len() - 1]);
            }
            Some(self.model.infer(cache, &toks, conts.len(), ext_len, rng))
        } else {
            None
        };
        conts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut total = first[c[0] as usize];
                for j in 1..c.len() {
                    let ext = ext.as_ref().expect("longer continuations have an extension");
                    let row = i * ext_len + j - 1;
                    let mut lp = ext.logits[row * v..(row + 1) * v].to_vec();
                    log_softmax(&mut lp);
                    total += lp[c[j] as usize];
                }
                total / c.len() as f32
            })
            .collect()
    }

    /// Mean negative log-likelihood per token of `text` (lower = more familiar).
    pub fn text_nll(&self, text: &str) -> f32 {
        let v = self.model.cfg.vocab_size;
        let mut ids = self.lead.clone();
        ids.extend(self.tokenizer.encode(text));
        ids.truncate(self.block());
        if ids.len() < 2 {
            return 0.0;
        }
        let out = self.model.infer(&KvCache::empty(&self.model.cfg), &ids, 1, ids.len(), None);
        let n = ids.len() - 1;
        let total: f32 = (0..n)
            .map(|r| {
                let mut lp = out.logits[r * v..(r + 1) * v].to_vec();
                log_softmax(&mut lp);
                -lp[ids[r + 1] as usize]
            })
            .sum();
        total / n as f32
    }

    /// Sample a continuation. Stops at a paragraph break or after two sentences.
    pub fn generate(&self, prompt: &str, max_new: usize, temperature: f32, top_k: usize, rng: &mut Rng) -> String {
        let p = self.prefix(prompt, max_new.min(self.block() - 1), None);
        self.generate_after(&p, max_new, temperature, top_k, rng)
    }

    /// `generate`, continuing after `p`.
    pub fn generate_after(&self, p: &Prefix, max_new: usize, temperature: f32, top_k: usize, rng: &mut Rng) -> String {
        let (c, v) = (self.model.cfg.d_model, self.model.cfg.vocab_size);
        let max_new = max_new.min(self.block() - p.ids.len());
        let (mut cache, mut logp) = (p.cache.clone(), p.logp.clone());
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let next = sample(&logp, temperature, top_k, rng);
            out.push(next);
            let text = self.tokenizer.decode(&out);
            if text.trim().contains("\n\n") || text.matches(['.', '!', '?']).count() >= 2 || cache.len + 1 >= self.block() {
                break;
            }
            let step = self.model.infer(&cache, &[next], 1, 1, None);
            cache.push_row(&step, 0, c);
            logp = step.logits[..v].to_vec();
            log_softmax(&mut logp);
        }
        let text = self.tokenizer.decode(&out);
        let first_para = text.trim().split("\n\n").next().unwrap_or("").to_string();
        first_para.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Sample from log-probs with temperature and top-k (temperature 0 = greedy).
pub fn sample(logp: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return logp.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    }
    let mut idx: Vec<usize> = (0..logp.len()).collect();
    let k = if top_k == 0 || top_k > logp.len() { logp.len() } else { top_k };
    idx.select_nth_unstable_by(k - 1, |&a, &b| logp[b].total_cmp(&logp[a]));
    idx.truncate(k);
    let max = idx.iter().map(|&i| logp[i]).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = idx.iter().map(|&i| ((logp[i] - max) / temperature).exp()).collect();
    let mut r = rng.uniform() * weights.iter().sum::<f32>();
    for (w, &i) in weights.iter().zip(&idx) {
        r -= w;
        if r <= 0.0 {
            return i as u32;
        }
    }
    idx[k - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::tests::trained;

    #[test]
    fn no_model_yet_explains_what_to_do() {
        let tmp = tempfile::tempdir().unwrap();
        let s = crate::config::testing::settings(tmp.path());
        let err = Brain::load(&s).err().unwrap();
        assert!(err.downcast_ref::<NoBrain>().is_some() && err.to_string().contains("council train"));
    }

    #[test]
    fn scores_generates_and_senses_familiarity() {
        let (_tmp, s) = trained(60);
        let brain = Brain::load(&s).unwrap();
        let conts = vec![" glass.".to_string(), " zebra.".to_string()];
        let scores = brain.score("Light is refracted when it passes from air into", &conts, None);
        assert!(scores[0] > scores[1], "{scores:?}");
        // batched scoring equals scoring one at a time
        let one = brain.score("Light is refracted when it passes from air into", &conts[1..], None);
        assert!((one[0] - scores[1]).abs() < 1e-4);
        assert!(brain.text_nll("The rays of light bend toward the perpendicular.") + 0.5
            < brain.text_nll("Zxq vbnm qwpl kjhg tyvr."));
        let text = brain.generate("White light is", 8, 0.8, 40, &mut Rng::new(0));
        assert!(!text.contains("\n\n"));
        // dropout on: the same question scores slightly differently
        let noisy: Vec<f32> = (0..4).map(|i| brain.score("The market", &conts[..1], Some(&mut Rng::new(i)))[0]).collect();
        assert!(noisy.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6));
        assert_eq!(brain.score("The market", &conts[..1], None), brain.score("The market", &conts[..1], None));
    }

    #[test]
    fn prefixes_give_the_same_scores_as_whole_prompts() {
        let (_tmp, s) = trained(30);
        let brain = Brain::load(&s).unwrap();
        let conts = vec![" glass.".to_string(), " a prism of light.".to_string()];
        let whole = brain.score("Light is refracted when it passes from air into", &conts, None);
        let p = brain.prefix("Light is refracted when it passes from air", 16, None);
        let extended = brain.extend(&p, " into", None);
        let after = brain.score_after(&extended, &conts, None);
        assert!(whole.iter().zip(&after).all(|(a, b)| (a - b).abs() < 1e-4), "{whole:?} vs {after:?}");
        // the same sample from a shared prefix as from the whole prompt
        let a = brain.generate("White light is", 8, 0.8, 40, &mut Rng::new(3));
        let b = brain.generate_after(&brain.prefix("White light is", 8, None), 8, 0.8, 40, &mut Rng::new(3));
        assert_eq!(a, b);
    }

    #[test]
    fn sampling_respects_top_k_and_greedy() {
        let logp = vec![-5.0, -0.1, -3.0, -0.2];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logp, 0.0, 0, &mut rng), 1);
        assert!((0..200).all(|_| matches!(sample(&logp, 1.0, 2, &mut rng), 1 | 3)));
    }
}
