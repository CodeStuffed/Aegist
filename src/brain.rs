//! The trained model at inference time: it writes code.
//!
//!   generate(prompt)        continue a prompt, token by token, streaming
//!   generate_many(prompt)   several candidates from one prompt, in parallel,
//!                           sharing the prompt's cached keys and values
//!
//! Every token comes with the probability the model gave it, so callers can
//! show where it was unsure and refuse to stand behind low-confidence code.
//! The prompt's own familiarity (how surprising the model found it) is
//! measured for free while reading it.
//!
//! Speed: prompts are read in one batched pass that computes the output
//! layer only where needed, and generation guesses ahead by copying from
//! the prompt (prompt-lookup decoding): the guess is checked in the same
//! pass as the next token, so every token is still exactly what the model
//! would have sampled, but repeated code comes out several tokens per pass.

use crate::checkpoint::{self, Meta};
use crate::config::Settings;
use crate::corpus::{EOT, FILE, MID, PRE, SUF};
use crate::kernels::{log_softmax, Rope};
use crate::model::{InferModel, KvCache, Layout, Model};
use crate::quant::Precision;
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// There's no trained model yet.
#[derive(Debug)]
pub struct NoBrain(pub String);

impl std::fmt::Display for NoBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NoBrain {}

#[derive(Clone, Debug)]
pub struct GenOptions {
    pub max_new: usize,
    /// 0 = always the most likely token.
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub speculative: bool,
    /// Stop early when the last few tokens' average probability falls below
    /// this: the model has lost track and is guessing (0 = never).
    pub give_up_below: f32,
}

impl GenOptions {
    pub fn from_settings(s: &Settings) -> Self {
        GenOptions { max_new: s.inference.max_new_tokens, temperature: s.inference.temperature, top_p: s.inference.top_p, seed: 0,
                     speculative: s.inference.speculative, give_up_below: s.honesty.give_up_below }
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The model ended the file (or the gap it was filling).
    EndOfFile,
    /// The caller's stop rule matched (e.g. the function ended).
    Rule,
    MaxTokens,
    /// No room left in the context.
    Context,
    /// The caller asked to stop.
    Cancelled,
    /// The model lost track (a run of tokens it barely believed), so it
    /// stopped instead of guessing on.
    Unsure,
}

/// Tokens in the stretch that early give-up and `Generation::weakest` look at.
pub const STRETCH: usize = 8;

#[derive(Clone, Debug)]
pub struct Generation {
    pub text: String,
    /// (start, end, probability): the byte range of `text` each token wrote
    /// and how likely the model thought it was.
    pub spans: Vec<(usize, usize, f32)>,
    pub stop: Stop,
    /// Mean negative log-likelihood per token of the end of the prompt:
    /// how familiar the code around the request is to the model.
    pub prompt_nll: f32,
    pub prompt_tokens: usize,
    /// Tokens that came from a correct guess (speculation).
    pub guessed: usize,
    pub seconds: f64,
}

impl Generation {
    pub fn tokens(&self) -> usize {
        self.spans.len()
    }

    /// Average probability of the tokens written (0-1).
    pub fn mean_prob(&self) -> f32 {
        if self.spans.is_empty() {
            return 0.0;
        }
        self.spans.iter().map(|s| s.2).sum::<f32>() / self.spans.len() as f32
    }

    /// Mean log-probability per token (for ranking candidates).
    pub fn mean_logprob(&self) -> f32 {
        if self.spans.is_empty() {
            return f32::NEG_INFINITY;
        }
        self.spans.iter().map(|s| s.2.max(1e-9).ln()).sum::<f32>() / self.spans.len() as f32
    }

    /// Tokens the model gave less than `below` probability.
    pub fn uncertain(&self, below: f32) -> usize {
        self.spans.iter().filter(|s| s.2 < below).count()
    }

    /// The lowest average probability over any `STRETCH` tokens in a row:
    /// a short made-up stretch that a good average would hide (1.0 if empty).
    pub fn weakest(&self) -> f32 {
        let probs: Vec<f32> = self.spans.iter().map(|s| s.2).collect();
        if probs.is_empty() {
            return 1.0;
        }
        let k = probs.len().min(STRETCH);
        let mut sum: f32 = probs[..k].iter().sum();
        let mut low = sum;
        for i in k..probs.len() {
            sum += probs[i] - probs[i - k];
            low = low.min(sum);
        }
        (low / k as f32).clamp(0.0, 1.0)
    }

    pub fn tokens_per_second(&self) -> f64 {
        self.spans.len() as f64 / self.seconds.max(1e-9)
    }
}

/// Decides when generated text is complete: given everything written so
/// far, Some(n) stops and keeps the first n bytes.
pub type StopRule<'a> = &'a (dyn Fn(&str) -> Option<usize> + Sync);

pub fn never_stop(_: &str) -> Option<usize> {
    None
}

pub struct Brain {
    pub model: InferModel,
    pub tokenizer: Tokenizer,
    pub meta: Meta,
    pub num_params: usize,
    /// How many times the trained context the model may read.
    pub context_scale: usize,
    markers: Vec<u32>,
}

/// A prompt already read: its cached keys/values, the next token's
/// log-probabilities, and how familiar it was.
struct Read {
    ids: Vec<u32>,
    cache: Arc<KvCache>,
    logp: Vec<f32>,
    nll: f32,
    rope: Rope,
    max_new: usize,
}

impl Brain {
    /// Load the trained model at the precision in settings (inference.precision).
    pub fn load(settings: &Settings) -> Result<Brain> {
        Self::load_at(settings, settings.inference.precision)
    }

    pub fn load_at(settings: &Settings, precision: Precision) -> Result<Brain> {
        match checkpoint::load(settings)? {
            Some((model, tokenizer, meta)) => Ok(Self::from_parts(&model, tokenizer, meta, precision, settings.inference.context_scale)),
            None => Err(NoBrain(format!(
                "There's no trained model yet (looked in {}). Give it code to learn from, then train it:\n  \
                 aegist learn <folder or git URL>     (or: aegist learn --pack python)\n  aegist train --hours 1",
                checkpoint::brain_dir(settings).display()
            ))
            .into()),
        }
    }

    pub fn from_parts(model: &Model, tokenizer: Tokenizer, meta: Meta, precision: Precision, context_scale: usize) -> Brain {
        let num_params = Layout::new(&model.cfg).total;
        let markers = [FILE, PRE, SUF, MID, EOT].iter().map(|&c| c as u32).collect();
        Brain { model: model.to_inference(precision), tokenizer, meta, num_params, context_scale: context_scale.max(1), markers }
    }

    pub fn describe(&self) -> String {
        let c = &self.model.cfg;
        format!("{}x{} transformer, {:.1}M params, {} weights ({:.0} MB), {}-token context (reads up to {})",
            c.n_layer, c.d_model, self.num_params as f64 / 1e6, self.model.precision, self.model.weight_bytes() as f64 / 1e6,
            c.block_size, self.max_context())
    }

    /// The most tokens a prompt plus its answer can take.
    pub fn max_context(&self) -> usize {
        self.model.cfg.block_size * self.context_scale
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.encode(text).len()
    }

    /// The model's typical loss on code it hasn't trained on (held-out),
    /// for judging how familiar a prompt is.
    pub fn typical_nll(&self) -> Option<f32> {
        self.meta.stats.val_loss
    }

    fn is_marker(&self, id: u32) -> bool {
        self.markers.contains(&id)
    }

    /// Encode, leaving room for `max_new` tokens (fewer if the prompt is
    /// long; the prompt is cut from the left as a last resort).
    fn fit(&self, prompt: &str, max_new: usize) -> (Vec<u32>, usize) {
        let mut ids = self.tokenizer.encode(prompt);
        if ids.is_empty() {
            ids.push(FILE as u32);
        }
        let ctx = self.max_context();
        let max_new = max_new.min(ctx.saturating_sub(ids.len()).max(ctx / 8)).max(1);
        if ids.len() + max_new > ctx {
            ids.drain(..ids.len() + max_new - ctx);
        }
        (ids, max_new)
    }

    /// Read the prompt in one pass.
    fn read(&self, prompt: &str, max_new: usize) -> Read {
        let (ids, max_new) = self.fit(prompt, max_new);
        let (c, v) = (self.model.cfg.d_model, self.model.cfg.vocab_size);
        let rope = self.model.rope_for(ids.len() + max_new);
        // logits for the last few hundred prompt tokens: the familiarity
        // measure, plus the next token's distribution
        let rows = ids.len().min(256);
        let out = self.model.infer_with(&rope, &KvCache::empty(&self.model.cfg), &ids, 1, ids.len(), rows, None);
        let mut cache = KvCache::empty(&self.model.cfg);
        cache.push_all(&out, c);
        let mut nll = 0.0;
        for r in 0..rows - 1 {
            let mut lp = out.logits[r * v..(r + 1) * v].to_vec();
            log_softmax(&mut lp);
            nll -= lp[ids[ids.len() - rows + r + 1] as usize];
        }
        let nll = if rows > 1 { nll / (rows - 1) as f32 } else { 0.0 };
        let mut logp = out.logits[(rows - 1) * v..rows * v].to_vec();
        log_softmax(&mut logp);
        Read { ids, cache: Arc::new(cache), logp, nll, rope, max_new }
    }

    /// Continue `prompt`. `on_text` gets each new piece of text and the
    /// probability of the token that wrote it; returning false stops.
    pub fn generate(&self, prompt: &str, opts: &GenOptions, stop: StopRule, on_text: &mut dyn FnMut(&str, f32) -> bool) -> Generation {
        let start = Instant::now();
        let read = self.read(prompt, opts.max_new);
        let mut g = self.continue_read(&read, opts, stop, on_text);
        g.seconds = start.elapsed().as_secs_f64();
        g
    }

    /// One candidate per entry of `opts`, all continuing `prompt` (read once),
    /// written in parallel.
    pub fn generate_many(&self, prompt: &str, opts: &[GenOptions], stop: StopRule) -> Vec<Generation> {
        self.generate_many_streaming(prompt, opts, stop, &|_, _, _| true)
    }

    /// `generate_many`, passing each candidate's new text to `on_text`
    /// (candidate index, text, probability); returning false stops that one.
    pub fn generate_many_streaming(&self, prompt: &str, opts: &[GenOptions], stop: StopRule,
                                   on_text: &(dyn Fn(usize, &str, f32) -> bool + Sync)) -> Vec<Generation> {
        let start = Instant::now();
        let longest = opts.iter().map(|o| o.max_new).max().unwrap_or(1);
        let read = self.read(prompt, longest);
        let read_s = start.elapsed().as_secs_f64();
        opts.par_iter()
            .enumerate()
            .map(|(i, o)| {
                let t = Instant::now();
                let mut g = self.continue_read(&read, o, stop, &mut |text, p| on_text(i, text, p));
                g.seconds = read_s + t.elapsed().as_secs_f64();
                g
            })
            .collect()
    }

    fn continue_read(&self, read: &Read, opts: &GenOptions, stop: StopRule, on_text: &mut dyn FnMut(&str, f32) -> bool) -> Generation {
        let (c, v) = (self.model.cfg.d_model, self.model.cfg.vocab_size);
        let max_new = opts.max_new.min(read.max_new);
        let mut rng = Rng::new(opts.seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut cache = KvCache::fork(&read.cache);
        let mut seq = read.ids.clone(); // prompt + accepted tokens, for guessing
        let mut text = String::new();
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut spans: Vec<(usize, usize, f32)> = Vec::new();
        let mut guessed = 0;
        let mut stop_reason = Stop::MaxTokens;

        // Take token `id` (probability `p`) into the output. Returns why to
        // stop, if it's time to.
        let mut emit = |id: u32, p: f32, text: &mut String, spans: &mut Vec<(usize, usize, f32)>| -> Option<Stop> {
            if self.is_marker(id) {
                return Some(Stop::EndOfFile);
            }
            pending_bytes.extend_from_slice(self.tokenizer.bytes(id));
            let valid = match std::str::from_utf8(&pending_bytes) {
                Ok(s) => s.len(),
                Err(e) if e.error_len().is_some() => pending_bytes.len(), // invalid, not just unfinished
                Err(e) => e.valid_up_to(),
            };
            let piece = String::from_utf8_lossy(&pending_bytes[..valid]).into_owned();
            pending_bytes.drain(..valid);
            let begin = text.len();
            text.push_str(&piece);
            spans.push((begin, text.len(), p));
            let cut = stop(text).or_else(|| if piece.contains('\n') { looping(text) } else { None });
            if let Some(cut) = cut {
                let cut = cut.min(text.len());
                text.truncate(cut);
                spans.retain(|s| s.0 < cut);
                if let Some(last) = spans.last_mut() {
                    last.1 = last.1.min(cut);
                }
                return Some(Stop::Rule);
            }
            if !on_text(&piece, p) {
                return Some(Stop::Cancelled);
            }
            if opts.give_up_below > 0.0 && spans.len() >= STRETCH
                && spans[spans.len() - STRETCH..].iter().map(|s| s.2).sum::<f32>() / (STRETCH as f32) < opts.give_up_below {
                return Some(Stop::Unsure);
            }
            None
        };

        let sample_from = |logp: &[f32], rng: &mut Rng| -> (u32, f32) {
            let id = sample(logp, opts.temperature, opts.top_p, rng);
            (id, logp[id as usize].exp())
        };
        let (mut pending, mut pending_p) = sample_from(&read.logp, &mut rng);
        let mut written = 0;
        'outer: loop {
            if let Some(s) = emit(pending, pending_p, &mut text, &mut spans) {
                stop_reason = s;
                break;
            }
            seq.push(pending);
            written += 1;
            if written >= max_new {
                break;
            }
            if cache.total_len() + 1 >= read.rope.max_pos() {
                stop_reason = Stop::Context;
                break;
            }
            let room = (max_new - written).min(read.rope.max_pos() - cache.total_len() - 1);
            let draft: Vec<u32> = if opts.speculative { guess(&seq, room.min(8)) } else { Vec::new() };
            let mut input = vec![pending];
            input.extend_from_slice(&draft);
            let out = self.model.infer_with(&read.rope, &cache, &input, 1, input.len(), input.len(), None);
            cache.push_row(&out, 0, c);
            let row = |r: usize| {
                let mut lp = out.logits[r * v..(r + 1) * v].to_vec();
                log_softmax(&mut lp);
                lp
            };
            for (j, &d) in draft.iter().enumerate() {
                let lp = row(j);
                let (next, p) = sample_from(&lp, &mut rng);
                if next != d {
                    (pending, pending_p) = (next, p);
                    continue 'outer;
                }
                // the guess is what the model would have written: keep it
                if let Some(s) = emit(d, p, &mut text, &mut spans) {
                    stop_reason = s;
                    break 'outer;
                }
                guessed += 1;
                seq.push(d);
                cache.push_row(&out, j + 1, c);
                written += 1;
                if written >= max_new {
                    break 'outer;
                }
            }
            (pending, pending_p) = sample_from(&row(draft.len()), &mut rng);
        }
        Generation { text, spans, stop: stop_reason, prompt_nll: read.nll, prompt_tokens: read.ids.len(), guessed, seconds: 0.0 }
    }

    /// How likely the model finds each of `options` as the text right after
    /// `prompt`: (total log-probability, tokens). The prompt is read once and
    /// every option continues its cached keys and values, in parallel, so
    /// asking about ten options costs little more than asking about one.
    pub fn score_continuations(&self, prompt: &str, options: &[String]) -> Vec<(f32, usize)> {
        let encoded: Vec<Vec<u32>> = options.iter().map(|o| self.tokenizer.encode(o)).collect();
        let longest = encoded.iter().map(|e| e.len()).max().unwrap_or(1).max(1);
        let read = self.read(prompt, longest);
        let v = self.model.cfg.vocab_size;
        encoded
            .par_iter()
            .map(|ids| {
                let ids = &ids[..ids.len().min(read.max_new)];
                let Some(&first) = ids.first() else { return (0.0, 0) };
                let mut total = read.logp[first as usize];
                if ids.len() > 1 {
                    let n = ids.len() - 1;
                    let cache = KvCache::fork(&read.cache);
                    let out = self.model.infer_with(&read.rope, &cache, &ids[..n], 1, n, n, None);
                    for r in 0..n {
                        let mut lp = out.logits[r * v..(r + 1) * v].to_vec();
                        log_softmax(&mut lp);
                        total += lp[ids[r + 1] as usize];
                    }
                }
                (total, ids.len())
            })
            .collect()
    }

    /// Mean negative log-likelihood per token of `text` (lower = more familiar).
    pub fn text_nll(&self, text: &str) -> f32 {
        let v = self.model.cfg.vocab_size;
        let mut ids = self.tokenizer.encode(text);
        ids.truncate(self.model.cfg.block_size);
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
}

/// Where generated text starts going round in circles, if it does: the
/// same block of lines (one line three times over, or two to four lines
/// twice) right after itself. Returns where the repeat begins. Real code
/// repeats short lines like `}` or `pass`, so only substantial lines count.
pub fn looping(text: &str) -> Option<usize> {
    let body = text.strip_suffix('\n')?;
    let mut starts = vec![0];
    starts.extend(body.match_indices('\n').map(|(i, _)| i + 1));
    let lines: Vec<&str> = body.split('\n').collect();
    let n = lines.len();
    let solid = |l: &str| l.trim().len() > 3;
    for k in 1..=4 {
        let times = if k == 1 { 3 } else { 2 };
        if n < k * times {
            continue;
        }
        let block = &lines[n - k..];
        if !block.iter().any(|l| solid(l)) {
            continue;
        }
        if (1..times).all(|t| &lines[n - k * (t + 1)..n - k * t] == block) {
            return Some(starts[n - k * (times - 1)]);
        }
    }
    None
}

/// Prompt-lookup guess: find the most recent earlier place where the last
/// few tokens appeared, and propose what followed them there.
pub fn guess(seq: &[u32], max_len: usize) -> Vec<u32> {
    if max_len == 0 {
        return Vec::new();
    }
    let len = seq.len();
    for n in (1..=3).rev() {
        if len <= n {
            continue;
        }
        let tail = &seq[len - n..];
        // search back through the last 8192 tokens, most recent first
        let lo = len.saturating_sub(8192);
        let mut i = len - n;
        while i > lo {
            i -= 1;
            if &seq[i..i + n] == tail {
                let from = i + n;
                let to = (from + max_len).min(len);
                if to > from {
                    return seq[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// Sample from log-probs with temperature and nucleus (top-p) filtering;
/// temperature 0 = the most likely token.
pub fn sample(logp: &[f32], temperature: f32, top_p: f32, rng: &mut Rng) -> u32 {
    let argmax = || logp.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    if temperature <= 0.0 {
        return argmax();
    }
    // the few hundred most likely tokens hold practically all the mass
    let k = logp.len().min(256);
    let mut idx: Vec<usize> = (0..logp.len()).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| logp[b].total_cmp(&logp[a]));
        idx.truncate(k);
    }
    idx.sort_unstable_by(|&a, &b| logp[b].total_cmp(&logp[a]));
    let max = logp[idx[0]];
    let mut weights: Vec<f32> = idx.iter().map(|&i| ((logp[i] - max) / temperature).exp()).collect();
    let total: f32 = weights.iter().sum();
    let mut acc = 0.0;
    let mut keep = weights.len();
    for (n, w) in weights.iter().enumerate() {
        acc += w / total;
        if acc >= top_p {
            keep = n + 1;
            break;
        }
    }
    weights.truncate(keep);
    let mut r = rng.uniform() * weights.iter().sum::<f32>();
    for (w, &i) in weights.iter().zip(&idx) {
        r -= w;
        if r <= 0.0 {
            return i as u32;
        }
    }
    idx[keep - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::tests::{trained, CORPUS};

    fn opts(max_new: usize, temperature: f32, speculative: bool) -> GenOptions {
        GenOptions { max_new, temperature, top_p: 0.95, seed: 7, speculative, give_up_below: 0.0 }
    }

    #[test]
    fn no_model_yet_explains_what_to_do() {
        let tmp = tempfile::tempdir().unwrap();
        let s = crate::config::testing::settings(tmp.path());
        let err = Brain::load(&s).err().unwrap();
        assert!(err.downcast_ref::<NoBrain>().is_some() && err.to_string().contains("aegist learn"));
    }

    #[test]
    fn guessing_ahead_never_changes_what_is_written() {
        let (_tmp, s) = trained(80);
        let brain = Brain::load_at(&s, Precision::F32).unwrap();
        let prompt = format!("{FILE}m1.py\n{}", &CORPUS[..60]);
        for temperature in [0.0, 0.8] {
            let plain = brain.generate(&prompt, &opts(40, temperature, false), &never_stop, &mut |_, _| true);
            let fast = brain.generate(&prompt, &opts(40, temperature, true), &never_stop, &mut |_, _| true);
            assert_eq!(plain.text, fast.text, "temperature {temperature}");
            assert_eq!(plain.spans.len(), fast.spans.len());
            for (a, b) in plain.spans.iter().zip(&fast.spans) {
                assert!((a.2 - b.2).abs() < 1e-3, "{a:?} {b:?}");
            }
            assert_eq!(plain.guessed, 0);
        }
        // a model trained on repeated code copies it, and the guesses land
        let fast = brain.generate(&prompt, &opts(60, 0.0, true), &never_stop, &mut |_, _| true);
        assert!(fast.guessed > 10, "guessed {}", fast.guessed);
        assert!(fast.mean_prob() > 0.0 && fast.mean_prob() <= 1.0, "{}", fast.mean_prob());
        assert!(fast.prompt_nll < brain.text_nll("zq xv wq") + 5.0);
    }

    #[test]
    fn candidates_share_the_prompt_and_stop_rules_cut() {
        let (_tmp, s) = trained(60);
        let brain = Brain::load(&s).unwrap();
        let prompt = format!("{FILE}m2.py\ndef add(a, b):\n");
        let one = brain.generate(&prompt, &opts(30, 0.0, true), &never_stop, &mut |_, _| true);
        let many = brain.generate_many(&prompt, &[opts(30, 0.0, true), opts(30, 0.0, false)], &never_stop);
        assert_eq!(many[0].text, one.text);
        assert_eq!(many[1].text, one.text);
        let first_line = |t: &str| t.find('\n').map(|i| i + 1);
        let g = brain.generate(&prompt, &opts(30, 0.0, true), &first_line, &mut |_, _| true);
        assert_eq!(g.stop, Stop::Rule);
        assert!(g.text.ends_with('\n') && g.text.matches('\n').count() == 1, "{:?}", g.text);
        let mut pieces = String::new();
        let streamed = brain.generate(&prompt, &opts(12, 0.0, true), &never_stop, &mut |p, _| {
            pieces.push_str(p);
            true
        });
        assert_eq!(pieces, streamed.text);
        let cancelled = brain.generate(&prompt, &opts(12, 0.0, true), &never_stop, &mut |_, _| false);
        assert_eq!((cancelled.stop, cancelled.spans.len()), (Stop::Cancelled, 1));
    }

    #[test]
    fn scoring_options_matches_reading_them_whole() {
        let (_tmp, s) = trained(40);
        let brain = Brain::load_at(&s, Precision::F32).unwrap();
        let prompt = format!("{FILE}m4.py\ndef add(a, b):\n");
        let options = ["    return a + b\n".to_string(), "zzz qq".to_string(), String::new()];
        let scores = brain.score_continuations(&prompt, &options);
        let v = brain.model.cfg.vocab_size;
        for (o, &(total, n)) in options.iter().zip(&scores) {
            let (p, q) = (brain.tokenizer.encode(&prompt), brain.tokenizer.encode(o));
            assert_eq!(n, q.len());
            let ids: Vec<u32> = p.iter().chain(&q).copied().collect();
            let out = brain.model.infer(&KvCache::empty(&brain.model.cfg), &ids, 1, ids.len(), None);
            let want: f32 = (p.len()..ids.len())
                .map(|t| {
                    let mut lp = out.logits[(t - 1) * v..t * v].to_vec();
                    log_softmax(&mut lp);
                    lp[ids[t] as usize]
                })
                .sum();
            assert!((total - want).abs() < 1e-2, "{o:?}: {total} vs {want}");
        }
        assert!(scores[0].0 / scores[0].1 as f32 > scores[1].0 / scores[1].1 as f32, "{scores:?}");
    }

    #[test]
    fn long_prompts_are_read_past_the_trained_context() {
        let (_tmp, s) = trained(10);
        let brain = Brain::load(&s).unwrap();
        let block = brain.model.cfg.block_size;
        let long = format!("{FILE}m3.py\n{}", CORPUS.repeat(2));
        assert!(brain.count_tokens(&long) > block, "prompt should exceed the trained context");
        let g = brain.generate(&long, &opts(8, 0.0, true), &never_stop, &mut |_, _| true);
        assert!(g.prompt_tokens > block && g.prompt_tokens + g.tokens() <= brain.max_context());
    }

    #[test]
    fn it_stops_when_it_loses_track_and_finds_weak_stretches() {
        let (_tmp, s) = trained(10);
        let brain = Brain::load_at(&s, Precision::F32).unwrap();
        let prompt = format!("{FILE}m5.py\nzq xv wq qq zz");
        let never = brain.generate(&prompt, &opts(40, 1.0, true), &never_stop, &mut |_, _| true);
        assert_ne!(never.stop, Stop::Unsure);
        let strict = brain.generate(&prompt, &GenOptions { give_up_below: 1.01, ..opts(40, 1.0, true) }, &never_stop, &mut |_, _| true);
        assert_eq!((strict.stop, strict.tokens()), (Stop::Unsure, STRETCH));
        let g = |ps: &[f32]| Generation { text: String::new(), spans: ps.iter().map(|&p| (0, 0, p)).collect(), stop: Stop::EndOfFile,
                                           prompt_nll: 0.0, prompt_tokens: 0, guessed: 0, seconds: 0.0 };
        let mut ps = vec![0.9f32; 30];
        ps[12..20].fill(0.05);
        let hidden = g(&ps);
        assert!(hidden.mean_prob() > 0.6 && (hidden.weakest() - 0.05).abs() < 1e-4, "{}", hidden.weakest());
        assert_eq!(g(&[]).weakest(), 1.0);
        assert!((g(&[0.5, 0.7]).weakest() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn sampling_respects_top_p_and_greedy() {
        let logp: Vec<f32> = [0.05f32, 0.6, 0.05, 0.3].iter().map(|p| p.ln()).collect();
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logp, 0.0, 1.0, &mut rng), 1);
        assert!((0..300).all(|_| matches!(sample(&logp, 1.0, 0.85, &mut rng), 1 | 3)));
        assert!((0..300).any(|_| sample(&logp, 1.0, 1.0, &mut rng) == 0));
        assert_eq!(guess(&[1, 2, 3, 4, 9, 2, 3], 3), vec![4, 9, 2]);
        assert_eq!(looping("x = 1\nfrom a import b\nfrom a import b\nfrom a import b\n"), Some(22));
        assert_eq!(looping("a = f(1)\nb = g(2)\na = f(1)\nb = g(2)\n"), Some(18));
        assert_eq!(looping("    }\n    }\n    }\n"), None);
        assert_eq!(looping("a = 1\nb = 2\nc = 3\n"), None);
        assert_eq!(looping("a = 1\na = 1"), None);
        assert_eq!(guess(&[5, 6, 7], 4), Vec::<u32>::new());
    }
}
