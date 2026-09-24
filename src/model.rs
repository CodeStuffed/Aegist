//! A decoder-only transformer, written from scratch. No pretrained weights:
//! it starts as random numbers and learns only from the text it trains on.
//!
//! Per block (pre-norm, as in current small language models):
//!     x = x + Dropout(Attention(RMSNorm(x)))     rotary positions on q and k
//!     x = x + Dropout(SwiGLU-MLP(RMSNorm(x)))
//! then a final RMSNorm, and the output layer shares the token embedding.
//!
//! All weights live in one flat Vec<f32>; `Layout` says where each one is.

use crate::kernels::{self as k, Rope};
use crate::rng::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,
    pub mlp_hidden: usize,
    pub dropout: f32,
    pub rope_base: f32,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    /// SwiGLU hidden width: ~8/3 of d_model, rounded up to a multiple of 32.
    pub fn default_mlp_hidden(d_model: usize) -> usize {
        (d_model * 8 / 3).div_ceil(32) * 32
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.d_model % self.n_head == 0, "d_model must be divisible by n_head");
        anyhow::ensure!(self.head_dim() % 2 == 0, "head size (d_model / n_head) must be even");
        anyhow::ensure!(self.vocab_size > 0 && self.block_size > 1, "empty vocabulary or context");
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LayerAt {
    pub rms1: usize,
    pub wqkv: usize,
    pub wo: usize,
    pub rms2: usize,
    pub w13: usize,
    pub w2: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub wte: usize,
    pub layers: Vec<LayerAt>,
    pub rmsf: usize,
    pub total: usize,
    /// (offset, len) of every matrix; weight decay applies to these only.
    pub matrices: Vec<(usize, usize)>,
}

impl Layout {
    pub fn new(cfg: &ModelConfig) -> Self {
        let (c, h, v) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size);
        let mut at = 0;
        let mut matrices = Vec::new();
        let mut take = |n: usize, matrix: bool| {
            let o = at;
            at += n;
            if matrix {
                matrices.push((o, n));
            }
            o
        };
        let wte = take(v * c, true);
        let layers = (0..cfg.n_layer)
            .map(|_| LayerAt {
                rms1: take(c, false),
                wqkv: take(c * 3 * c, true),
                wo: take(c * c, true),
                rms2: take(c, false),
                w13: take(c * 2 * h, true),
                w2: take(h * c, true),
            })
            .collect();
        let rmsf = take(c, false);
        Layout { wte, layers, rmsf, total: at, matrices }
    }
}

pub struct Model {
    pub cfg: ModelConfig,
    pub layout: Layout,
    pub params: Vec<f32>,
    rope: Rope,
}

/// Everything the forward pass keeps for the backward pass, allocated once
/// for a (batch, seq) shape and reused every step.
pub struct Acts {
    pub b: usize,
    pub t: usize,
    x: Vec<Vec<f32>>, // residual stream entering each layer, plus the final one
    xn1: Vec<Vec<f32>>,
    inv1: Vec<Vec<f32>>,
    qkv: Vec<Vec<f32>>,
    probs: Vec<Vec<f32>>,
    att: Vec<Vec<f32>>,
    mask1: Vec<Vec<f32>>,
    xmid: Vec<Vec<f32>>,
    xn2: Vec<Vec<f32>>,
    inv2: Vec<Vec<f32>>,
    ab: Vec<Vec<f32>>,
    hg: Vec<Vec<f32>>,
    mask2: Vec<Vec<f32>>,
    xnf: Vec<f32>,
    invf: Vec<f32>,
    pub logits: Vec<f32>,
    tmp: Vec<f32>,
    // backward buffers
    dx: Vec<f32>,
    dn: Vec<f32>,
    dqkv: Vec<f32>,
    dab: Vec<f32>,
    dhg: Vec<f32>,
}

impl Acts {
    pub fn new(cfg: &ModelConfig, b: usize, t: usize) -> Self {
        let (c, h, v, l, nh) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size, cfg.n_layer, cfg.n_head);
        let bt = b * t;
        let per = |n: usize| (0..l).map(|_| vec![0f32; n]).collect::<Vec<_>>();
        let masks = |n: usize| if cfg.dropout > 0.0 { per(n) } else { per(0) };
        Acts {
            b,
            t,
            x: (0..=l).map(|_| vec![0f32; bt * c]).collect(),
            xn1: per(bt * c),
            inv1: per(bt),
            qkv: per(bt * 3 * c),
            probs: per(b * nh * t * t),
            att: per(bt * c),
            mask1: masks(bt * c),
            xmid: per(bt * c),
            xn2: per(bt * c),
            inv2: per(bt),
            ab: per(bt * 2 * h),
            hg: per(bt * h),
            mask2: masks(bt * c),
            xnf: vec![0f32; bt * c],
            invf: vec![0f32; bt],
            logits: vec![0f32; bt * v],
            tmp: vec![0f32; bt * c],
            dx: vec![0f32; bt * c],
            dn: vec![0f32; bt * c],
            dqkv: vec![0f32; bt * 3 * c],
            dab: vec![0f32; bt * 2 * h],
            dhg: vec![0f32; bt * h],
        }
    }
}

/// Keys and values of a prefix already run through the model (per layer,
/// len x C, rotated), so later tokens can attend to it without recomputing.
#[derive(Clone, Debug)]
pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn empty(cfg: &ModelConfig) -> Self {
        KvCache { k: vec![Vec::new(); cfg.n_layer], v: vec![Vec::new(); cfg.n_layer], len: 0 }
    }

    /// Append row `row` of an inference step's new keys/values.
    pub fn push_row(&mut self, out: &Inference, row: usize, c: usize) {
        for layer in 0..self.k.len() {
            self.k[layer].extend_from_slice(&out.new_k[layer][row * c..(row + 1) * c]);
            self.v[layer].extend_from_slice(&out.new_v[layer][row * c..(row + 1) * c]);
        }
        self.len += 1;
    }
}

pub struct Inference {
    pub logits: Vec<f32>, // (b*l) x V
    pub new_k: Vec<Vec<f32>>,
    pub new_v: Vec<Vec<f32>>,
}

impl Model {
    pub fn new(cfg: ModelConfig, seed: u64) -> Self {
        let layout = Layout::new(&cfg);
        let mut params = vec![0f32; layout.total];
        let mut rng = Rng::new(seed);
        let (c, h) = (cfg.d_model, cfg.mlp_hidden);
        let resid_std = 0.02 / ((2 * cfg.n_layer) as f32).sqrt();
        let mut fill = |off: usize, n: usize, std: f32| {
            params[off..off + n].iter_mut().for_each(|p| *p = rng.normal() * std);
        };
        fill(layout.wte, cfg.vocab_size * c, 0.02);
        for la in &layout.layers {
            fill(la.wqkv, c * 3 * c, 0.02);
            fill(la.wo, c * c, resid_std);
            fill(la.w13, c * 2 * h, 0.02);
            fill(la.w2, h * c, resid_std);
        }
        for la in &layout.layers {
            params[la.rms1..la.rms1 + c].fill(1.0);
            params[la.rms2..la.rms2 + c].fill(1.0);
        }
        params[layout.rmsf..layout.rmsf + c].fill(1.0);
        Self::from_params(cfg, params)
    }

    pub fn from_params(cfg: ModelConfig, params: Vec<f32>) -> Self {
        let layout = Layout::new(&cfg);
        assert_eq!(params.len(), layout.total, "parameter count doesn't match the config");
        let rope = Rope::new(cfg.block_size, cfg.head_dim(), cfg.rope_base);
        Model { cfg, layout, params, rope }
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    fn p(&self, off: usize, n: usize) -> &[f32] {
        &self.params[off..off + n]
    }

    /// Training/eval forward pass over `tokens` (b x t). Fills acts.logits.
    /// `dropout_seed`: Some(seed) switches dropout on.
    pub fn forward(&self, acts: &mut Acts, tokens: &[u32], dropout_seed: Option<u64>) {
        let cfg = &self.cfg;
        let (b, t) = (acts.b, acts.t);
        let (c, h, v, nh) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size, cfg.n_head);
        let bt = b * t;
        assert_eq!(tokens.len(), bt);
        let drop = dropout_seed.filter(|_| cfg.dropout > 0.0);

        k::embed(&mut acts.x[0], self.p(self.layout.wte, v * c), tokens, c);
        for (l, la) in self.layout.layers.iter().enumerate() {
            let (xs, rest) = acts.x.split_at_mut(l + 1);
            let x_in = &xs[l];
            let x_out = &mut rest[0];

            k::rmsnorm_fwd(&mut acts.xn1[l], &mut acts.inv1[l], x_in, self.p(la.rms1, c), c);
            k::matmul(&mut acts.qkv[l], &acts.xn1[l], self.p(la.wqkv, c * 3 * c), bt, c, 3 * c, false, false, false);
            self.rope.apply(&mut acts.qkv[l], c, nh, |row| row % t, false);
            k::attention_fwd(&mut acts.att[l], &mut acts.probs[l], &acts.qkv[l], b, t, c, nh);
            k::matmul(&mut acts.tmp, &acts.att[l], self.p(la.wo, c * c), bt, c, c, false, false, false);
            if let Some(seed) = drop {
                k::dropout_mask(&mut acts.mask1[l], cfg.dropout, seed ^ (2 * l as u64 + 1));
            }
            k::residual(&mut acts.xmid[l], x_in, &acts.tmp, drop.map(|_| acts.mask1[l].as_slice()));

            k::rmsnorm_fwd(&mut acts.xn2[l], &mut acts.inv2[l], &acts.xmid[l], self.p(la.rms2, c), c);
            k::matmul(&mut acts.ab[l], &acts.xn2[l], self.p(la.w13, c * 2 * h), bt, c, 2 * h, false, false, false);
            k::swiglu_fwd(&mut acts.hg[l], &acts.ab[l], h);
            k::matmul(&mut acts.tmp, &acts.hg[l], self.p(la.w2, h * c), bt, h, c, false, false, false);
            if let Some(seed) = drop {
                k::dropout_mask(&mut acts.mask2[l], cfg.dropout, seed ^ (2 * l as u64 + 2));
            }
            k::residual(x_out, &acts.xmid[l], &acts.tmp, drop.map(|_| acts.mask2[l].as_slice()));
        }
        let last = &acts.x[cfg.n_layer];
        k::rmsnorm_fwd(&mut acts.xnf, &mut acts.invf, last, self.p(self.layout.rmsf, c), c);
        k::matmul(&mut acts.logits, &acts.xnf, self.p(self.layout.wte, v * c), bt, c, v, false, true, false);
    }

    /// Mean next-token loss, without gradients.
    pub fn loss(&self, acts: &mut Acts, tokens: &[u32], targets: &[u32]) -> f32 {
        self.forward(acts, tokens, None);
        k::cross_entropy(&acts.logits, targets, self.cfg.vocab_size)
    }

    /// Forward + backward. Adds gradients into `grads` (same layout as
    /// params) and returns the mean loss.
    pub fn forward_backward(&self, acts: &mut Acts, grads: &mut [f32], tokens: &[u32], targets: &[u32], dropout_seed: Option<u64>) -> f32 {
        self.forward(acts, tokens, dropout_seed);
        let cfg = &self.cfg;
        let (b, t) = (acts.b, acts.t);
        let (c, h, v, nh, nl) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size, cfg.n_head, cfg.n_layer);
        let bt = b * t;
        let drop = dropout_seed.is_some() && cfg.dropout > 0.0;
        let lay = &self.layout;

        let loss = k::cross_entropy_grad(&mut acts.logits, targets, v);
        // logits = xnf . wte^T
        k::matmul(&mut acts.dn, &acts.logits, self.p(lay.wte, v * c), bt, v, c, false, false, false);
        k::matmul(&mut grads[lay.wte..lay.wte + v * c], &acts.logits, &acts.xnf, v, bt, c, true, false, true);
        acts.dx.fill(0.0);
        k::rmsnorm_bwd(&mut acts.dx, &mut grads[lay.rmsf..lay.rmsf + c], &acts.dn, &acts.x[nl], &acts.invf, self.p(lay.rmsf, c), c);

        for l in (0..nl).rev() {
            let la = &lay.layers[l];
            // MLP branch: x_out = xmid + mask2 * (hg . W2)
            let dmlp: &[f32] = if drop {
                acts.tmp.iter_mut().zip(&acts.dx).zip(&acts.mask2[l]).for_each(|((o, d), m)| *o = d * m);
                &acts.tmp
            } else {
                &acts.dx
            };
            k::matmul(&mut grads[la.w2..la.w2 + h * c], &acts.hg[l], dmlp, h, bt, c, true, false, true);
            k::matmul(&mut acts.dhg, dmlp, self.p(la.w2, h * c), bt, c, h, false, true, false);
            k::swiglu_bwd(&mut acts.dab, &acts.dhg, &acts.ab[l], h);
            k::matmul(&mut grads[la.w13..la.w13 + c * 2 * h], &acts.xn2[l], &acts.dab, c, bt, 2 * h, true, false, true);
            k::matmul(&mut acts.dn, &acts.dab, self.p(la.w13, c * 2 * h), bt, 2 * h, c, false, true, false);
            k::rmsnorm_bwd(&mut acts.dx, &mut grads[la.rms2..la.rms2 + c], &acts.dn, &acts.xmid[l], &acts.inv2[l], self.p(la.rms2, c), c);

            // attention branch: xmid = x_in + mask1 * (att . Wo)
            let dproj: &[f32] = if drop {
                acts.tmp.iter_mut().zip(&acts.dx).zip(&acts.mask1[l]).for_each(|((o, d), m)| *o = d * m);
                &acts.tmp
            } else {
                &acts.dx
            };
            k::matmul(&mut grads[la.wo..la.wo + c * c], &acts.att[l], dproj, c, bt, c, true, false, true);
            k::matmul(&mut acts.dn, dproj, self.p(la.wo, c * c), bt, c, c, false, true, false);
            k::attention_bwd(&mut acts.dqkv, &acts.dn, &acts.qkv[l], &acts.probs[l], b, t, c, nh);
            self.rope.apply(&mut acts.dqkv, c, nh, |row| row % t, true);
            k::matmul(&mut grads[la.wqkv..la.wqkv + c * 3 * c], &acts.xn1[l], &acts.dqkv, c, bt, 3 * c, true, false, true);
            k::matmul(&mut acts.dn, &acts.dqkv, self.p(la.wqkv, c * 3 * c), bt, 3 * c, c, false, true, false);
            k::rmsnorm_bwd(&mut acts.dx, &mut grads[la.rms1..la.rms1 + c], &acts.dn, &acts.x[l], &acts.inv1[l], self.p(la.rms1, c), c);
        }
        k::embed_bwd(&mut grads[lay.wte..lay.wte + v * c], &acts.dx, tokens, c);
        loss
    }

    /// Inference: `b` sequences of `l` new tokens, all continuing the prefix
    /// in `cache`. Returns logits for every new token plus their keys/values.
    /// `rng`: Some switches dropout on (Monte Carlo dropout).
    pub fn infer(&self, cache: &KvCache, tokens: &[u32], b: usize, l: usize, mut rng: Option<&mut Rng>) -> Inference {
        let cfg = &self.cfg;
        let (c, h, v, nh) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size, cfg.n_head);
        let n = b * l;
        assert_eq!(tokens.len(), n);
        assert!(cache.len + l <= cfg.block_size, "sequence longer than the model's context");
        let p0 = cache.len;
        let mut x = vec![0f32; n * c];
        let mut xn = vec![0f32; n * c];
        let mut inv = vec![0f32; n];
        let mut qkv = vec![0f32; n * 3 * c];
        let mut att = vec![0f32; n * c];
        let mut tmp = vec![0f32; n * c];
        let mut ab = vec![0f32; n * 2 * h];
        let mut hg = vec![0f32; n * h];
        let mut mask = vec![0f32; n * c];
        let mut new_k = Vec::with_capacity(cfg.n_layer);
        let mut new_v = Vec::with_capacity(cfg.n_layer);
        let mut xnext = vec![0f32; n * c];

        k::embed(&mut x, self.p(self.layout.wte, v * c), tokens, c);
        for (li, la) in self.layout.layers.iter().enumerate() {
            k::rmsnorm_fwd(&mut xn, &mut inv, &x, self.p(la.rms1, c), c);
            k::matmul(&mut qkv, &xn, self.p(la.wqkv, c * 3 * c), n, c, 3 * c, false, false, false);
            self.rope.apply(&mut qkv, c, nh, |row| p0 + row % l, false);
            k::attention_infer(&mut att, &qkv, &cache.k[li], &cache.v[li], p0, b, l, c, nh);
            let (mut kk, mut vv) = (Vec::with_capacity(n * c), Vec::with_capacity(n * c));
            for r in qkv.chunks(3 * c) {
                kk.extend_from_slice(&r[c..2 * c]);
                vv.extend_from_slice(&r[2 * c..]);
            }
            new_k.push(kk);
            new_v.push(vv);
            k::matmul(&mut tmp, &att, self.p(la.wo, c * c), n, c, c, false, false, false);
            let m = self.maybe_mask(&mut mask, rng.as_deref_mut());
            k::residual(&mut xnext, &x, &tmp, m);
            std::mem::swap(&mut x, &mut xnext);

            k::rmsnorm_fwd(&mut xn, &mut inv, &x, self.p(la.rms2, c), c);
            k::matmul(&mut ab, &xn, self.p(la.w13, c * 2 * h), n, c, 2 * h, false, false, false);
            k::swiglu_fwd(&mut hg, &ab, h);
            k::matmul(&mut tmp, &hg, self.p(la.w2, h * c), n, h, c, false, false, false);
            let m = self.maybe_mask(&mut mask, rng.as_deref_mut());
            k::residual(&mut xnext, &x, &tmp, m);
            std::mem::swap(&mut x, &mut xnext);
        }
        k::rmsnorm_fwd(&mut xn, &mut inv, &x, self.p(self.layout.rmsf, c), c);
        let mut logits = vec![0f32; n * v];
        k::matmul(&mut logits, &xn, self.p(self.layout.wte, v * c), n, c, v, false, true, false);
        Inference { logits, new_k, new_v }
    }

    fn maybe_mask<'a>(&self, mask: &'a mut [f32], rng: Option<&mut Rng>) -> Option<&'a [f32]> {
        let rng = rng?;
        if self.cfg.dropout <= 0.0 {
            return None;
        }
        k::dropout_mask(mask, self.cfg.dropout, rng.next_u64());
        Some(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(dropout: f32) -> ModelConfig {
        ModelConfig { vocab_size: 13, block_size: 8, n_layer: 2, n_head: 2, d_model: 8, mlp_hidden: 12, dropout, rope_base: 10000.0 }
    }

    fn batch(rng: &mut Rng, b: usize, t: usize) -> (Vec<u32>, Vec<u32>) {
        let toks: Vec<u32> = (0..b * t).map(|_| rng.below(13) as u32).collect();
        let tgts: Vec<u32> = (0..b * t).map(|_| rng.below(13) as u32).collect();
        (toks, tgts)
    }

    /// Directional derivative check per parameter group: (L(p+eps*d) -
    /// L(p-eps*d)) / 2eps must match grad . d, for a random d in that group.
    #[test]
    fn gradients_match_finite_differences() {
        for dropout in [0.0f32, 0.3] {
            let mut model = Model::new(tiny(dropout), 5);
            // Larger-than-init weights and non-trivial norm gains, so every
            // gradient is big enough for an f32 finite difference to resolve.
            let mut rng = Rng::new(11);
            for &(off, n) in &model.layout.matrices.clone() {
                model.params[off..off + n].iter_mut().for_each(|w| *w *= 12.0);
            }
            for la in model.layout.layers.clone() {
                for off in [la.rms1, la.rms2] {
                    model.params[off..off + 8].iter_mut().for_each(|g| *g = 1.0 + 0.3 * rng.normal());
                }
            }
            let (b, t) = (2, 6);
            let (toks, tgts) = batch(&mut rng, b, t);
            let mut acts = Acts::new(&model.cfg, b, t);
            let mut grads = vec![0f32; model.num_params()];
            let seed = Some(77);
            model.forward_backward(&mut acts, &mut grads, &toks, &tgts, seed);

            let lay = model.layout.clone();
            let c = 8;
            let mut groups: Vec<(&str, usize, usize)> = vec![("wte", lay.wte, 13 * c), ("rmsf", lay.rmsf, c)];
            for la in &lay.layers {
                groups.extend([("rms1", la.rms1, c), ("wqkv", la.wqkv, 3 * c * c), ("wo", la.wo, c * c),
                               ("rms2", la.rms2, c), ("w13", la.w13, c * 24), ("w2", la.w2, 12 * c)]);
            }
            for (name, off, n) in groups {
                let d: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
                let analytic: f64 = (0..n).map(|i| grads[off + i] as f64 * d[i] as f64).sum();
                let eps = 1e-3f32;
                let mut loss_at = |sign: f32| {
                    for i in 0..n {
                        model.params[off + i] += sign * eps * d[i];
                    }
                    model.forward(&mut acts, &toks, seed);
                    let l = k::cross_entropy(&acts.logits, &tgts, 13);
                    for i in 0..n {
                        model.params[off + i] -= sign * eps * d[i];
                    }
                    l as f64
                };
                let numeric = (loss_at(1.0) - loss_at(-1.0)) / (2.0 * eps as f64);
                let err = (numeric - analytic).abs() / (numeric.abs() + analytic.abs() + 1e-4);
                eprintln!("{name:5} dropout {dropout}: numeric {numeric:+.6} analytic {analytic:+.6} err {err:.1e}");
                assert!(err < 2e-2, "{name} (dropout {dropout}): numeric {numeric:.6} vs analytic {analytic:.6}");
            }
        }
    }

    #[test]
    fn cached_inference_matches_training_forward() {
        let model = Model::new(tiny(0.0), 3);
        let mut rng = Rng::new(2);
        let (toks, _) = batch(&mut rng, 1, 8);
        let mut acts = Acts::new(&model.cfg, 1, 8);
        model.forward(&mut acts, &toks, None);

        // all at once from an empty cache
        let full = model.infer(&KvCache::empty(&model.cfg), &toks, 1, 8, None);
        assert!(full.logits.iter().zip(&acts.logits).all(|(a, b)| (a - b).abs() < 1e-4));

        // prefix of 5, then one token at a time
        let pre = model.infer(&KvCache::empty(&model.cfg), &toks[..5], 1, 5, None);
        let mut cache = KvCache::empty(&model.cfg);
        for r in 0..5 {
            cache.push_row(&pre, r, 8);
        }
        let prefix = cache.clone();
        for i in 5..8 {
            let step = model.infer(&cache, &toks[i..i + 1], 1, 1, None);
            let want = &acts.logits[i * 13..(i + 1) * 13];
            assert!(step.logits.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4), "token {i}");
            cache.push_row(&step, 0, 8);
        }

        // two different continuations of the same prefix, batched
        let conts = [toks[5], toks[6], 3, 4];
        let both = model.infer(&prefix, &conts, 2, 2, None);
        assert!(both.logits[..13].iter().zip(&acts.logits[5 * 13..6 * 13]).all(|(a, b)| (a - b).abs() < 1e-4));
        assert!(both.logits[13..26].iter().zip(&acts.logits[6 * 13..7 * 13]).all(|(a, b)| (a - b).abs() < 1e-4));
    }
}
