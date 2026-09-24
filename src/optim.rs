//! AdamW with decoupled weight decay (matrices only, not norm gains), and
//! global gradient-norm clipping. Updates run in parallel chunks.

use rayon::prelude::*;

pub struct AdamW {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: u64,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// (offset, len, decay?) covering every parameter.
    segments: Vec<(usize, usize, bool)>,
}

impl AdamW {
    pub fn new(n: usize, matrices: &[(usize, usize)], weight_decay: f32) -> Self {
        let mut segments = Vec::new();
        let mut at = 0;
        let mut sorted = matrices.to_vec();
        sorted.sort();
        for (off, len) in sorted {
            if off > at {
                segments.push((at, off - at, false));
            }
            segments.push((off, len, true));
            at = off + len;
        }
        if at < n {
            segments.push((at, n - at, false));
        }
        AdamW { m: vec![0.0; n], v: vec![0.0; n], t: 0, beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay, segments }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.t += 1;
        let (b1, b2, eps) = (self.beta1, self.beta2, self.eps);
        let c1 = 1.0 - b1.powi(self.t as i32);
        let c2 = 1.0 - b2.powi(self.t as i32);
        for &(off, len, decay) in &self.segments {
            let wd = if decay { lr * self.weight_decay } else { 0.0 };
            let r = off..off + len;
            params[r.clone()]
                .par_chunks_mut(8192)
                .zip(grads[r.clone()].par_chunks(8192))
                .zip(self.m[r.clone()].par_chunks_mut(8192))
                .zip(self.v[r].par_chunks_mut(8192))
                .for_each(|(((p, g), m), v)| {
                    for i in 0..p.len() {
                        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                        p[i] -= wd * p[i] + lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
                    }
                });
        }
    }
}

/// Scale gradients down so their global norm is at most `max_norm`; returns the norm before.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let norm = grads.par_chunks(8192).map(|c| c.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>()).sum::<f64>().sqrt() as f32;
    if norm > max_norm {
        let s = max_norm / (norm + 1e-6);
        grads.par_iter_mut().for_each(|g| *g *= s);
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_a_quadratic_and_decays_only_matrices() {
        // params 0..4 are a "matrix", 4..6 are gains
        let mut p = vec![3.0f32, -2.0, 1.0, 0.5, 2.0, -1.0];
        let mut opt = AdamW::new(6, &[(0, 4)], 0.0);
        for _ in 0..2000 {
            let g: Vec<f32> = p.iter().map(|x| 2.0 * x).collect();
            opt.step(&mut p, &g, 0.01);
        }
        assert!(p.iter().all(|x| x.abs() < 1e-2), "{p:?}");

        let mut p = vec![1.0f32; 6];
        let mut opt = AdamW::new(6, &[(0, 4)], 0.5);
        opt.step(&mut p, &[0.0; 6], 0.1);
        assert!(p[..4].iter().all(|&x| x < 1.0) && p[4..].iter().all(|&x| x == 1.0));
    }

    #[test]
    fn clipping() {
        let mut g = vec![3.0f32, 4.0];
        assert_eq!(clip_grad_norm(&mut g, 1.0), 5.0);
        assert!((g[0] - 0.6).abs() < 1e-4 && (g[1] - 0.8).abs() < 1e-4);
    }
}
