//! The math the transformer is made of, forward and backward, written by
//! hand and parallelized across cores with rayon. Matrix multiplies go
//! through the `gemm` crate (SIMD, multithreaded) - the role BLAS plays for
//! NumPy. Everything else is fused into single passes over memory.

use crate::rng::Rng;
use gemm::Parallelism;
use rayon::prelude::*;

pub const RMS_EPS: f32 = 1e-5;
/// Below these sizes an operation stays on one thread: for small work
/// (like generating one token), splitting it costs more than it saves.
const MIN_ROWS: usize = 16;
const MIN_ELEMS: usize = 16 * 1024;

fn parallelism(work: usize) -> Parallelism {
    let threads = rayon::current_num_threads();
    if threads > 1 && work >= 1 << 18 {
        Parallelism::Rayon(threads)
    } else {
        Parallelism::None
    }
}

/// c (m x n) = [c +] op(a) . op(b), where op(a) is m x k and op(b) is k x n.
/// `ta` / `tb`: the operand is stored transposed (k x m / n x k).
#[allow(clippy::too_many_arguments)]
pub fn matmul(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, ta: bool, tb: bool, accumulate: bool) {
    assert!(c.len() >= m * n && a.len() >= m * k && b.len() >= k * n, "matmul shape mismatch");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            c[..m * n].fill(0.0);
        }
        return;
    }
    let (a_rs, a_cs) = if ta { (1, m as isize) } else { (k as isize, 1) };
    let (b_rs, b_cs) = if tb { (1, k as isize) } else { (n as isize, 1) };
    // SAFETY: the asserts above guarantee every index gemm touches is in bounds.
    unsafe {
        gemm::gemm(
            m, n, k,
            c.as_mut_ptr(), 1, n as isize, accumulate,
            a.as_ptr(), a_cs, a_rs,
            b.as_ptr(), b_cs, b_rs,
            if accumulate { 1.0 } else { 0.0 }, 1.0,
            false, false, false,
            parallelism(m * n * k),
        );
    }
}

/// a . b, with AVX2 + FMA when the CPU has it (checked at runtime).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: the features are available (checked just above).
        return unsafe { dot_avx2(a, b) };
    }
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    acc.iter().sum::<f32>() + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let k = a.len().min(b.len());
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 16 <= k {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8)), s1);
        i += 16;
    }
    let v = _mm256_add_ps(s0, s1);
    let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    let mut total = _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
    while i < k {
        total += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    total
}

// ---------------------------------------------------------------- RMSNorm

/// out = x / rms(x) * g, row by row; also stores 1/rms per row.
pub fn rmsnorm_fwd(out: &mut [f32], inv: &mut [f32], x: &[f32], g: &[f32], c: usize) {
    out.par_chunks_mut(c)
        .zip(inv.par_iter_mut())
        .zip(x.par_chunks(c))
        .with_min_len(MIN_ROWS)
        .for_each(|((o, iv), xr)| {
            let ms = xr.iter().map(|v| v * v).sum::<f32>() / c as f32;
            let r = 1.0 / (ms + RMS_EPS).sqrt();
            *iv = r;
            for i in 0..c {
                o[i] = xr[i] * r * g[i];
            }
        });
}

/// dx += d(out)/dx . dy ;  dg += d(out)/dg . dy
pub fn rmsnorm_bwd(dx: &mut [f32], dg: &mut [f32], dy: &[f32], x: &[f32], inv: &[f32], g: &[f32], c: usize) {
    dx.par_chunks_mut(c)
        .zip(dy.par_chunks(c))
        .zip(x.par_chunks(c))
        .zip(inv.par_iter())
        .with_min_len(MIN_ROWS)
        .for_each(|(((dxr, dyr), xr), &r)| {
            let dot = (0..c).map(|i| dyr[i] * g[i] * xr[i]).sum::<f32>() / c as f32;
            for i in 0..c {
                dxr[i] += r * (dyr[i] * g[i] - xr[i] * r * r * dot);
            }
        });
    let partial = dy
        .par_chunks(c)
        .zip(x.par_chunks(c))
        .zip(inv.par_iter())
        .with_min_len(MIN_ROWS)
        .fold(
            || vec![0f32; c],
            |mut acc, ((dyr, xr), &r)| {
                for i in 0..c {
                    acc[i] += dyr[i] * xr[i] * r;
                }
                acc
            },
        )
        .reduce(|| vec![0f32; c], |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        });
    dg.iter_mut().zip(&partial).for_each(|(d, p)| *d += p);
}

// ------------------------------------------------------ rotary positions

/// Rotary position embedding tables: each pair of dimensions in a head is
/// rotated by an angle proportional to the token's position.
#[derive(Clone, Debug)]
pub struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(max_pos: usize, head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0f32; max_pos * half];
        let mut sin = vec![0f32; max_pos * half];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = (base as f64).powf(-2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos[pos * half + i] = angle.cos() as f32;
                sin[pos * half + i] = angle.sin() as f32;
            }
        }
        Rope { half, cos, sin }
    }

    /// Positions it has angles for.
    pub fn max_pos(&self) -> usize {
        if self.half == 0 { 0 } else { self.cos.len() / self.half }
    }

    /// (cos, sin), each max_pos x head_dim/2.
    pub fn tables(&self) -> (&[f32], &[f32]) {
        (&self.cos, &self.sin)
    }

    /// Rotate the q and k blocks of `qkv` rows (row stride 3c) in place.
    /// `inverse` undoes the rotation (used on gradients).
    pub fn apply(&self, qkv: &mut [f32], c: usize, n_head: usize, position: impl Fn(usize) -> usize + Sync, inverse: bool) {
        let hd = c / n_head;
        let half = self.half;
        qkv.par_chunks_mut(3 * c).enumerate().with_min_len(MIN_ROWS).for_each(|(row, r)| {
            let pos = position(row);
            let cs = &self.cos[pos * half..(pos + 1) * half];
            let sn = &self.sin[pos * half..(pos + 1) * half];
            for block in [0, c] {
                for h in 0..n_head {
                    let v = &mut r[block + h * hd..block + (h + 1) * hd];
                    for i in 0..half {
                        let (x0, x1) = (v[2 * i], v[2 * i + 1]);
                        let s = if inverse { -sn[i] } else { sn[i] };
                        v[2 * i] = x0 * cs[i] - x1 * s;
                        v[2 * i + 1] = x0 * s + x1 * cs[i];
                    }
                }
            }
        });
    }
}

// -------------------------------------------------------------- attention
//
// Attention runs as matrix multiplies per (sequence, head): scores = Q.K^T,
// out = P.V, and the backward equivalents. Q, K and V are read straight out
// of the [q | k | v] rows through strides, and each (sequence, head) task
// writes only its own rows x head-columns of the output, so tasks never
// overlap.

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
#[derive(Clone, Copy)]
struct ConstPtr(*const f32);
unsafe impl Send for ConstPtr {}
unsafe impl Sync for ConstPtr {}

/// dst (m x n, strides dst_rs/dst_cs) = [dst +] scale * lhs (m x k) . rhs (k x n),
/// every operand addressed by (row stride, column stride). Single-threaded:
/// callers parallelize across tasks.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_strided(
    m: usize, n: usize, k: usize,
    dst: *mut f32, dst_rs: usize, dst_cs: usize, accumulate: bool,
    lhs: *const f32, lhs_rs: usize, lhs_cs: usize,
    rhs: *const f32, rhs_rs: usize, rhs_cs: usize,
    scale: f32,
) {
    gemm::gemm(
        m, n, k,
        dst, dst_cs as isize, dst_rs as isize, accumulate,
        lhs, lhs_cs as isize, lhs_rs as isize,
        rhs, rhs_cs as isize, rhs_rs as isize,
        if accumulate { 1.0 } else { 0.0 }, scale,
        false, false, false, Parallelism::None,
    );
}

/// Causal softmax over row i of a t x t score matrix (only j <= i count).
fn causal_softmax_rows(s: &mut [f32], t: usize, offset: usize) {
    for i in 0..s.len() / t {
        let row = &mut s[i * t..(i + 1) * t];
        let n = offset + i + 1;
        let max = row[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row[..n].iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        row[..n].iter_mut().for_each(|v| *v *= inv);
        row[n..].fill(0.0);
    }
}

/// Queries handled together by the attention kernels. Each task keeps only
/// ATT_CHUNK x T scores at a time instead of T x T, which is what makes long
/// contexts affordable: memory grows with T, not T squared.
pub const ATT_CHUNK: usize = 128;

/// Causal self-attention for training. qkv rows hold [q | k | v] (q and k
/// already rotated). Writes out (B*T x C) and each row's log-sum-exp of its
/// scores (B*nh*T), from which the backward pass recomputes the weights -
/// the T x T weights themselves are never kept.
#[allow(clippy::too_many_arguments)]
pub fn attention_fwd(out: &mut [f32], lse: &mut [f32], qkv: &[f32], b: usize, t: usize, c: usize, nh: usize) {
    assert!(out.len() >= b * t * c && lse.len() >= b * nh * t && qkv.len() >= b * t * 3 * c);
    let hd = c / nh;
    let scale = 1.0 / (hd as f32).sqrt();
    let chunks = t.div_ceil(ATT_CHUNK);
    let (o, q, ls) = (Ptr(out.as_mut_ptr()), ConstPtr(qkv.as_ptr()), Ptr(lse.as_mut_ptr()));
    (0..b * nh * chunks).into_par_iter().for_each(|task| {
        let (bh, ci) = (task / chunks, task % chunks);
        let (bb, h) = (bh / nh, bh % nh);
        let (i0, i1) = (ci * ATT_CHUNK, ((ci + 1) * ATT_CHUNK).min(t));
        let (rows, n) = (i1 - i0, i1);
        let (o, q, ls) = (o, q, ls);
        let mut p = vec![0f32; rows * n];
        // SAFETY: offsets stay inside the asserted buffers; this task writes
        // only rows i0..i1 of sequence bb, head h's columns of `out`, and the
        // matching entries of `lse` - disjoint from every other task.
        unsafe {
            let qp = q.0.add(bb * t * 3 * c + h * hd);
            let (kp, vp) = (qp.add(c), qp.add(2 * c));
            gemm_strided(rows, n, hd, p.as_mut_ptr(), n, 1, false, qp.add(i0 * 3 * c), 3 * c, 1, kp, 1, 3 * c, scale);
            for r in 0..rows {
                let row = &mut p[r * n..(r + 1) * n];
                let valid = i0 + r + 1;
                let max = row[..valid].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row[..valid].iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                row[..valid].iter_mut().for_each(|v| *v *= inv);
                row[valid..].fill(0.0);
                *ls.0.add(bh * t + i0 + r) = max + sum.ln();
            }
            gemm_strided(rows, hd, n, o.0.add((bb * t + i0) * c + h * hd), c, 1, false, p.as_ptr(), n, 1, vp, 3 * c, 1, 1.0);
        }
    });
}

/// Backward of attention_fwd: writes d[q | k | v] (gradients w.r.t. the
/// rotated q and k; the caller un-rotates) into dqkv rows. dout and `att`
/// (attention_fwd's output) are B*T x C. The weights are recomputed a chunk
/// of queries at a time from `lse`.
#[allow(clippy::too_many_arguments)]
pub fn attention_bwd(dqkv: &mut [f32], dout: &[f32], qkv: &[f32], att: &[f32], lse: &[f32], b: usize, t: usize, c: usize, nh: usize) {
    assert!(dqkv.len() >= b * t * 3 * c && dout.len() >= b * t * c && att.len() >= b * t * c && lse.len() >= b * nh * t);
    let hd = c / nh;
    let scale = 1.0 / (hd as f32).sqrt();
    let (d, q, g, a) = (Ptr(dqkv.as_mut_ptr()), ConstPtr(qkv.as_ptr()), ConstPtr(dout.as_ptr()), ConstPtr(att.as_ptr()));
    (0..b * nh).into_par_iter().for_each(|bh| {
        let (bb, h) = (bh / nh, bh % nh);
        let (d, q, g, a) = (d, q, g, a);
        let lse = &lse[bh * t..(bh + 1) * t];
        let mut p = vec![0f32; ATT_CHUNK.min(t) * t];
        let mut dp = vec![0f32; ATT_CHUNK.min(t) * t];
        // SAFETY: as in attention_fwd; this task writes only its own sequence's
        // rows and head columns of the q, k and v blocks of dqkv.
        unsafe {
            let qp = q.0.add(bb * t * 3 * c + h * hd);
            let (kp, vp) = (qp.add(c), qp.add(2 * c));
            let dop = g.0.add(bb * t * c + h * hd);
            let ap = a.0.add(bb * t * c + h * hd);
            let dq = d.0.add(bb * t * 3 * c + h * hd);
            let (dk, dv) = (dq.add(c), dq.add(2 * c));
            // D_i = dO_i . O_i  (equals the row sum of P * dP)
            let dsum: Vec<f32> = (0..t)
                .map(|i| dot(std::slice::from_raw_parts(dop.add(i * c), hd), std::slice::from_raw_parts(ap.add(i * c), hd)))
                .collect();
            for i in 0..t {
                std::slice::from_raw_parts_mut(dk.add(i * 3 * c), hd).fill(0.0);
                std::slice::from_raw_parts_mut(dv.add(i * 3 * c), hd).fill(0.0);
            }
            for i0 in (0..t).step_by(ATT_CHUNK) {
                let i1 = (i0 + ATT_CHUNK).min(t);
                let (rows, n) = (i1 - i0, i1);
                let (p, dp) = (&mut p[..rows * n], &mut dp[..rows * n]);
                gemm_strided(rows, n, hd, p.as_mut_ptr(), n, 1, false, qp.add(i0 * 3 * c), 3 * c, 1, kp, 1, 3 * c, scale);
                for r in 0..rows {
                    let row = &mut p[r * n..(r + 1) * n];
                    let (valid, l) = (i0 + r + 1, lse[i0 + r]);
                    row[..valid].iter_mut().for_each(|v| *v = (*v - l).exp());
                    row[valid..].fill(0.0);
                }
                // dV[0..n] += P^T . dO_chunk ;  dP = dO_chunk . V^T
                gemm_strided(n, hd, rows, dv, 3 * c, 1, true, p.as_ptr(), 1, n, dop.add(i0 * c), c, 1, 1.0);
                gemm_strided(rows, n, hd, dp.as_mut_ptr(), n, 1, false, dop.add(i0 * c), c, 1, vp, 1, 3 * c, 1.0);
                // dS = P * (dP - D) * scale, written over dP
                for r in 0..rows {
                    let (pr, dr) = (&p[r * n..(r + 1) * n], &mut dp[r * n..(r + 1) * n]);
                    let (valid, di) = (i0 + r + 1, dsum[i0 + r]);
                    for j in 0..valid {
                        dr[j] = pr[j] * (dr[j] - di) * scale;
                    }
                    dr[valid..].fill(0.0);
                }
                // dQ_chunk = dS . K ;  dK[0..n] += dS^T . Q_chunk
                gemm_strided(rows, hd, n, dq.add(i0 * 3 * c), 3 * c, 1, false, dp.as_ptr(), n, 1, kp, 3 * c, 1, 1.0);
                gemm_strided(n, hd, rows, dk, 3 * c, 1, true, dp.as_ptr(), 1, n, qp.add(i0 * 3 * c), 3 * c, 1, 1.0);
            }
        }
    });
}

/// Attention at inference: B sequences of L new tokens, each continuing a
/// shared prefix whose keys/values (already rotated) are cached. The prefix
/// may be split into segments - (keys, values), each rows x C - so several
/// continuations can share one cached prompt without copying it.
#[allow(clippy::too_many_arguments)]
pub fn attention_infer(out: &mut [f32], qkv: &[f32], segments: &[(&[f32], &[f32])], b: usize, l: usize, c: usize, nh: usize) {
    assert!(out.len() >= b * l * c && qkv.len() >= b * l * 3 * c);
    assert!(segments.iter().all(|(k, v)| k.len() == v.len() && k.len() % c == 0));
    let hd = c / nh;
    let scale = 1.0 / (hd as f32).sqrt();
    let segs: Vec<(ConstPtr, ConstPtr, usize)> =
        segments.iter().filter(|(k, _)| !k.is_empty()).map(|(k, v)| (ConstPtr(k.as_ptr()), ConstPtr(v.as_ptr()), k.len() / c)).collect();
    let p: usize = segs.iter().map(|s| s.2).sum();
    let (o, q) = (Ptr(out.as_mut_ptr()), ConstPtr(qkv.as_ptr()));
    if l <= 4 {
        // A few new tokens (generation): plain dot products beat setting up
        // matrix multiplies for such small shapes.
        let n = p + l;
        let work = b * nh * l * n * hd;
        (0..b * nh).into_par_iter().with_min_len(if work < MIN_ELEMS * 8 { b * nh } else { 1 }).for_each(|bh| {
            let (bb, h) = (bh / nh, bh % nh);
            let (o, q) = (o, q);
            let segs = &segs;
            let mut scores = vec![0f32; n];
            // SAFETY: offsets stay inside the asserted buffers, and this task
            // writes only its own (sequence, head) slice of `out`.
            unsafe {
                let row_of = |j: usize, block: usize| -> &[f32] {
                    let mut j = j;
                    for (k, v, rows) in segs.iter() {
                        if j < *rows {
                            let base = if block == 1 { k.0 } else { v.0 };
                            return std::slice::from_raw_parts(base.add(j * c + h * hd), hd);
                        }
                        j -= rows;
                    }
                    std::slice::from_raw_parts(q.0.add((bb * l + j) * 3 * c + block * c + h * hd), hd)
                };
                for i in 0..l {
                    let qrow = std::slice::from_raw_parts(q.0.add((bb * l + i) * 3 * c + h * hd), hd);
                    let len = p + i + 1;
                    let mut max = f32::NEG_INFINITY;
                    for (j, sj) in scores[..len].iter_mut().enumerate() {
                        *sj = dot(qrow, row_of(j, 1)) * scale;
                        max = max.max(*sj);
                    }
                    let mut sum = 0.0;
                    for sj in scores[..len].iter_mut() {
                        *sj = (*sj - max).exp();
                        sum += *sj;
                    }
                    let out = std::slice::from_raw_parts_mut(o.0.add((bb * l + i) * c + h * hd), hd);
                    out.fill(0.0);
                    for (j, &sj) in scores[..len].iter().enumerate() {
                        let w = sj / sum;
                        out.iter_mut().zip(row_of(j, 2)).for_each(|(a, v)| *a += w * v);
                    }
                }
            }
        });
        return;
    }
    // A longer run of new tokens (a prompt): a chunk of queries at a time,
    // so the scores held at once are ATT_CHUNK x (P + L), never L x (P + L).
    let chunks = l.div_ceil(ATT_CHUNK);
    (0..b * nh * chunks).into_par_iter().for_each(|task| {
        let (bh, ci) = (task / chunks, task % chunks);
        let (bb, h) = (bh / nh, bh % nh);
        let (r0, r1) = (ci * ATT_CHUNK, ((ci + 1) * ATT_CHUNK).min(l));
        let (rows, cols) = (r1 - r0, p + r1);
        let (o, q) = (o, q);
        let mut s = vec![0f32; rows * cols];
        // SAFETY: as in attention_fwd; this task writes rows r0..r1 of its
        // own sequence and head in `out`.
        unsafe {
            let qp = q.0.add(bb * l * 3 * c + h * hd);
            let (kp, vp) = (qp.add(c), qp.add(2 * c));
            let qc = qp.add(r0 * 3 * c);
            let mut col = 0;
            for (k, _, n) in &segs {
                gemm_strided(rows, *n, hd, s.as_mut_ptr().add(col), cols, 1, false, qc, 3 * c, 1, k.0.add(h * hd), 1, c, scale);
                col += n;
            }
            gemm_strided(rows, r1, hd, s.as_mut_ptr().add(p), cols, 1, false, qc, 3 * c, 1, kp, 1, 3 * c, scale);
            causal_softmax_rows(&mut s, cols, p + r0);
            let op = o.0.add((bb * l + r0) * c + h * hd);
            let mut col = 0;
            for (_, v, n) in &segs {
                gemm_strided(rows, hd, *n, op, c, 1, col > 0, s.as_ptr().add(col), cols, 1, v.0.add(h * hd), c, 1, 1.0);
                col += n;
            }
            gemm_strided(rows, hd, r1, op, c, 1, p > 0, s.as_ptr().add(p), cols, 1, vp, 3 * c, 1, 1.0);
        }
    });
}

// ----------------------------------------------------------------- SwiGLU

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// h = silu(a) * b, where each row of `ab` is [a (H) | b (H)].
pub fn swiglu_fwd(h: &mut [f32], ab: &[f32], hidden: usize) {
    h.par_chunks_mut(hidden).zip(ab.par_chunks(2 * hidden)).with_min_len(MIN_ROWS).for_each(|(hr, abr)| {
        let (a, bb) = abr.split_at(hidden);
        for i in 0..hidden {
            hr[i] = a[i] * sigmoid(a[i]) * bb[i];
        }
    });
}

pub fn swiglu_bwd(dab: &mut [f32], dh: &[f32], ab: &[f32], hidden: usize) {
    dab.par_chunks_mut(2 * hidden)
        .zip(dh.par_chunks(hidden))
        .zip(ab.par_chunks(2 * hidden))
        .with_min_len(MIN_ROWS)
        .for_each(|((d, dhr), abr)| {
            let (a, bb) = abr.split_at(hidden);
            let (da, db) = d.split_at_mut(hidden);
            for i in 0..hidden {
                let s = sigmoid(a[i]);
                da[i] = dhr[i] * bb[i] * s * (1.0 + a[i] * (1.0 - s));
                db[i] = dhr[i] * a[i] * s;
            }
        });
}

// ---------------------------------------------------- loss and softmaxes

/// Log-softmax of one row, in place.
pub fn log_softmax(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    row.iter_mut().for_each(|v| *v -= lse);
}

/// Mean cross-entropy of `targets` under softmax(logits), without gradients.
pub fn cross_entropy(logits: &[f32], targets: &[u32], v: usize) -> f32 {
    let total: f64 = logits
        .par_chunks(v)
        .zip(targets.par_iter())
        .map(|(row, &tgt)| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
            (lse - row[tgt as usize]) as f64
        })
        .sum();
    (total / targets.len() as f64) as f32
}

/// Mean cross-entropy, and turns `logits` into d(loss)/d(logits) in place.
pub fn cross_entropy_grad(logits: &mut [f32], targets: &[u32], v: usize) -> f32 {
    let n = targets.len() as f32;
    let total: f64 = logits
        .par_chunks_mut(v)
        .zip(targets.par_iter())
        .map(|(row, &tgt)| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            let loss = -(row[tgt as usize] / sum).ln();
            for x in row.iter_mut() {
                *x /= sum * n;
            }
            row[tgt as usize] -= 1.0 / n;
            loss as f64
        })
        .sum();
    (total / n as f64) as f32
}

// ---------------------------------------------- dropout, residual, embed

/// Fill `mask` with 0 (dropped, probability p) or 1/(1-p) (kept).
pub fn dropout_mask(mask: &mut [f32], p: f32, seed: u64) {
    let keep = 1.0 / (1.0 - p);
    mask.par_chunks_mut(4096).enumerate().for_each(|(i, chunk)| {
        let mut rng = Rng::fork(seed, i as u64);
        chunk.iter_mut().for_each(|m| *m = if rng.uniform() < p { 0.0 } else { keep });
    });
}

/// out = x + y (* mask)
pub fn residual(out: &mut [f32], x: &[f32], y: &[f32], mask: Option<&[f32]>) {
    match mask {
        Some(m) => out.par_iter_mut().zip(x).zip(y).zip(m).with_min_len(MIN_ELEMS).for_each(|(((o, a), b), k)| *o = a + b * k),
        None => out.par_iter_mut().zip(x).zip(y).with_min_len(MIN_ELEMS).for_each(|((o, a), b)| *o = a + b),
    }
}

pub fn embed(out: &mut [f32], wte: &[f32], tokens: &[u32], c: usize) {
    out.par_chunks_mut(c).zip(tokens.par_iter()).with_min_len(MIN_ROWS).for_each(|(o, &t)| {
        o.copy_from_slice(&wte[t as usize * c..(t as usize + 1) * c]);
    });
}

#[inline]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    y.iter_mut().zip(x).for_each(|(yi, xi)| *yi += a * xi);
}

pub fn embed_bwd(dwte: &mut [f32], dx: &[f32], tokens: &[u32], c: usize) {
    for (row, &t) in tokens.iter().enumerate() {
        axpy(&mut dwte[t as usize * c..(t as usize + 1) * c], 1.0, &dx[row * c..(row + 1) * c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    let av = if ta { a[p * m + i] } else { a[i * k + p] };
                    let bv = if tb { b[j * k + p] } else { b[p * n + j] };
                    c[i * n + j] += av * bv;
                }
            }
        }
        c
    }

    #[test]
    fn matmul_all_transposes_and_accumulate() {
        let mut rng = Rng::new(3);
        let (m, k, n) = (37, 29, 41);
        let a: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.normal()).collect();
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            let want = naive(&a, &b, m, k, n, ta, tb);
            let mut c = vec![1f32; m * n];
            matmul(&mut c, &a, &b, m, k, n, ta, tb, false);
            assert!(c.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-3));
            matmul(&mut c, &a, &b, m, k, n, ta, tb, true);
            assert!(c.iter().zip(&want).all(|(x, y)| (x - 2.0 * y).abs() < 2e-3));
        }
    }

    /// Plain causal attention for one (sequence, head): (out, dq, dk, dv).
    #[allow(clippy::type_complexity)]
    fn reference_attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>], dout: &[Vec<f32>]) -> [Vec<Vec<f32>>; 4] {
        let (t, hd) = (q.len(), q[0].len());
        let scale = 1.0 / (hd as f32).sqrt();
        let dotp = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let mut p = vec![vec![0f32; t]; t];
        for i in 0..t {
            let s: Vec<f32> = (0..=i).map(|j| dotp(&q[i], &k[j]) * scale).collect();
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = s.iter().map(|x| (x - m).exp()).sum();
            for j in 0..=i {
                p[i][j] = (s[j] - m).exp() / z;
            }
        }
        let mut out = vec![vec![0f32; hd]; t];
        let (mut dq, mut dk, mut dv) = (vec![vec![0f32; hd]; t], vec![vec![0f32; hd]; t], vec![vec![0f32; hd]; t]);
        for i in 0..t {
            for j in 0..=i {
                for d in 0..hd {
                    out[i][d] += p[i][j] * v[j][d];
                    dv[j][d] += p[i][j] * dout[i][d];
                }
            }
            let dp: Vec<f32> = (0..=i).map(|j| dotp(&dout[i], &v[j])).collect();
            let rs: f32 = (0..=i).map(|j| p[i][j] * dp[j]).sum();
            for j in 0..=i {
                let ds = p[i][j] * (dp[j] - rs) * scale;
                for d in 0..hd {
                    dq[i][d] += ds * k[j][d];
                    dk[j][d] += ds * q[i][d];
                }
            }
        }
        [out, dq, dk, dv]
    }

    #[test]
    fn chunked_attention_matches_plain_attention() {
        let (b, t, c, nh) = (2, 2 * ATT_CHUNK + 37, 16, 2);
        let hd = c / nh;
        let mut rng = Rng::new(21);
        let qkv: Vec<f32> = (0..b * t * 3 * c).map(|_| rng.normal()).collect();
        let dout: Vec<f32> = (0..b * t * c).map(|_| rng.normal()).collect();
        let mut out = vec![0f32; b * t * c];
        let mut lse = vec![0f32; b * nh * t];
        attention_fwd(&mut out, &mut lse, &qkv, b, t, c, nh);
        let mut dqkv = vec![7f32; b * t * 3 * c]; // stale values must not leak in
        attention_bwd(&mut dqkv, &dout, &qkv, &out, &lse, b, t, c, nh);
        let close = |a: f32, b: f32| (a - b).abs() < 2e-3 * (1.0 + b.abs());
        for bb in 0..b {
            for h in 0..nh {
                let col = |base: &[f32], stride: usize, off: usize| -> Vec<Vec<f32>> {
                    (0..t).map(|i| base[(bb * t + i) * stride + off + h * hd..][..hd].to_vec()).collect()
                };
                let [ro, rdq, rdk, rdv] = reference_attention(&col(&qkv, 3 * c, 0), &col(&qkv, 3 * c, c), &col(&qkv, 3 * c, 2 * c), &col(&dout, c, 0));
                for (got, want, name) in [(col(&out, c, 0), ro, "out"), (col(&dqkv, 3 * c, 0), rdq, "dq"), (col(&dqkv, 3 * c, c), rdk, "dk"),
                                          (col(&dqkv, 3 * c, 2 * c), rdv, "dv")] {
                    for i in 0..t {
                        for d in 0..hd {
                            assert!(close(got[i][d], want[i][d]), "{name} seq {bb} head {h} row {i}: {} vs {}", got[i][d], want[i][d]);
                        }
                    }
                }
            }
        }

        // inference over a cached prefix in two segments gives the same rows
        let (p1, p2) = (100, ATT_CHUNK + 20);
        let l = t - p1 - p2;
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for i in 0..p1 + p2 {
            ks.extend_from_slice(&qkv[i * 3 * c + c..i * 3 * c + 2 * c]);
            vs.extend_from_slice(&qkv[i * 3 * c + 2 * c..i * 3 * c + 3 * c]);
        }
        let segs = [(&ks[..p1 * c], &vs[..p1 * c]), (&ks[p1 * c..], &vs[p1 * c..])];
        for new in [l, 3] {
            let rows = &qkv[(p1 + p2) * 3 * c..(p1 + p2 + new) * 3 * c];
            let mut got = vec![0f32; new * c];
            attention_infer(&mut got, rows, &segs, 1, new, c, nh);
            for i in 0..new * c {
                assert!(close(got[i], out[(p1 + p2) * c + i]), "inference row {} ({new} new)", i / c);
            }
        }
    }

    #[test]
    fn rope_inverse_undoes_rotation() {
        let rope = Rope::new(8, 4, 10000.0);
        let mut rng = Rng::new(1);
        let orig: Vec<f32> = (0..5 * 3 * 8).map(|_| rng.normal()).collect();
        let mut x = orig.clone();
        rope.apply(&mut x, 8, 2, |r| r, false);
        assert!(x.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-3));
        rope.apply(&mut x, 8, 2, |r| r, true);
        assert!(x.iter().zip(&orig).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn dropout_mask_rate() {
        let mut m = vec![0f32; 100_000];
        dropout_mask(&mut m, 0.25, 9);
        let dropped = m.iter().filter(|&&v| v == 0.0).count() as f32 / m.len() as f32;
        assert!((dropped - 0.25).abs() < 0.01);
        assert!(m.iter().all(|&v| v == 0.0 || (v - 1.0 / 0.75).abs() < 1e-6));
    }
}
