//! Weight quantization for inference: the trained model's matrices stored as
//! 8-bit or 4-bit integers plus scales, so it takes 4x or ~7x less memory and
//! generates faster (one-token-at-a-time generation is limited by how fast
//! weights stream from memory). Training always uses full f32 weights; this
//! is applied when a trained model is loaded to answer questions.
//!
//!   f32   exact
//!   int8  one scale per output row;               ~4x smaller, near-identical answers
//!   int4  one scale per 32 weights along a row;   ~6.4x smaller, small accuracy cost
//!
//! Every matrix is stored as rows = outputs (n rows of k inputs), so one
//! output is a dot product with one contiguous row.

use crate::kernels;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    Int8,
    Int4,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self { Precision::F32 => "f32", Precision::Int8 => "int8", Precision::Int4 => "int4" })
    }
}

pub const GROUP: usize = 32;
/// Batches up to this many rows use the integer kernels directly; bigger
/// batches (prompt prefill) dequantize once and use the f32 matrix multiply.
const DIRECT_ROWS: usize = 8;
/// Output rows per parallel task in the one-row kernels.
const ROWS_PER_TASK: usize = 256;
/// Weights per parallel task in the one-row kernels.
const PARALLEL_WEIGHTS: usize = 256 * 1024;

pub enum QMat {
    F32 { rows: usize, cols: usize, w: Vec<f32> },
    Int8 { rows: usize, cols: usize, q: Vec<i8>, scale: Vec<f32> },
    Int4 { rows: usize, cols: usize, q: Vec<u8>, scale: Vec<f32> },
}

fn groups(cols: usize) -> usize {
    cols.div_ceil(GROUP)
}

/// Bytes per int4 row. Within each group of 32 weights, byte i holds weight
/// i (low nibble) and weight i+16 (high nibble), so both halves line up with
/// contiguous inputs in the dot product.
fn packed(cols: usize) -> usize {
    (cols / GROUP) * (GROUP / 2) + (cols % GROUP).min(GROUP / 2)
}

/// (byte index within the row, is it the high nibble) for weight k.
#[inline]
fn nibble_at(k: usize) -> (usize, bool) {
    let (g, i) = (k / GROUP, k % GROUP);
    (g * (GROUP / 2) + i % (GROUP / 2), i >= GROUP / 2)
}

impl QMat {
    /// From a row-major rows x cols matrix (rows = outputs).
    pub fn from_rows(rows: usize, cols: usize, w: &[f32], precision: Precision) -> Self {
        assert_eq!(w.len(), rows * cols);
        match precision {
            Precision::F32 => QMat::F32 { rows, cols, w: w.to_vec() },
            Precision::Int8 => {
                let mut q = vec![0i8; rows * cols];
                let mut scale = vec![0f32; rows];
                q.par_chunks_mut(cols).zip(scale.par_iter_mut()).zip(w.par_chunks(cols)).for_each(|((qr, s), wr)| {
                    let max = wr.iter().fold(0f32, |m, v| m.max(v.abs()));
                    *s = if max > 0.0 { max / 127.0 } else { 1.0 };
                    let inv = 1.0 / *s;
                    qr.iter_mut().zip(wr).for_each(|(q, &v)| *q = (v * inv).round().clamp(-127.0, 127.0) as i8);
                });
                QMat::Int8 { rows, cols, q, scale }
            }
            Precision::Int4 => {
                let (g, p) = (groups(cols), packed(cols));
                let mut q = vec![0u8; rows * p];
                let mut scale = vec![0f32; rows * g];
                q.par_chunks_mut(p).zip(scale.par_chunks_mut(g)).zip(w.par_chunks(cols)).for_each(|((qr, sr), wr)| {
                    for (gi, chunk) in wr.chunks(GROUP).enumerate() {
                        // The largest-magnitude weight maps exactly to -8, so all
                        // 16 levels (-8..=7) are in use and the extreme is exact.
                        let peak = chunk.iter().fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                        sr[gi] = if peak != 0.0 { peak / -8.0 } else { 1.0 };
                        let inv = 1.0 / sr[gi];
                        for (i, &v) in chunk.iter().enumerate() {
                            let nib = ((v * inv).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                            let (byte, high) = nibble_at(gi * GROUP + i);
                            qr[byte] |= if high { nib << 4 } else { nib };
                        }
                    }
                });
                QMat::Int4 { rows, cols, q, scale }
            }
        }
    }

    /// From a k x n matrix stored inputs-major (how the training weights are laid out).
    pub fn from_inputs_major(k: usize, n: usize, w: &[f32], precision: Precision) -> Self {
        let mut t = vec![0f32; n * k];
        t.par_chunks_mut(k).enumerate().for_each(|(j, row)| {
            for i in 0..k {
                row[i] = w[i * n + j];
            }
        });
        Self::from_rows(n, k, &t, precision)
    }

    pub fn rows(&self) -> usize {
        match self { QMat::F32 { rows, .. } | QMat::Int8 { rows, .. } | QMat::Int4 { rows, .. } => *rows }
    }

    pub fn cols(&self) -> usize {
        match self { QMat::F32 { cols, .. } | QMat::Int8 { cols, .. } | QMat::Int4 { cols, .. } => *cols }
    }

    /// Memory used by the weights.
    pub fn bytes(&self) -> usize {
        match self {
            QMat::F32 { w, .. } => 4 * w.len(),
            QMat::Int8 { q, scale, .. } => q.len() + 4 * scale.len(),
            QMat::Int4 { q, scale, .. } => q.len() + 4 * scale.len(),
        }
    }

    /// Dequantize one row into `out` (length cols).
    pub fn row(&self, r: usize, out: &mut [f32]) {
        match self {
            QMat::F32 { cols, w, .. } => out.copy_from_slice(&w[r * cols..(r + 1) * cols]),
            QMat::Int8 { cols, q, scale, .. } => {
                let s = scale[r];
                out.iter_mut().zip(&q[r * cols..(r + 1) * cols]).for_each(|(o, &v)| *o = v as f32 * s);
            }
            QMat::Int4 { cols, q, scale, .. } => {
                let (g, p) = (groups(*cols), packed(*cols));
                let (qr, sr) = (&q[r * p..(r + 1) * p], &scale[r * g..(r + 1) * g]);
                let mut vals = [0f32; GROUP];
                for (gi, (o, s)) in out.chunks_mut(GROUP).zip(sr).enumerate() {
                    unpack_group(&qr[gi * GROUP / 2..], o.len(), &mut vals);
                    o.iter_mut().zip(&vals).for_each(|(o, v)| *o = v * s);
                }
            }
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let (rows, cols) = (self.rows(), self.cols());
        let mut out = vec![0f32; rows * cols];
        out.par_chunks_mut(cols * ROWS_PER_TASK).enumerate().for_each(|(ci, block)| {
            for (i, o) in block.chunks_mut(cols).enumerate() {
                self.row(ci * ROWS_PER_TASK + i, o);
            }
        });
        out
    }

    /// y (m x rows) = x (m x cols) . W^T
    pub fn matmul(&self, y: &mut [f32], x: &[f32], m: usize) {
        let (n, k) = (self.rows(), self.cols());
        assert!(y.len() >= m * n && x.len() >= m * k);
        match self {
            QMat::F32 { w, .. } if m > DIRECT_ROWS => kernels::matmul(y, x, w, m, k, n, false, true, false),
            _ if m > DIRECT_ROWS => {
                let w = self.dequantize();
                kernels::matmul(y, x, &w, m, k, n, false, true, false);
            }
            _ => {
                // Split across threads only when there's enough work per task.
                let rows_per_task = (PARALLEL_WEIGHTS / k.max(1)).max(ROWS_PER_TASK);
                for (yr, xr) in y.chunks_mut(n).take(m).zip(x.chunks(k)) {
                    yr.par_chunks_mut(rows_per_task).enumerate().for_each(|(ci, ys)| gemv(self, xr, ys, ci * rows_per_task));
                }
            }
        }
    }

}

// ---------------------------------------------------------------------
// One-row kernels (y = W . x), used when generating a token at a time.
// On x86 CPUs with AVX2 + FMA (virtually every one since 2013) the integer
// dot products use hand-written SIMD; elsewhere, plain loops. The check is
// done once per call, so one binary runs everywhere at full speed.

fn gemv(m: &QMat, x: &[f32], y: &mut [f32], r0: usize) {
    #[cfg(target_arch = "x86_64")]
    let simd = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    match m {
        QMat::F32 { cols, w, .. } => {
            for (i, yi) in y.iter_mut().enumerate() {
                let r = r0 + i;
                *yi = kernels::dot(x, &w[r * cols..(r + 1) * cols]);
            }
        }
        QMat::Int8 { cols, q, scale, .. } => {
            for (i, yi) in y.iter_mut().enumerate() {
                let r = r0 + i;
                let row = &q[r * cols..(r + 1) * cols];
                #[cfg(target_arch = "x86_64")]
                if simd {
                    // SAFETY: AVX2 and FMA are available (checked above); lengths match.
                    *yi = unsafe { simd_x86::dot_i8(x, row) } * scale[r];
                    continue;
                }
                let _ = simd;
                *yi = x.iter().zip(row).map(|(a, &b)| a * b as f32).sum::<f32>() * scale[r];
            }
        }
        QMat::Int4 { cols, q, scale, .. } => {
            let (g, p) = (groups(*cols), packed(*cols));
            for (i, yi) in y.iter_mut().enumerate() {
                let r = r0 + i;
                let (qr, sr) = (&q[r * p..(r + 1) * p], &scale[r * g..(r + 1) * g]);
                let full = cols / GROUP;
                let mut total;
                #[cfg(target_arch = "x86_64")]
                {
                    total = if simd {
                        // SAFETY: AVX2 and FMA are available; `full` groups fit in x, qr and sr.
                        unsafe { simd_x86::dot_i4_groups(x, qr, sr, full) }
                    } else {
                        dot_i4_groups(x, qr, sr, full)
                    };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    total = dot_i4_groups(x, qr, sr, full);
                }
                if full < g {
                    let mut vals = [0f32; GROUP];
                    let xg = &x[full * GROUP..*cols];
                    unpack_group(&qr[full * (GROUP / 2)..], xg.len(), &mut vals);
                    total += xg.iter().zip(&vals).map(|(a, b)| a * b).sum::<f32>() * sr[full];
                }
                *yi = total;
            }
        }
    }
}

/// Full 32-weight int4 groups, portable version.
fn dot_i4_groups(x: &[f32], qr: &[u8], sr: &[f32], full: usize) -> f32 {
    let mut acc = [0f32; GROUP / 2];
    for gi in 0..full {
        let xg = &x[gi * GROUP..(gi + 1) * GROUP];
        let b = &qr[gi * (GROUP / 2)..(gi + 1) * (GROUP / 2)];
        for l in 0..GROUP / 2 {
            let lo = (b[l] & 15) as f32 - 8.0;
            let hi = (b[l] >> 4) as f32 - 8.0;
            acc[l] += sr[gi] * (xg[l] * lo + xg[l + GROUP / 2] * hi);
        }
    }
    acc.iter().sum()
}

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// x . q for int8 q, 32 at a time with four accumulators.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_i8(x: &[f32], q: &[i8]) -> f32 {
        let k = x.len().min(q.len());
        let (xp, qp) = (x.as_ptr(), q.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= k {
            let a = _mm_loadu_si128(qp.add(i) as *const __m128i);
            let b = _mm_loadu_si128(qp.add(i + 16) as *const __m128i);
            let parts = [a, _mm_srli_si128(a, 8), b, _mm_srli_si128(b, 8)];
            for (j, part) in parts.into_iter().enumerate() {
                let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(part));
                acc[j] = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 8 * j)), w, acc[j]);
            }
            i += 32;
        }
        let mut total = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        while i < k {
            total += *xp.add(i) * *qp.add(i) as f32;
            i += 1;
        }
        total
    }

    /// Full 32-weight int4 groups: byte i of a group holds weight i (low
    /// nibble) and weight i+16 (high nibble), stored as value + 8.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_i4_groups(x: &[f32], qr: &[u8], sr: &[f32], full: usize) -> f32 {
        assert!(x.len() >= full * 32 && qr.len() >= full * 16 && sr.len() >= full);
        let (xp, qp) = (x.as_ptr(), qr.as_ptr());
        let mask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_ps(8.0);
        let mut acc = _mm256_setzero_ps();
        for g in 0..full {
            let bytes = _mm_loadu_si128(qp.add(g * 16) as *const __m128i);
            let lo = _mm_and_si128(bytes, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            let xg = xp.add(g * 32);
            let mut part = _mm256_setzero_ps();
            for (j, nibbles) in [lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)].into_iter().enumerate() {
                let w = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(nibbles)), eight);
                part = _mm256_fmadd_ps(_mm256_loadu_ps(xg.add(8 * j)), w, part);
            }
            acc = _mm256_fmadd_ps(part, _mm256_set1_ps(*sr.get_unchecked(g)), acc);
        }
        hsum(acc)
    }
}

/// Unpack a group of `len` (<= 32) 4-bit weights to -8..=7.
#[inline]
fn unpack_group(bytes: &[u8], len: usize, out: &mut [f32; GROUP]) {
    let half = GROUP / 2;
    for i in 0..len.min(half) {
        out[i] = (bytes[i] & 15) as f32 - 8.0;
    }
    for i in half..len {
        out[i] = (bytes[i - half] >> 4) as f32 - 8.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| r.normal() * 0.05).collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum();
        (num / den).sqrt()
    }

    #[test]
    fn quantized_matmul_matches_f32_both_paths() {
        let (n, k) = (70, 100); // not multiples of 32 or 64, on purpose
        let w = random(n * k, 1);
        let reference = QMat::from_rows(n, k, &w, Precision::F32);
        for (precision, tol) in [(Precision::Int8, 0.01), (Precision::Int4, 0.12)] {
            let qm = QMat::from_rows(n, k, &w, precision);
            for m in [1, 3, 20] {
                let x = random(m * k, 2 + m as u64);
                let (mut want, mut got) = (vec![0f32; m * n], vec![0f32; m * n]);
                reference.matmul(&mut want, &x, m);
                qm.matmul(&mut got, &x, m);
                let e = rel_err(&got, &want);
                assert!(e < tol, "{precision} m={m}: relative error {e}");
            }
            let mut row = vec![0f32; k];
            qm.row(5, &mut row);
            assert!(rel_err(&row, &w[5 * k..6 * k]) < tol);
        }
    }

    #[test]
    fn sizes() {
        let w = random(64 * 128, 3);
        let f = QMat::from_rows(64, 128, &w, Precision::F32).bytes();
        let i8b = QMat::from_rows(64, 128, &w, Precision::Int8).bytes();
        let i4b = QMat::from_rows(64, 128, &w, Precision::Int4).bytes();
        assert!(i8b * 3 < f && i4b * 6 < f, "{f} {i8b} {i4b}");
    }

    #[test]
    fn inputs_major_is_transposed() {
        let (k, n) = (3, 2);
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // k x n
        let m = QMat::from_inputs_major(k, n, &w, Precision::F32);
        let mut y = vec![0f32; 2];
        m.matmul(&mut y, &[1.0, 1.0, 1.0], 1);
        assert_eq!(y, vec![9.0, 12.0]); // column sums
    }
}
