//! Small, fast, seedable random numbers (xoshiro256**). Written here so
//! every random choice in training and sampling is reproducible from a seed.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 spreads a single seed across the 256-bit state.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    /// Seeded from the clock: different every run.
    pub fn from_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Rng::new(nanos ^ (std::process::id() as u64).rotate_left(32))
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * n as f64) as usize
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.uniform() as f64).max(1e-12);
        let u2 = self.uniform() as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// A child generator for one independent stream (e.g. one thread's chunk).
    pub fn fork(seed: u64, stream: u64) -> Self {
        Rng::new(seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_well_spread() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        assert_eq!(a.next_u64(), b.next_u64());
        let mut r = Rng::new(1);
        let n = 100_000;
        let mean: f32 = (0..n).map(|_| r.uniform()).sum::<f32>() / n as f32;
        assert!((mean - 0.5).abs() < 0.01);
        let normals: Vec<f32> = (0..n).map(|_| r.normal()).collect();
        let m = normals.iter().sum::<f32>() / n as f32;
        let var = normals.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n as f32;
        assert!(m.abs() < 0.02 && (var - 1.0).abs() < 0.03);
        assert!((0..1000).all(|_| r.below(10) < 10));
    }
}
