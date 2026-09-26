//! This machine's RAM, cores and SIMD support, and the model size (tier)
//! that fits it. The tier only sizes a NEW model; a trained one keeps its size.

use crate::config::Tier;
use std::collections::BTreeMap;

pub struct Hardware {
    pub ram_gb: f64,
    pub cpu_cores: usize,
    pub threads: usize,
    pub simd: &'static str,
}

pub fn detect() -> Hardware {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Hardware {
        ram_gb: sys.total_memory() as f64 / (1u64 << 30) as f64,
        cpu_cores: sys.physical_core_count().unwrap_or(threads),
        threads,
        simd: simd_level(),
    }
}

fn simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512";
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2 + FMA";
        }
        return "SSE";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON";
    }
    #[allow(unreachable_code)]
    "none detected"
}

/// First automatic tier (smallest max_ram_gb first) whose limit this much
/// memory fits under; anything bigger gets the largest. Manual tiers are
/// only used when asked for by name.
pub fn pick_tier(ram_gb: f64, tiers: &BTreeMap<String, Tier>) -> String {
    let mut ordered: Vec<_> = tiers.iter().filter(|(_, t)| !t.manual).collect();
    ordered.sort_by(|a, b| a.1.max_ram_gb.total_cmp(&b.1.max_ram_gb));
    ordered
        .iter()
        .find(|(_, t)| ram_gb <= t.max_ram_gb)
        .or(ordered.last())
        .map(|(name, _)| name.to_string())
        .expect("settings.yaml has no model.tiers")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;
    use std::path::PathBuf;

    #[test]
    fn tiers_follow_settings_thresholds() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let s = Settings::from_file(&root.join("config/aegist.yaml"), &root).unwrap();
        for (gb, want) in [(7.6, "tiny"), (8.0, "tiny"), (15.5, "small"), (31.2, "medium"), (64.0, "large"), (1024.0, "large")] {
            assert_eq!(pick_tier(gb, &s.model.tiers), want, "{gb} GB");
        }
        let hw = detect();
        assert!(hw.ram_gb > 0.0 && hw.cpu_cores >= 1 && hw.threads >= 1);
    }
}
