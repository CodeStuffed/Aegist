//! A pretend GPU for tests: device memory is host memory, kernels run
//! through the CPU emulator in emulate.cpp (the same kernels.cu source the
//! GPU compiles), and matrix multiplies follow cuBLAS's column-major rules
//! exactly - so the whole GPU training step can be checked against the CPU
//! model on machines without an NVIDIA card.

use super::{Arg, Backend, CublasGemm, DevPtr, Gemm, Kernel};

/// BLOCK in emulate.cpp
const EMU_BLOCK: u32 = 8;
use anyhow::{bail, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};

extern "C" {
    fn council_emu_launch(name: *const c_char, gx: u32, gy: u32, bx: u32, params: *mut *mut c_void) -> i32;
}

#[derive(Default)]
pub struct EmulatedBackend {
    allocs: RefCell<HashMap<DevPtr, Vec<u64>>>,
}

impl EmulatedBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

/// cuBLAS sgemmStridedBatched, by the book (column-major).
///
/// Safety: every pointer must address enough f32s for the shapes given.
unsafe fn sgemm_ref(g: &CublasGemm) {
    let (m, n, k) = (g.m as usize, g.n as usize, g.k as usize);
    for bi in 0..g.batch as usize {
        let a = (g.a as *const f32).add(bi * g.stride_a as usize);
        let b = (g.b as *const f32).add(bi * g.stride_b as usize);
        let c = (g.c as *mut f32).add(bi * g.stride_c as usize);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    let av = if g.trans_a { *a.add(p + i * g.lda as usize) } else { *a.add(i + p * g.lda as usize) };
                    let bv = if g.trans_b { *b.add(j + p * g.ldb as usize) } else { *b.add(p + j * g.ldb as usize) };
                    acc += av * bv;
                }
                let out = c.add(i + j * g.ldc as usize);
                *out = g.alpha * acc + if g.beta == 0.0 { 0.0 } else { g.beta * *out };
            }
        }
    }
}

/// As sgemm_ref, with A and B in bfloat16.
///
/// Safety: as sgemm_ref (A and B hold u16s).
unsafe fn gemm_bf16_ref(g: &CublasGemm) {
    let (m, n, k) = (g.m as usize, g.n as usize, g.k as usize);
    let f = |p: *const u16, i: usize| f32::from_bits((*p.add(i) as u32) << 16);
    for bi in 0..g.batch as usize {
        let a = (g.a as *const u16).add(bi * g.stride_a as usize);
        let b = (g.b as *const u16).add(bi * g.stride_b as usize);
        let c = (g.c as *mut f32).add(bi * g.stride_c as usize);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    let av = if g.trans_a { f(a, p + i * g.lda as usize) } else { f(a, i + p * g.lda as usize) };
                    let bv = if g.trans_b { f(b, j + p * g.ldb as usize) } else { f(b, p + j * g.ldb as usize) };
                    acc += av * bv;
                }
                let out = c.add(i + j * g.ldc as usize);
                *out = g.alpha * acc + if g.beta == 0.0 { 0.0 } else { g.beta * *out };
            }
        }
    }
}

impl Backend for EmulatedBackend {
    fn describe(&self) -> String {
        "CPU emulation of the GPU kernels".into()
    }

    fn alloc(&self, bytes: usize) -> Result<DevPtr> {
        let v = vec![0u64; bytes.div_ceil(8).max(1)];
        let p = v.as_ptr() as DevPtr;
        self.allocs.borrow_mut().insert(p, v);
        Ok(p)
    }

    fn free(&self, p: DevPtr) {
        self.allocs.borrow_mut().remove(&p);
    }

    fn upload(&self, dst: DevPtr, src: &[u8]) -> Result<()> {
        // Safety: dst is an allocation at least src.len() bytes long (as on a GPU, the caller's job).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len()) };
        Ok(())
    }

    fn download(&self, src: DevPtr, dst: &mut [u8]) -> Result<()> {
        // Safety: as upload.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    fn zero(&self, dst: DevPtr, bytes: usize) -> Result<()> {
        // Safety: as upload.
        unsafe { std::ptr::write_bytes(dst as *mut u8, 0, bytes) };
        Ok(())
    }

    fn block(&self) -> u32 {
        EMU_BLOCK
    }

    fn max_grid(&self) -> u32 {
        4 // grid-stride loops cover the rest; each emulated block costs barriers
    }

    fn launch(&self, k: Kernel, grid: (u32, u32), args: &[Arg]) -> Result<()> {
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut c_void).collect();
        let name = CString::new(k.name()).unwrap();
        // Safety: one pointer per kernel parameter, each to a value of its size.
        if unsafe { council_emu_launch(name.as_ptr(), grid.0, grid.1, EMU_BLOCK, ptrs.as_mut_ptr()) } != 0 {
            bail!("the emulator has no kernel {}", k.name());
        }
        Ok(())
    }

    fn gemm(&self, g: &Gemm) -> Result<()> {
        // Safety: as upload - buffers sized by the caller.
        unsafe { sgemm_ref(&g.to_cublas()) };
        Ok(())
    }

    fn gemm_bf16(&self, g: &Gemm) -> Result<()> {
        // Safety: as gemm.
        unsafe { gemm_bf16_ref(&g.to_cublas()) };
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }

    fn memory(&self) -> Result<(u64, u64)> {
        Ok((16 << 30, 16 << 30))
    }

    fn set_tf32(&self, _on: bool) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{self_check, GpuTrainer};
    use crate::model::{Acts, Model, ModelConfig};
    use crate::optim::AdamW;

    #[test]
    fn the_gpu_step_matches_the_cpu_model() {
        let r = self_check(EmulatedBackend::new()).unwrap();
        assert!(r.passed, "{r:?}");
        assert!(r.worst_grad.1 < 1e-4, "{r:?}");
        let bf16 = r.bf16_grad.clone().unwrap().1;
        eprintln!("{r:?}");
        assert!(bf16 > 1e-4 && bf16 < 2e-2, "bf16 gradients {bf16}: should be close, but visibly rounded");
    }

    #[test]
    fn train_on_the_gpu_then_resume_on_the_cpu() {
        use crate::trainer::{self, Budget, Compute, Size};
        use std::sync::atomic::AtomicBool;
        let tmp = tempfile::tempdir().unwrap();
        let mut s = crate::config::testing::settings(&tmp.path().join("data"));
        s.gpu.tokens_per_step = 4 * 48;
        s.gpu.learning_rate = 3e-3;
        s.gpu.warmup_steps = 5;
        let src = tmp.path().join("notes.txt");
        std::fs::write(&src, trainer::tests::CORPUS.repeat(20)).unwrap();
        crate::corpus::import_texts(&src, &s).unwrap();
        let gpu = Compute::Gpu { backend: Box::new(EmulatedBackend::new()), flops: 1e12, bf16: true };
        let mut lines = Vec::new();
        let r = trainer::train_on(&s, Budget::Steps(40), Size::FromRam, &gpu, &mut |l| lines.push(l), Some(0), &AtomicBool::new(false)).unwrap();
        assert!(lines.iter().any(|l| l.contains("Training on the GPU: 192 tokens per step")), "{lines:?}");
        let vocab = r.stats.val_loss.map(|_| crate::checkpoint::load(&s).unwrap().unwrap().0.cfg.vocab_size).unwrap() as f32;
        let after_gpu = r.stats.val_loss.unwrap();
        assert!(after_gpu < vocab.ln() - 1.0, "held-out loss {after_gpu} after training on the GPU");
        // the checkpoint (weights and AdamW state from the device) carries on on the CPU
        let (_, _, _) = crate::checkpoint::load(&s).unwrap().unwrap();
        let cpu = Compute::Cpu { flops: 1e9, ram_gb: 64.0 };
        let r2 = trainer::train_on(&s, Budget::Steps(10), Size::FromRam, &cpu, &mut |_| {}, Some(1), &AtomicBool::new(false)).unwrap();
        assert_eq!(r2.stats.steps, 50);
        assert!(r2.stats.val_loss.unwrap() < after_gpu + 0.3, "{} then {}", after_gpu, r2.stats.val_loss.unwrap());
        let mut opt = AdamW::new(1, &[], 0.0);
        let model = crate::checkpoint::load(&s).unwrap().unwrap().0;
        opt.m = vec![0.0; model.num_params()];
        opt.v = vec![0.0; model.num_params()];
        crate::checkpoint::load_optimizer(&s, &mut opt);
        assert_eq!(opt.t, 50);
    }

    #[test]
    fn eval_loss_and_dropout_training() {
        let cfg = ModelConfig { vocab_size: 97, block_size: 16, n_layer: 2, n_head: 2, d_model: 32, mlp_hidden: 64, dropout: 0.2, rope_base: 1e4 };
        let model = Model::new(cfg.clone(), 3);
        let mut rng = crate::rng::Rng::new(2);
        let n = 4 * cfg.block_size;
        let x: Vec<u32> = (0..n).map(|_| rng.below(97) as u32).collect();
        let y: Vec<u32> = x.iter().map(|&t| (t * 7 + 1) % 97).collect();
        let mut acts = Acts::new(&cfg, 4, cfg.block_size);
        let cpu = model.loss(&mut acts, &x, &y);
        let opt = AdamW::new(model.num_params(), &model.layout.matrices, 0.1);
        let mut g = GpuTrainer::new(EmulatedBackend::new(), &model, &opt, 2, true).unwrap();
        assert!((g.loss(&x, &y).unwrap() - cpu).abs() < 2e-2); // bf16 matrix multiplies
        // training with dropout on lowers the loss on this (learnable) mapping
        let first = g.loss(&x, &y).unwrap();
        for step in 0..30 {
            g.forward_backward(&x, &y, Some(step)).unwrap();
            g.clip(1.0).unwrap();
            g.adamw(1e-2).unwrap();
        }
        let after = g.loss(&x, &y).unwrap();
        assert!(after < first - 1.0, "{first} -> {after}");
        assert!(g.forward_backward(&x[..10], &y[..10], None).is_err());
    }
}
