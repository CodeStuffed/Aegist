//! Training on an NVIDIA GPU.
//!
//! The same transformer as model.rs, computed on the graphics card:
//! matrix multiplies by cuBLAS (NVIDIA's matrix library, in TF32 - tensor
//! cores with ~10-bit mantissas, fp32 range and accumulation), everything
//! else by the hand-written kernels in kernels.cu, compiled when training
//! starts by NVRTC. The CUDA libraries are loaded at run time, so the same
//! `council` program runs on machines without them (it just trains on the CPU).
//!
//! To fit big models in 16 GB, only each layer's input is kept during the
//! forward pass; the backward pass recomputes one layer at a time from it
//! (activation checkpointing - about a third more compute for a fraction of
//! the memory). A step's batch is split into micro-batches whose gradients
//! add up.
//!
//! Before training, `self_check` runs a small model on both GPU and CPU and
//! compares every gradient, so a broken driver or GPU never trains silently
//! wrong.

pub mod cuda;
#[cfg(feature = "gpu-emulator")]
pub mod emulated;

use crate::model::{Layout, Model, ModelConfig};
use crate::optim::AdamW;
use anyhow::{bail, Result};

pub type DevPtr = u64;

/// The CUDA source of every kernel.
pub const KERNELS_CU: &str = include_str!("kernels.cu");
/// Threads per block on the GPU (BLOCK in kernels.cu).
pub const BLOCK: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    EmbedFwd,
    EmbedBwd,
    RmsnormFwd,
    RmsnormBwdDx,
    RmsnormBwdDg,
    Rope,
    CausalSoftmax,
    SoftmaxBwd,
    SwigluFwd,
    SwigluBwd,
    Residual,
    MaskMul,
    CrossEntropy,
    Sumsq,
    Scale,
    Adamw,
    ToBf16,
}

impl Kernel {
    pub const ALL: [Kernel; 17] = [
        Kernel::EmbedFwd, Kernel::EmbedBwd, Kernel::RmsnormFwd, Kernel::RmsnormBwdDx, Kernel::RmsnormBwdDg, Kernel::Rope,
        Kernel::CausalSoftmax, Kernel::SoftmaxBwd, Kernel::SwigluFwd, Kernel::SwigluBwd, Kernel::Residual, Kernel::MaskMul,
        Kernel::CrossEntropy, Kernel::Sumsq, Kernel::Scale, Kernel::Adamw, Kernel::ToBf16,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Kernel::EmbedFwd => "embed_fwd",
            Kernel::EmbedBwd => "embed_bwd",
            Kernel::RmsnormFwd => "rmsnorm_fwd",
            Kernel::RmsnormBwdDx => "rmsnorm_bwd_dx",
            Kernel::RmsnormBwdDg => "rmsnorm_bwd_dg",
            Kernel::Rope => "rope",
            Kernel::CausalSoftmax => "causal_softmax",
            Kernel::SoftmaxBwd => "softmax_bwd",
            Kernel::SwigluFwd => "swiglu_fwd",
            Kernel::SwigluBwd => "swiglu_bwd",
            Kernel::Residual => "residual",
            Kernel::MaskMul => "mask_mul",
            Kernel::CrossEntropy => "cross_entropy",
            Kernel::Sumsq => "sumsq",
            Kernel::Scale => "scale",
            Kernel::Adamw => "adamw",
            Kernel::ToBf16 => "to_bf16",
        }
    }
}

/// A kernel argument, passed by value.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(DevPtr),
    I32(i32),
    I64(i64),
    U64(u64),
    F32(f32),
}

impl Arg {
    /// The value's bytes in an 8-byte slot (little-endian, so 4-byte values
    /// sit at its start), as cuLaunchKernel reads them.
    pub fn slot(self) -> u64 {
        match self {
            Arg::Ptr(p) | Arg::U64(p) => p,
            Arg::I64(v) => v as u64,
            Arg::I32(v) => v as u32 as u64,
            Arg::F32(v) => v.to_bits() as u64,
        }
    }
}

/// C (m x n) = alpha * op(A) . op(B) + beta * C for `batch` matrices, all
/// row-major like the rest of the program: element (i, j) of a matrix with
/// leading dimension ld is at [i * ld + j]; op(A) is m x k (stored k x m if
/// `ta`), op(B) is k x n (stored n x k if `tb`); batch item i starts
/// stride_* elements after item i-1.
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub a: DevPtr,
    pub lda: usize,
    pub stride_a: usize,
    pub ta: bool,
    pub b: DevPtr,
    pub ldb: usize,
    pub stride_b: usize,
    pub tb: bool,
    pub beta: f32,
    pub c: DevPtr,
    pub ldc: usize,
    pub stride_c: usize,
    pub batch: usize,
}

/// The same multiply as cuBLAS (column-major) sees it: a row-major matrix is
/// its transpose in column-major, so C^T = op(B)^T . op(A)^T - B goes first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CublasGemm {
    pub trans_a: bool,
    pub trans_b: bool,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: f32,
    pub a: DevPtr,
    pub lda: i32,
    pub stride_a: i64,
    pub b: DevPtr,
    pub ldb: i32,
    pub stride_b: i64,
    pub beta: f32,
    pub c: DevPtr,
    pub ldc: i32,
    pub stride_c: i64,
    pub batch: i32,
}

impl Gemm {
    pub fn to_cublas(&self) -> CublasGemm {
        CublasGemm {
            trans_a: self.tb,
            trans_b: self.ta,
            m: self.n as i32,
            n: self.m as i32,
            k: self.k as i32,
            alpha: self.alpha,
            a: self.b,
            lda: self.ldb as i32,
            stride_a: self.stride_b as i64,
            b: self.a,
            ldb: self.lda as i32,
            stride_b: self.stride_a as i64,
            beta: self.beta,
            c: self.c,
            ldc: self.ldc as i32,
            stride_c: self.stride_c as i64,
            batch: self.batch as i32,
        }
    }
}

/// A device that runs the kernels: the real GPU, or the CPU emulator in tests.
pub trait Backend {
    fn describe(&self) -> String;
    fn alloc(&self, bytes: usize) -> Result<DevPtr>;
    fn free(&self, p: DevPtr);
    fn upload(&self, dst: DevPtr, src: &[u8]) -> Result<()>;
    fn download(&self, src: DevPtr, dst: &mut [u8]) -> Result<()>;
    fn zero(&self, dst: DevPtr, bytes: usize) -> Result<()>;
    /// Threads per block the kernels were compiled for.
    fn block(&self) -> u32 {
        BLOCK
    }
    /// Most blocks for a grid-stride loop.
    fn max_grid(&self) -> u32 {
        8192
    }
    /// Launch with `grid` blocks of `block()` threads.
    fn launch(&self, k: Kernel, grid: (u32, u32), args: &[Arg]) -> Result<()>;
    fn gemm(&self, g: &Gemm) -> Result<()>;
    /// The same with A and B in bfloat16 (C stays f32; fp32 accumulation).
    fn gemm_bf16(&self, g: &Gemm) -> Result<()>;
    /// Whether bf16 matrix multiplies run on tensor cores here (Ampere / RTX 30xx and newer).
    fn supports_bf16(&self) -> bool {
        true
    }
    fn sync(&self) -> Result<()>;
    /// (free, total) device memory in bytes.
    fn memory(&self) -> Result<(u64, u64)>;
    /// TF32 tensor-core matrix multiplies (fast) or strict fp32 (for checking).
    fn set_tf32(&self, on: bool) -> Result<()>;
}

/// A borrowed backend works too (the trainer keeps the GPU open across runs).
impl<T: Backend + ?Sized> Backend for &T {
    fn describe(&self) -> String {
        (**self).describe()
    }
    fn alloc(&self, bytes: usize) -> Result<DevPtr> {
        (**self).alloc(bytes)
    }
    fn free(&self, p: DevPtr) {
        (**self).free(p)
    }
    fn upload(&self, dst: DevPtr, src: &[u8]) -> Result<()> {
        (**self).upload(dst, src)
    }
    fn download(&self, src: DevPtr, dst: &mut [u8]) -> Result<()> {
        (**self).download(src, dst)
    }
    fn zero(&self, dst: DevPtr, bytes: usize) -> Result<()> {
        (**self).zero(dst, bytes)
    }
    fn block(&self) -> u32 {
        (**self).block()
    }
    fn max_grid(&self) -> u32 {
        (**self).max_grid()
    }
    fn launch(&self, k: Kernel, grid: (u32, u32), args: &[Arg]) -> Result<()> {
        (**self).launch(k, grid, args)
    }
    fn gemm(&self, g: &Gemm) -> Result<()> {
        (**self).gemm(g)
    }
    fn gemm_bf16(&self, g: &Gemm) -> Result<()> {
        (**self).gemm_bf16(g)
    }
    fn supports_bf16(&self) -> bool {
        (**self).supports_bf16()
    }
    fn sync(&self) -> Result<()> {
        (**self).sync()
    }
    fn memory(&self) -> Result<(u64, u64)> {
        (**self).memory()
    }
    fn set_tf32(&self, on: bool) -> Result<()> {
        (**self).set_tf32(on)
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // Safety: f32 has no invalid bit patterns and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // Safety: as above; every byte pattern is a valid f32.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    // Safety: as above.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Device memory for training `cfg` with micro-batches of `seqs` sequences
/// (bf16: plus the bf16 copy of the weights and conversion scratch).
pub fn training_bytes(cfg: &ModelConfig, seqs: usize, bf16: bool) -> u64 {
    let (c, h, v, l, nh, t) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.block_size);
    let n = seqs * t;
    let params = Layout::new(cfg).total;
    let floats = 4 * params                       // weights, gradients, two AdamW moments
        + (l + 1) * n * c                         // each layer's input (checkpoints)
        + 7 * n * c + 6 * n * c + n * c           // one layer's scratch, backward buffers, final norm
        + 6 * n * h + 4 * n                       // SwiGLU forward/backward, norms' 1/rms, losses
        + 2 * seqs * nh * t * t                   // attention weights and their gradients
        + n * v                                   // logits
        + t * c                                   // rotary tables
        + 2 * n + 2 * 4096;                       // tokens, targets, reduction scratch
    let halves = if bf16 { params + n * v.max(3 * c).max(2 * h) + n * (3 * c).max(2 * h) } else { 0 };
    4 * floats as u64 + 2 * halves as u64
}

/// Memory the CUDA context, cuBLAS and fragmentation take besides our buffers.
pub const OVERHEAD_BYTES: u64 = 700 << 20;

/// Most sequences per micro-batch that fit in `free` bytes (at most `max`).
pub fn micro_batch_for(cfg: &ModelConfig, free: u64, max: usize, bf16: bool) -> Option<usize> {
    (1..=max.max(1)).rev().find(|&s| training_bytes(cfg, s, bf16) + OVERHEAD_BYTES <= free)
}

#[derive(Clone, Copy)]
struct Buf {
    ptr: DevPtr,
}

impl Buf {
    /// Pointer to float element `i`.
    fn at(self, i: usize) -> DevPtr {
        self.ptr + 4 * i as u64
    }
}

/// Forward/backward/AdamW for one model, living on the device.
pub struct GpuTrainer<B: Backend> {
    be: B,
    pub cfg: ModelConfig,
    layout: Layout,
    n_params: usize,
    /// sequences per micro-batch
    pub seqs: usize,
    /// matrix multiplies in bf16 (else TF32/fp32)
    pub bf16: bool,
    params16: DevPtr,
    s16a: DevPtr,
    s16b: DevPtr,
    pub adam_t: u64,
    segments: Vec<(usize, usize, bool)>,
    weight_decay: f32,
    params: Buf,
    grads: Buf,
    m: Buf,
    v: Buf,
    cos: Buf,
    sin: Buf,
    tokens: Buf,
    targets: Buf,
    x: Vec<Buf>,
    xn1: Buf,
    inv1: Buf,
    qkv: Buf,
    probs: Buf,
    att: Buf,
    tmp: Buf,
    xmid: Buf,
    xn2: Buf,
    inv2: Buf,
    ab: Buf,
    hg: Buf,
    xnf: Buf,
    invf: Buf,
    logits: Buf,
    losses: Buf,
    dx: Buf,
    dn: Buf,
    dqkv: Buf,
    ds: Buf,
    dab: Buf,
    dhg: Buf,
    partial: Buf,
    allocated: Vec<DevPtr>,
}

impl<B: Backend> Drop for GpuTrainer<B> {
    fn drop(&mut self) {
        for &p in &self.allocated {
            self.be.free(p);
        }
    }
}


const PARTIALS: usize = 1024;

impl<B: Backend> GpuTrainer<B> {
    /// Put `model` (and the optimizer's state) on the device, with room for
    /// micro-batches of `seqs` sequences.
    pub fn new(be: B, model: &Model, opt: &AdamW, seqs: usize, bf16: bool) -> Result<Self> {
        let cfg = model.cfg.clone();
        let layout = Layout::new(&cfg);
        let (c, h, v, l, nh, t) = (cfg.d_model, cfg.mlp_hidden, cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.block_size);
        let n = seqs * t;
        let np = layout.total;
        let mut allocated = Vec::new();
        let mut buf = |floats: usize| -> Result<Buf> {
            let ptr = be.alloc(4 * floats.max(1))?;
            allocated.push(ptr);
            Ok(Buf { ptr })
        };
        let params = buf(np)?;
        let grads = buf(np)?;
        let m = buf(np)?;
        let vv = buf(np)?;
        let half = cfg.head_dim() / 2;
        let cos = buf(t * half)?;
        let sin = buf(t * half)?;
        let tokens = buf(n)?;
        let targets = buf(n)?;
        let x = (0..=l).map(|_| buf(n * c)).collect::<Result<Vec<_>>>()?;
        let bufs = [n * c, n, n * 3 * c, seqs * nh * t * t, n * c, n * c, n * c, n * c, n, n * 2 * h, n * h, n * c, n, n * v, n];
        let mut made = bufs.iter().map(|&f| buf(f)).collect::<Result<Vec<_>>>()?.into_iter();
        let mut next = || made.next().unwrap();
        let (xn1, inv1, qkv, probs, att, tmp, xmid, xn2, inv2, ab, hg, xnf, invf, logits, losses) =
            (next(), next(), next(), next(), next(), next(), next(), next(), next(), next(), next(), next(), next(), next(), next());
        let dx = buf(n * c)?;
        let dn = buf(n * c)?;
        let dqkv = buf(n * 3 * c)?;
        let ds = buf(seqs * nh * t * t)?;
        let dab = buf(n * 2 * h)?;
        let dhg = buf(n * h)?;
        let partial = buf(2 * PARTIALS)?; // doubles
        // bf16 halves: two per f32-sized slot
        let (params16, s16a, s16b) = if bf16 {
            (buf(np.div_ceil(2))?.ptr, buf((n * v.max(3 * c).max(2 * h)).div_ceil(2))?.ptr, buf((n * (3 * c).max(2 * h)).div_ceil(2))?.ptr)
        } else {
            (0, 0, 0)
        };
        let g = GpuTrainer {
            be,
            cfg: cfg.clone(),
            layout,
            n_params: np,
            seqs,
            bf16,
            params16,
            s16a,
            s16b,
            adam_t: opt.t,
            segments: opt.segments().to_vec(),
            weight_decay: opt.weight_decay(),
            params,
            grads,
            m,
            v: vv,
            cos,
            sin,
            tokens,
            targets,
            x,
            xn1,
            inv1,
            qkv,
            probs,
            att,
            tmp,
            xmid,
            xn2,
            inv2,
            ab,
            hg,
            xnf,
            invf,
            logits,
            losses,
            dx,
            dn,
            dqkv,
            ds,
            dab,
            dhg,
            partial,
            allocated,
        };
        g.be.upload(g.params.ptr, f32_bytes(&model.params))?;
        g.be.upload(g.m.ptr, f32_bytes(&opt.m))?;
        g.be.upload(g.v.ptr, f32_bytes(&opt.v))?;
        let rope = crate::kernels::Rope::new(t, cfg.head_dim(), cfg.rope_base);
        let (cs, sn) = rope.tables();
        g.be.upload(g.cos.ptr, f32_bytes(cs))?;
        g.be.upload(g.sin.ptr, f32_bytes(sn))?;
        g.refresh_bf16()?;
        Ok(g)
    }

    pub fn backend(&self) -> &B {
        &self.be
    }

    fn p(&self, off: usize) -> DevPtr {
        self.params.at(off)
    }

    fn gr(&self, off: usize) -> DevPtr {
        self.grads.at(off)
    }

    /// The bf16 copy of the weights, after they change.
    fn refresh_bf16(&self) -> Result<()> {
        if self.bf16 {
            let n = self.n_params;
            self.launch(Kernel::ToBf16, self.grid_for(n), &[Arg::Ptr(self.params16), Arg::Ptr(self.params.ptr), Arg::I64(n as i64)])?;
        }
        Ok(())
    }

    /// A bf16 version of `count` floats at `p`: the weights' copy, or converted into `scratch`.
    fn half(&self, p: DevPtr, count: usize, scratch: DevPtr) -> Result<DevPtr> {
        let base = self.params.ptr;
        if p >= base && p < base + 4 * self.n_params as u64 {
            return Ok(self.params16 + (p - base) / 2);
        }
        self.launch(Kernel::ToBf16, self.grid_for(count), &[Arg::Ptr(scratch), Arg::Ptr(p), Arg::I64(count as i64)])?;
        Ok(scratch)
    }

    /// c (m x n) [+]= op(a) . op(b), dense row-major, as kernels::matmul
    /// (in bf16 with fp32 accumulation when enabled).
    #[allow(clippy::too_many_arguments)]
    fn mm(&self, c: DevPtr, a: DevPtr, b: DevPtr, m: usize, k: usize, n: usize, ta: bool, tb: bool, acc: bool) -> Result<()> {
        let (a, b) = if self.bf16 { (self.half(a, m * k, self.s16a)?, self.half(b, k * n, self.s16b)?) } else { (a, b) };
        let g = Gemm {
            m, n, k,
            alpha: 1.0,
            a, lda: if ta { m } else { k }, stride_a: 0, ta,
            b, ldb: if tb { k } else { n }, stride_b: 0, tb,
            beta: if acc { 1.0 } else { 0.0 },
            c, ldc: n, stride_c: 0,
            batch: 1,
        };
        if self.bf16 { self.be.gemm_bf16(&g) } else { self.be.gemm(&g) }
    }

    fn launch(&self, k: Kernel, grid: (u32, u32), args: &[Arg]) -> Result<()> {
        self.be.launch(k, grid, args)
    }

    /// Blocks for a loop over `n` items (grid-stride kernels).
    fn grid_for(&self, n: usize) -> (u32, u32) {
        ((n.div_ceil(self.be.block() as usize)).clamp(1, self.be.max_grid() as usize) as u32, 1)
    }

    fn rows(&self) -> usize {
        self.seqs * self.cfg.block_size
    }

    fn rmsnorm_fwd(&self, out: Buf, inv: Buf, x: Buf, g_off: usize) -> Result<()> {
        let c = self.cfg.d_model as i32;
        self.launch(Kernel::RmsnormFwd, (self.rows() as u32, 1), &[Arg::Ptr(out.ptr), Arg::Ptr(inv.ptr), Arg::Ptr(x.ptr), Arg::Ptr(self.p(g_off)), Arg::I32(c)])
    }

    /// dx += d(rmsnorm)/dx . dy, and the gain's gradient
    fn rmsnorm_bwd(&self, dy: Buf, x: Buf, inv: Buf, g_off: usize) -> Result<()> {
        let (n, c) = (self.rows(), self.cfg.d_model);
        self.launch(Kernel::RmsnormBwdDx, (n as u32, 1), &[
            Arg::Ptr(self.dx.ptr), Arg::Ptr(dy.ptr), Arg::Ptr(x.ptr), Arg::Ptr(inv.ptr), Arg::Ptr(self.p(g_off)), Arg::I32(c as i32),
        ])?;
        let chunk = 256usize;
        self.launch(Kernel::RmsnormBwdDg, (c.div_ceil(self.be.block() as usize) as u32, n.div_ceil(chunk) as u32), &[
            Arg::Ptr(self.gr(g_off)), Arg::Ptr(dy.ptr), Arg::Ptr(x.ptr), Arg::Ptr(inv.ptr), Arg::I64(n as i64), Arg::I32(c as i32), Arg::I64(chunk as i64),
        ])
    }

    fn rope(&self, qkv: Buf, sign: f32) -> Result<()> {
        let (n, c, t) = (self.rows(), self.cfg.d_model, self.cfg.block_size);
        let half = self.cfg.head_dim() / 2;
        self.launch(Kernel::Rope, self.grid_for(n * c), &[
            Arg::Ptr(qkv.ptr), Arg::Ptr(self.cos.ptr), Arg::Ptr(self.sin.ptr), Arg::I64(n as i64), Arg::I32(t as i32), Arg::I32(c as i32),
            Arg::I32(half as i32), Arg::F32(sign),
        ])
    }

    /// Per sequence, all heads at once: dst_h = alpha * lhs_h . rhs_h.
    #[allow(clippy::too_many_arguments)]
    fn heads(&self, m: usize, n: usize, k: usize, alpha: f32,
             a: DevPtr, lda: usize, sa: usize, ta: bool, b: DevPtr, ldb: usize, sb: usize, tb: bool,
             c: DevPtr, ldc: usize, sc: usize) -> Result<()> {
        self.be.gemm(&Gemm { m, n, k, alpha, a, lda, stride_a: sa, ta, b, ldb, stride_b: sb, tb, beta: 0.0, c, ldc, stride_c: sc, batch: self.cfg.n_head })
    }

    fn attention_fwd(&self) -> Result<()> {
        let (c, nh, t) = (self.cfg.d_model, self.cfg.n_head, self.cfg.block_size);
        let hd = c / nh;
        let scale = 1.0 / (hd as f32).sqrt();
        for s in 0..self.seqs {
            let q = self.qkv.at(s * t * 3 * c);
            let (k, _) = (self.qkv.at(s * t * 3 * c + c), ());
            let p = self.probs.at(s * nh * t * t);
            // scores = scale * Q . K^T
            self.heads(t, t, hd, scale, q, 3 * c, hd, false, k, 3 * c, hd, true, p, t, t * t)?;
        }
        self.launch(Kernel::CausalSoftmax, ((self.seqs * nh * t) as u32, 1), &[Arg::Ptr(self.probs.ptr), Arg::I32(t as i32)])?;
        for s in 0..self.seqs {
            let vv = self.qkv.at(s * t * 3 * c + 2 * c);
            let p = self.probs.at(s * nh * t * t);
            // out = P . V, each head into its columns
            self.heads(t, hd, t, 1.0, p, t, t * t, false, vv, 3 * c, hd, false, self.att.at(s * t * c), c, hd)?;
        }
        Ok(())
    }

    /// d[q|k|v] from d(out) (in dn), as kernels::attention_bwd.
    fn attention_bwd(&self) -> Result<()> {
        let (c, nh, t) = (self.cfg.d_model, self.cfg.n_head, self.cfg.block_size);
        let hd = c / nh;
        let scale = 1.0 / (hd as f32).sqrt();
        for s in 0..self.seqs {
            let base = s * t * 3 * c;
            let vv = self.qkv.at(base + 2 * c);
            let dout = self.dn.at(s * t * c);
            let (p, ds) = (self.probs.at(s * nh * t * t), self.ds.at(s * nh * t * t));
            // dP = dO . V^T ;  dV = P^T . dO
            self.heads(t, t, hd, 1.0, dout, c, hd, false, vv, 3 * c, hd, true, ds, t, t * t)?;
            self.heads(t, hd, t, 1.0, p, t, t * t, true, dout, c, hd, false, self.dqkv.at(base + 2 * c), 3 * c, hd)?;
        }
        self.launch(Kernel::SoftmaxBwd, ((self.seqs * nh * t) as u32, 1), &[Arg::Ptr(self.ds.ptr), Arg::Ptr(self.probs.ptr), Arg::I32(t as i32), Arg::F32(scale)])?;
        for s in 0..self.seqs {
            let base = s * t * 3 * c;
            let (q, k) = (self.qkv.at(base), self.qkv.at(base + c));
            let ds = self.ds.at(s * nh * t * t);
            // dQ = dS . K ;  dK = dS^T . Q
            self.heads(t, hd, t, 1.0, ds, t, t * t, false, k, 3 * c, hd, false, self.dqkv.at(base), 3 * c, hd)?;
            self.heads(t, hd, t, 1.0, ds, t, t * t, true, q, 3 * c, hd, false, self.dqkv.at(base + c), 3 * c, hd)?;
        }
        Ok(())
    }

    fn residual(&self, out: Buf, x: Buf, y: Buf, seed: Option<u64>) -> Result<()> {
        let n = self.rows() * self.cfg.d_model;
        let p = if seed.is_some() { self.cfg.dropout } else { 0.0 };
        self.launch(Kernel::Residual, self.grid_for(n), &[Arg::Ptr(out.ptr), Arg::Ptr(x.ptr), Arg::Ptr(y.ptr), Arg::I64(n as i64), Arg::F32(p), Arg::U64(seed.unwrap_or(0))])
    }

    /// One block's forward from x[l]; writes x[l+1] if `out`. Keeps its
    /// intermediate values in the (single) layer scratch buffers.
    fn layer_fwd(&self, l: usize, seed: Option<u64>, out: bool) -> Result<()> {
        let (c, h) = (self.cfg.d_model, self.cfg.mlp_hidden);
        let n = self.rows();
        let la = &self.layout.layers[l];
        let x_in = self.x[l];
        self.rmsnorm_fwd(self.xn1, self.inv1, x_in, la.rms1)?;
        self.mm(self.qkv.ptr, self.xn1.ptr, self.p(la.wqkv), n, c, 3 * c, false, false, false)?;
        self.rope(self.qkv, 1.0)?;
        self.attention_fwd()?;
        self.mm(self.tmp.ptr, self.att.ptr, self.p(la.wo), n, c, c, false, false, false)?;
        self.residual(self.xmid, x_in, self.tmp, seed.map(|s| s ^ (2 * l as u64 + 1)))?;
        self.rmsnorm_fwd(self.xn2, self.inv2, self.xmid, la.rms2)?;
        self.mm(self.ab.ptr, self.xn2.ptr, self.p(la.w13), n, c, 2 * h, false, false, false)?;
        self.launch(Kernel::SwigluFwd, self.grid_for(n * h), &[Arg::Ptr(self.hg.ptr), Arg::Ptr(self.ab.ptr), Arg::I64(n as i64), Arg::I32(h as i32)])?;
        self.mm(self.tmp.ptr, self.hg.ptr, self.p(la.w2), n, h, c, false, false, false)?;
        if out {
            self.residual(self.x[l + 1], self.xmid, self.tmp, seed.map(|s| s ^ (2 * l as u64 + 2)))?;
        }
        Ok(())
    }

    /// Forward over one micro-batch; returns the summed (not mean) loss.
    /// With `grad`, leaves d(loss * inv_n)/d(logits) in logits for `backward`.
    fn forward(&self, tokens: &[u32], targets: &[u32], seed: Option<u64>, grad: bool, inv_n: f32) -> Result<f64> {
        let (c, v, nl) = (self.cfg.d_model, self.cfg.vocab_size, self.cfg.n_layer);
        let n = self.rows();
        assert!(tokens.len() == n && targets.len() == n);
        self.be.upload(self.tokens.ptr, u32_bytes(tokens))?;
        self.be.upload(self.targets.ptr, u32_bytes(targets))?;
        self.launch(Kernel::EmbedFwd, self.grid_for(n * c), &[Arg::Ptr(self.x[0].ptr), Arg::Ptr(self.p(self.layout.wte)), Arg::Ptr(self.tokens.ptr), Arg::I64(n as i64), Arg::I32(c as i32)])?;
        for l in 0..nl {
            self.layer_fwd(l, seed, true)?;
        }
        self.rmsnorm_fwd(self.xnf, self.invf, self.x[nl], self.layout.rmsf)?;
        self.mm(self.logits.ptr, self.xnf.ptr, self.p(self.layout.wte), n, c, v, false, true, false)?;
        self.launch(Kernel::CrossEntropy, (n as u32, 1), &[
            Arg::Ptr(self.logits.ptr), Arg::Ptr(self.targets.ptr), Arg::Ptr(self.losses.ptr), Arg::I32(v as i32), Arg::F32(inv_n), Arg::I32(grad as i32),
        ])?;
        let mut losses = vec![0f32; n];
        self.be.download(self.losses.ptr, f32_bytes_mut(&mut losses))?;
        Ok(losses.iter().map(|&x| x as f64).sum())
    }

    /// Adds this micro-batch's gradients into `grads` (after `forward` with grad).
    fn backward(&self, seed: Option<u64>) -> Result<()> {
        let (c, h, v, nl) = (self.cfg.d_model, self.cfg.mlp_hidden, self.cfg.vocab_size, self.cfg.n_layer);
        let n = self.rows();
        let lay = &self.layout;
        let drop = seed.is_some() && self.cfg.dropout > 0.0;
        let mask = |out: Buf, d: Buf, s: u64| -> Result<()> {
            let len = n * c;
            self.launch(Kernel::MaskMul, self.grid_for(len), &[Arg::Ptr(out.ptr), Arg::Ptr(d.ptr), Arg::I64(len as i64), Arg::F32(self.cfg.dropout), Arg::U64(s)])
        };
        // logits = xnf . wte^T
        self.mm(self.dn.ptr, self.logits.ptr, self.p(lay.wte), n, v, c, false, false, false)?;
        self.mm(self.gr(lay.wte), self.logits.ptr, self.xnf.ptr, v, n, c, true, false, true)?;
        self.be.zero(self.dx.ptr, 4 * n * c)?;
        self.rmsnorm_bwd(self.dn, self.x[nl], self.invf, lay.rmsf)?;
        for l in (0..nl).rev() {
            let la = &lay.layers[l];
            self.layer_fwd(l, seed, false)?; // recompute this layer's values
            // MLP: x_out = xmid + mask2 * (hg . W2)
            let dmlp = if drop {
                mask(self.tmp, self.dx, seed.unwrap() ^ (2 * l as u64 + 2))?;
                self.tmp
            } else {
                self.dx
            };
            self.mm(self.gr(la.w2), self.hg.ptr, dmlp.ptr, h, n, c, true, false, true)?;
            self.mm(self.dhg.ptr, dmlp.ptr, self.p(la.w2), n, c, h, false, true, false)?;
            self.launch(Kernel::SwigluBwd, self.grid_for(n * h), &[Arg::Ptr(self.dab.ptr), Arg::Ptr(self.dhg.ptr), Arg::Ptr(self.ab.ptr), Arg::I64(n as i64), Arg::I32(h as i32)])?;
            self.mm(self.gr(la.w13), self.xn2.ptr, self.dab.ptr, c, n, 2 * h, true, false, true)?;
            self.mm(self.dn.ptr, self.dab.ptr, self.p(la.w13), n, 2 * h, c, false, true, false)?;
            self.rmsnorm_bwd(self.dn, self.xmid, self.inv2, la.rms2)?;
            // attention: xmid = x_in + mask1 * (att . Wo)
            let dproj = if drop {
                mask(self.tmp, self.dx, seed.unwrap() ^ (2 * l as u64 + 1))?;
                self.tmp
            } else {
                self.dx
            };
            self.mm(self.gr(la.wo), self.att.ptr, dproj.ptr, c, n, c, true, false, true)?;
            self.mm(self.dn.ptr, dproj.ptr, self.p(la.wo), n, c, c, false, true, false)?;
            self.attention_bwd()?;
            self.rope(self.dqkv, -1.0)?;
            self.mm(self.gr(la.wqkv), self.xn1.ptr, self.dqkv.ptr, c, n, 3 * c, true, false, true)?;
            self.mm(self.dn.ptr, self.dqkv.ptr, self.p(la.wqkv), n, 3 * c, c, false, true, false)?;
            self.rmsnorm_bwd(self.dn, self.x[l], self.inv1, la.rms1)?;
        }
        self.launch(Kernel::EmbedBwd, self.grid_for(n * c), &[Arg::Ptr(self.gr(lay.wte)), Arg::Ptr(self.dx.ptr), Arg::Ptr(self.tokens.ptr), Arg::I64(n as i64), Arg::I32(c as i32)])
    }

    fn micro_batches<'a>(&self, tokens: &'a [u32], targets: &'a [u32]) -> Result<impl Iterator<Item = (&'a [u32], &'a [u32])>> {
        let n = self.rows();
        if tokens.len() != targets.len() || tokens.is_empty() || !tokens.len().is_multiple_of(n) {
            bail!("a GPU batch must be a whole number of micro-batches ({n} tokens)");
        }
        Ok(tokens.chunks(n).zip(targets.chunks(n)))
    }

    /// Zero the gradients, then add the mean-loss gradients over all the
    /// micro-batches in `tokens`. Returns the mean loss.
    pub fn forward_backward(&self, tokens: &[u32], targets: &[u32], seed: Option<u64>) -> Result<f32> {
        self.be.zero(self.grads.ptr, 4 * self.n_params)?;
        let inv_n = 1.0 / tokens.len() as f32;
        let mut total = 0.0;
        for (i, (x, y)) in self.micro_batches(tokens, targets)?.enumerate() {
            let s = seed.map(|s| s ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407));
            total += self.forward(x, y, s, true, inv_n)?;
            self.backward(s)?;
        }
        Ok((total / tokens.len() as f64) as f32)
    }

    /// Mean loss without gradients.
    pub fn loss(&self, tokens: &[u32], targets: &[u32]) -> Result<f32> {
        let mut total = 0.0;
        for (x, y) in self.micro_batches(tokens, targets)? {
            total += self.forward(x, y, None, false, 0.0)?;
        }
        Ok((total / tokens.len() as f64) as f32)
    }

    /// Global gradient norm; scales gradients down to `max_norm` if above (as optim::clip_grad_norm).
    pub fn clip(&self, max_norm: f32) -> Result<f32> {
        let n = self.n_params;
        let blocks = (n.div_ceil(self.be.block() as usize)).clamp(1, PARTIALS);
        self.launch(Kernel::Sumsq, (blocks as u32, 1), &[Arg::Ptr(self.grads.ptr), Arg::I64(n as i64), Arg::Ptr(self.partial.ptr)])?;
        let mut raw = vec![0u8; 8 * blocks];
        self.be.download(self.partial.ptr, &mut raw)?;
        let norm = raw.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).sum::<f64>().sqrt() as f32;
        if norm > max_norm {
            let s = max_norm / (norm + 1e-6);
            self.launch(Kernel::Scale, self.grid_for(n), &[Arg::Ptr(self.grads.ptr), Arg::I64(n as i64), Arg::F32(s)])?;
        }
        Ok(norm)
    }

    /// One AdamW step (as optim::AdamW::step).
    pub fn adamw(&mut self, lr: f32) -> Result<()> {
        self.adam_t += 1;
        let (b1, b2, eps) = (crate::optim::BETA1, crate::optim::BETA2, crate::optim::EPS);
        let c1 = 1.0 - b1.powi(self.adam_t as i32);
        let c2 = 1.0 - b2.powi(self.adam_t as i32);
        for &(off, len, decay) in &self.segments {
            let wd = if decay { lr * self.weight_decay } else { 0.0 };
            self.launch(Kernel::Adamw, self.grid_for(len), &[
                Arg::Ptr(self.params.at(off)), Arg::Ptr(self.grads.at(off)), Arg::Ptr(self.m.at(off)), Arg::Ptr(self.v.at(off)), Arg::I64(len as i64),
                Arg::F32(lr), Arg::F32(b1), Arg::F32(b2), Arg::F32(c1), Arg::F32(c2), Arg::F32(eps), Arg::F32(wd),
            ])?;
        }
        self.refresh_bf16()
    }

    /// Copy weights and optimizer state back (for checkpoints).
    pub fn download_into(&self, model: &mut Model, opt: &mut AdamW) -> Result<()> {
        self.be.download(self.params.ptr, f32_bytes_mut(&mut model.params))?;
        self.be.download(self.m.ptr, f32_bytes_mut(&mut opt.m))?;
        self.be.download(self.v.ptr, f32_bytes_mut(&mut opt.v))?;
        opt.t = self.adam_t;
        Ok(())
    }

    pub fn download_grads(&self) -> Result<Vec<f32>> {
        let mut g = vec![0f32; self.n_params];
        self.be.download(self.grads.ptr, f32_bytes_mut(&mut g))?;
        Ok(g)
    }

    pub fn download_params(&self) -> Result<Vec<f32>> {
        let mut p = vec![0f32; self.n_params];
        self.be.download(self.params.ptr, f32_bytes_mut(&mut p))?;
        Ok(p)
    }
}

/// How far GPU results are from the CPU's.
#[derive(Debug, Clone)]
pub struct CheckReport {
    pub loss_cpu: f32,
    pub loss_gpu: f32,
    /// worst relative error of any weight matrix's gradient (L2 norm of the difference / L2 norm)
    pub worst_grad: (String, f64),
    /// relative error of the weights after one AdamW step from those gradients
    pub step_error: f64,
    /// the same gradients with bf16 matrix multiplies: worst relative error
    /// (bf16 keeps ~3 significant digits, so this is looser), if supported
    pub bf16_grad: Option<(String, f64)>,
    pub passed: bool,
}

/// Worst per-tensor relative error of `got` against `want` gradients.
fn worst_part(cfg: &ModelConfig, lay: &Layout, got: &[f32], want: &[f32]) -> (String, f64) {
    let mut parts: Vec<(String, usize, usize)> = vec![("embedding".into(), lay.wte, cfg.vocab_size * cfg.d_model)];
    for (i, la) in lay.layers.iter().enumerate() {
        let (c, h) = (cfg.d_model, cfg.mlp_hidden);
        for (name, off, len) in [("norm1", la.rms1, c), ("qkv", la.wqkv, 3 * c * c), ("out", la.wo, c * c), ("norm2", la.rms2, c), ("mlp_in", la.w13, 2 * c * h), ("mlp_out", la.w2, h * c)] {
            parts.push((format!("layer {i} {name}"), off, len));
        }
    }
    parts.push(("final norm".into(), lay.rmsf, cfg.d_model));
    parts
        .iter()
        .map(|(name, off, len)| (name.clone(), rel_err(&got[*off..off + len], &want[*off..off + len])))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// Relative L2 error of `got` against `want`.
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let (mut d, mut w) = (0f64, 0f64);
    for (&g, &x) in got.iter().zip(want) {
        d += ((g - x) as f64).powi(2);
        w += (x as f64).powi(2);
    }
    if w == 0.0 {
        d.sqrt()
    } else {
        (d / w).sqrt()
    }
}

/// Train one step of a small random model on the device and on the CPU (in
/// strict fp32), and compare the loss, every gradient and the updated weights.
pub fn self_check<B: Backend>(be: B) -> Result<CheckReport> {
    be.set_tf32(false)?;
    let cfg = ModelConfig { vocab_size: 300, block_size: 24, n_layer: 2, n_head: 4, d_model: 64, mlp_hidden: 96, dropout: 0.0, rope_base: 10000.0 };
    let (seqs, micro) = (2, 2);
    let mut model = Model::new(cfg.clone(), 7);
    // bigger than initial weights, so gradients are well above rounding noise
    model.params.iter_mut().for_each(|p| *p *= 4.0);
    let n = seqs * micro * cfg.block_size;
    let mut rng = crate::rng::Rng::new(11);
    let tokens: Vec<u32> = (0..n).map(|_| rng.below(cfg.vocab_size) as u32).collect();
    let targets: Vec<u32> = (0..n).map(|_| rng.below(cfg.vocab_size) as u32).collect();

    // CPU: one batch of all sequences
    let mut acts = crate::model::Acts::new(&cfg, seqs * micro, cfg.block_size);
    let mut grads = vec![0f32; model.num_params()];
    let loss_cpu = model.forward_backward(&mut acts, &mut grads, &tokens, &targets, None);
    let mut opt = AdamW::new(model.num_params(), &model.layout.matrices, 0.1);

    // bf16, if supported: on normally initialised weights (as training
    // uses) - the enlarged ones above make attention nearly one-hot, which
    // magnifies bf16's rounding far beyond what training sees
    let bf16_grad = if be.supports_bf16() {
        let plain = Model::new(cfg.clone(), 7);
        let mut want = vec![0f32; plain.num_params()];
        plain.forward_backward(&mut acts, &mut want, &tokens, &targets, None);
        let g16 = GpuTrainer::new(&be, &plain, &opt, seqs, true)?;
        g16.forward_backward(&tokens, &targets, None)?;
        Some(worst_part(&cfg, &plain.layout, &g16.download_grads()?, &want))
    } else {
        None
    };
    let mut g = GpuTrainer::new(be, &model, &opt, seqs, false)?;
    let loss_gpu = g.forward_backward(&tokens, &targets, None)?;
    let worst_grad = worst_part(&cfg, &model.layout, &g.download_grads()?, &grads);

    crate::optim::clip_grad_norm(&mut grads, 1.0);
    g.clip(1.0)?;
    opt.step(&mut model.params, &grads, 1e-3);
    g.adamw(1e-3)?;
    let step_error = rel_err(&g.download_params()?, &model.params);
    g.backend().set_tf32(true)?;
    let passed = ((loss_gpu - loss_cpu) / loss_cpu).abs() < 1e-4
        && worst_grad.1 < 1e-3
        && step_error < 1e-5
        && bf16_grad.as_ref().is_none_or(|b| b.1 < 5e-2);
    Ok(CheckReport { loss_cpu, loss_gpu, worst_grad, step_error, bf16_grad, passed })
}

/// Sustained matrix-multiply speed (TF32, or bf16 inputs), in FLOP/s.
pub fn gemm_flops<B: Backend>(be: &B, bf16: bool) -> Result<f64> {
    be.set_tf32(true)?;
    let n = 4096;
    let bytes = 4 * n * n;
    let (a, b, c) = (be.alloc(bytes)?, be.alloc(bytes)?, be.alloc(bytes)?);
    let result = (|| {
        for p in [a, b] {
            be.zero(p, bytes)?;
        }
        let g = Gemm { m: n, n, k: n, alpha: 1.0, a, lda: n, stride_a: 0, ta: false, b, ldb: n, stride_b: 0, tb: false, beta: 0.0, c, ldc: n, stride_c: 0, batch: 1 };
        let run = |g: &Gemm| if bf16 { be.gemm_bf16(g) } else { be.gemm(g) };
        run(&g)?;
        be.sync()?;
        let start = std::time::Instant::now();
        let reps = 20;
        for _ in 0..reps {
            run(&g)?;
        }
        be.sync()?;
        Ok(reps as f64 * 2.0 * (n * n * n) as f64 / start.elapsed().as_secs_f64())
    })();
    for p in [a, b, c] {
        be.free(p);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_list_matches_the_cuda_source() {
        let line = KERNELS_CU.lines().skip_while(|l| !l.contains("#define COUNCIL_KERNELS(X)")).take(4).collect::<String>();
        let names: Vec<&str> = line.split("X(").skip(1).map(|s| s.split(')').next().unwrap()).collect();
        assert_eq!(names, Kernel::ALL.iter().map(|k| k.name()).collect::<Vec<_>>());
        for k in Kernel::ALL {
            assert!(KERNELS_CU.contains(&format!("__global__ void {}(", k.name())), "{} has no kernel", k.name());
        }
    }

    /// cuBLAS semantics (column-major), the reference for the mapping.
    fn cublas_ref(g: &CublasGemm, mem: &mut [f32]) {
        for bi in 0..g.batch as usize {
            let (a0, b0, c0) = (g.a as usize + bi * g.stride_a as usize, g.b as usize + bi * g.stride_b as usize, g.c as usize + bi * g.stride_c as usize);
            for i in 0..g.m as usize {
                for j in 0..g.n as usize {
                    let mut acc = 0f32;
                    for p in 0..g.k as usize {
                        let a = if g.trans_a { mem[a0 + p + i * g.lda as usize] } else { mem[a0 + i + p * g.lda as usize] };
                        let b = if g.trans_b { mem[b0 + j + p * g.ldb as usize] } else { mem[b0 + p + j * g.ldb as usize] };
                        acc += a * b;
                    }
                    let c = &mut mem[c0 + i + j * g.ldc as usize];
                    *c = g.alpha * acc + if g.beta == 0.0 { 0.0 } else { g.beta * *c };
                }
            }
        }
    }

    #[test]
    fn row_major_gemm_maps_onto_cublas() {
        let mut rng = crate::rng::Rng::new(5);
        let (m, k, n) = (5, 7, 3);
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.normal()).collect();
            let mut want = vec![0f32; m * n];
            crate::kernels::matmul(&mut want, &a, &b, m, k, n, ta, tb, false);
            // "device memory" = one array; pointers are indices
            let mut mem: Vec<f32> = a.iter().chain(&b).cloned().chain(std::iter::repeat(0.0).take(m * n)).collect();
            let g = Gemm { m, n, k, alpha: 1.0, a: 0, lda: if ta { m } else { k }, stride_a: 0, ta, b: (m * k) as u64, ldb: if tb { k } else { n },
                           stride_b: 0, tb, beta: 0.0, c: (m * k + k * n) as u64, ldc: n, stride_c: 0, batch: 1 };
            cublas_ref(&g.to_cublas(), &mut mem);
            let got = &mem[m * k + k * n..];
            assert!(got.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-4), "ta={ta} tb={tb}");
        }
    }

    #[test]
    fn micro_batches_fit_the_memory() {
        let cfg = ModelConfig { vocab_size: 32768, block_size: 1024, n_layer: 24, n_head: 16, d_model: 1024, mlp_hidden: 2752, dropout: 0.1, rope_base: 1e4 };
        let gb = |x: f64| (x * (1u64 << 30) as f64) as u64;
        let s = micro_batch_for(&cfg, gb(15.0), 64, false).unwrap();
        assert!((8..=24).contains(&s), "{s} sequences");
        assert!(training_bytes(&cfg, s, false) + OVERHEAD_BYTES <= gb(15.0));
        assert!(micro_batch_for(&cfg, gb(15.0), 64, true).unwrap() < s); // bf16 copies take room
        assert!(micro_batch_for(&cfg, gb(4.0), 64, false).is_none());
    }
}
