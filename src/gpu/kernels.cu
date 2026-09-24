// The transformer's non-matrix-multiply math on an NVIDIA GPU, written by
// hand. Compiled at run time by NVRTC (NVIDIA's CUDA compiler library), or
// as plain C++ by gpu/emulate.cpp to test it on a CPU. Matrix multiplies go
// to cuBLAS instead.
//
// Every kernel mirrors a function in src/kernels.rs, so the GPU computes the
// same thing as the CPU. Rows use one block of 256 threads with shared-memory
// reductions (no warp shuffles, so the CPU emulator can run it exactly).

// Threads per block, a power of two (the CPU emulator uses fewer).
#ifndef BLOCK
#define BLOCK 256
#endif
#define RMS_EPS 1e-5f

// Every kernel, for the emulator's launch table (keep in sync with Kernel in gpu/mod.rs).
#define COUNCIL_KERNELS(X) \
    X(embed_fwd) X(embed_bwd) X(rmsnorm_fwd) X(rmsnorm_bwd_dx) X(rmsnorm_bwd_dg) X(rope) \
    X(causal_softmax) X(softmax_bwd) X(swiglu_fwd) X(swiglu_bwd) X(residual) X(mask_mul) \
    X(cross_entropy) X(sumsq) X(scale) X(adamw) X(to_bf16)

typedef unsigned long long u64;

__device__ float block_sum(float v, float* red) {
    int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}

__device__ float block_max(float v, float* red) {
    int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}

__device__ double block_sum_d(double v, double* red) {
    int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    double r = red[0];
    __syncthreads();
    return r;
}

// First element and step for a loop spreading `n` items over the whole grid.
#define GRID_LOOP(i, n) \
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

// Dropout keep-mask from a hash of (seed, index): 0 with probability p, else 1/(1-p).
__device__ float drop_mask(u64 seed, long long i, float p) {
    u64 z = seed ^ ((u64)i * 0x9E3779B97F4A7C15ULL);
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    float u = (float)(z >> 40) * (1.0f / 16777216.0f);
    return u < p ? 0.0f : 1.0f / (1.0f - p);
}

__device__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// out[row] = wte[token[row]]
extern "C" __global__ void embed_fwd(float* out, const float* wte, const unsigned* tokens, long long n, int c) {
    GRID_LOOP(i, n * c) {
        long long row = i / c;
        out[i] = wte[(long long)tokens[row] * c + i % c];
    }
}

// dwte[token[row]] += dx[row]
extern "C" __global__ void embed_bwd(float* dwte, const float* dx, const unsigned* tokens, long long n, int c) {
    GRID_LOOP(i, n * c) {
        long long row = i / c;
        atomicAdd(&dwte[(long long)tokens[row] * c + i % c], dx[i]);
    }
}

// One block per row: out = x / rms(x) * g, inv = 1 / rms(x)
extern "C" __global__ void rmsnorm_fwd(float* out, float* inv, const float* x, const float* g, int c) {
    __shared__ float red[BLOCK];
    long long row = blockIdx.x;
    const float* xr = x + row * c;
    float ss = 0.0f;
    for (int j = threadIdx.x; j < c; j += BLOCK) ss += xr[j] * xr[j];
    ss = block_sum(ss, red);
    float r = 1.0f / sqrtf(ss / c + RMS_EPS);
    if (threadIdx.x == 0) inv[row] = r;
    for (int j = threadIdx.x; j < c; j += BLOCK) out[row * c + j] = xr[j] * r * g[j];
}

// One block per row: dx += r * (dy * g - x * r^2 * mean(dy * g * x))
extern "C" __global__ void rmsnorm_bwd_dx(float* dx, const float* dy, const float* x, const float* inv, const float* g, int c) {
    __shared__ float red[BLOCK];
    long long row = blockIdx.x;
    const float* xr = x + row * c;
    const float* dyr = dy + row * c;
    float r = inv[row];
    float d = 0.0f;
    for (int j = threadIdx.x; j < c; j += BLOCK) d += dyr[j] * g[j] * xr[j];
    d = block_sum(d, red) / c;
    for (int j = threadIdx.x; j < c; j += BLOCK) dx[row * c + j] += r * (dyr[j] * g[j] - xr[j] * r * r * d);
}

// dg[j] += sum over rows of dy * x * r. Grid: (columns / BLOCK, row chunks).
extern "C" __global__ void rmsnorm_bwd_dg(float* dg, const float* dy, const float* x, const float* inv, long long rows, int c, long long rows_per_chunk) {
    int j = blockIdx.x * BLOCK + threadIdx.x;
    if (j >= c) return;
    long long start = (long long)blockIdx.y * rows_per_chunk;
    long long end = start + rows_per_chunk < rows ? start + rows_per_chunk : rows;
    float acc = 0.0f;
    for (long long row = start; row < end; row++) acc += dy[row * c + j] * x[row * c + j] * inv[row];
    atomicAdd(&dg[j], acc);
}

// Rotate each (even, odd) pair of the q and k blocks of qkv rows by its
// position's angle; sign = -1 undoes the rotation (for gradients).
extern "C" __global__ void rope(float* qkv, const float* cos_t, const float* sin_t, long long rows, int t, int c, int half, float sign) {
    int pairs = c / 2;  // per block (q or k)
    GRID_LOOP(i, rows * 2 * pairs) {
        long long row = i / (2 * pairs);
        int within = (int)(i % (2 * pairs));
        int block = within / pairs, pair = within % pairs;
        int k = pair % half;
        int pos = (int)(row % t);
        float cs = cos_t[pos * half + k], sn = sign * sin_t[pos * half + k];
        float* v = qkv + row * 3 * c + block * c + 2 * pair;
        float x0 = v[0], x1 = v[1];
        v[0] = x0 * cs - x1 * sn;
        v[1] = x0 * sn + x1 * cs;
    }
}

// One block per row of t x t score matrices: softmax over columns j <= i
// (i = row within its matrix), zero above the diagonal.
extern "C" __global__ void causal_softmax(float* s, int t) {
    __shared__ float red[BLOCK];
    long long row = blockIdx.x;
    int n = (int)(row % t) + 1;
    float* r = s + row * t;
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < n; j += BLOCK) m = fmaxf(m, r[j]);
    m = block_max(m, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += BLOCK) {
        float e = expf(r[j] - m);
        r[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < t; j += BLOCK) r[j] = j < n ? r[j] * inv : 0.0f;
}

// One block per row: ds (holding dP) = P * (dP - sum_j P * dP) * scale, causal.
extern "C" __global__ void softmax_bwd(float* ds, const float* p, int t, float scale) {
    __shared__ float red[BLOCK];
    long long row = blockIdx.x;
    int n = (int)(row % t) + 1;
    float* d = ds + row * t;
    const float* pr = p + row * t;
    float s = 0.0f;
    for (int j = threadIdx.x; j < n; j += BLOCK) s += pr[j] * d[j];
    s = block_sum(s, red);
    for (int j = threadIdx.x; j < t; j += BLOCK) d[j] = j < n ? pr[j] * (d[j] - s) * scale : 0.0f;
}

// h = silu(a) * b, rows of ab = [a | b]
extern "C" __global__ void swiglu_fwd(float* h, const float* ab, long long n, int hidden) {
    GRID_LOOP(i, n * hidden) {
        long long row = i / hidden;
        int j = (int)(i % hidden);
        float a = ab[row * 2 * hidden + j], b = ab[row * 2 * hidden + hidden + j];
        h[i] = a * sigmoidf(a) * b;
    }
}

extern "C" __global__ void swiglu_bwd(float* dab, const float* dh, const float* ab, long long n, int hidden) {
    GRID_LOOP(i, n * hidden) {
        long long row = i / hidden;
        int j = (int)(i % hidden);
        long long ia = row * 2 * hidden + j, ib = ia + hidden;
        float a = ab[ia], b = ab[ib], s = sigmoidf(a);
        dab[ia] = dh[i] * b * s * (1.0f + a * (1.0f - s));
        dab[ib] = dh[i] * a * s;
    }
}

// out = x + y * dropout_mask (p = 0: no dropout)
extern "C" __global__ void residual(float* out, const float* x, const float* y, long long n, float p, u64 seed) {
    GRID_LOOP(i, n) out[i] = x[i] + (p > 0.0f ? y[i] * drop_mask(seed, i, p) : y[i]);
}

// out = d * dropout_mask (the same mask `residual` used)
extern "C" __global__ void mask_mul(float* out, const float* d, long long n, float p, u64 seed) {
    GRID_LOOP(i, n) out[i] = d[i] * drop_mask(seed, i, p);
}

// One block per row of logits: loss[row] = logsumexp - logit[target]; with
// grad != 0, logits become d(mean loss)/d(logits), inv_n = 1 / tokens averaged over.
extern "C" __global__ void cross_entropy(float* logits, const unsigned* targets, float* loss, int v, float inv_n, int grad) {
    __shared__ float red[BLOCK];
    long long row = blockIdx.x;
    float* r = logits + row * v;
    unsigned tgt = targets[row];
    float lt = r[tgt];
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < v; j += BLOCK) m = fmaxf(m, r[j]);
    m = block_max(m, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < v; j += BLOCK) sum += expf(r[j] - m);
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) loss[row] = logf(sum) + m - lt;
    if (grad) {
        float inv = 1.0f / sum;
        for (int j = threadIdx.x; j < v; j += BLOCK) r[j] = expf(r[j] - m) * inv * inv_n;
        __syncthreads();
        if (threadIdx.x == 0) r[tgt] -= inv_n;
    }
}

// partial[block] = sum of squares of this block's share of g (in double)
extern "C" __global__ void sumsq(const float* g, long long n, double* partial) {
    __shared__ double red[BLOCK];
    double acc = 0.0;
    GRID_LOOP(i, n) acc += (double)g[i] * (double)g[i];
    acc = block_sum_d(acc, red);
    if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

extern "C" __global__ void scale(float* g, long long n, float s) {
    GRID_LOOP(i, n) g[i] *= s;
}

// f32 -> bfloat16 (the top 16 bits), rounding to nearest even, for the
// bf16 matrix multiplies (tensor cores, fp32 accumulation).
extern "C" __global__ void to_bf16(unsigned short* dst, const float* src, long long n) {
    GRID_LOOP(i, n) {
        unsigned int u = __float_as_uint(src[i]);
        if ((u & 0x7fffffffu) > 0x7f800000u) {
            dst[i] = (unsigned short)((u >> 16) | 0x40u);  // NaN stays NaN
        } else {
            u += 0x7fffu + ((u >> 16) & 1u);
            dst[i] = (unsigned short)(u >> 16);
        }
    }
}

// AdamW (as optim.rs): wd = lr * weight_decay for matrices, 0 for gains;
// c1, c2 = bias corrections 1 - beta^t.
extern "C" __global__ void adamw(float* p, const float* g, float* m, float* v, long long n, float lr, float b1, float b2, float c1, float c2, float eps, float wd) {
    GRID_LOOP(i, n) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= wd * p[i] + lr * (mi / c1) / (sqrtf(vi / c2) + eps);
    }
}
