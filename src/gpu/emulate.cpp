// Runs src/gpu/kernels.cu on the CPU, for tests on machines without an
// NVIDIA GPU (cargo test --features gpu-emulator). Each launch runs the
// grid's blocks one after another; a block's threads run as real threads
// meeting at a barrier for __syncthreads, and `__shared__` arrays become
// statics shared by them - the same semantics as on the GPU.

#include <math.h>
#include <barrier>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

struct dim3 {
    unsigned x = 1, y = 1, z = 1;
};
static thread_local dim3 threadIdx, blockIdx;
static dim3 blockDim, gridDim;
static std::barrier<>* emu_barrier = nullptr;
static std::mutex emu_atomic;

#define __global__
#define __device__
#define __shared__ static

static inline void __syncthreads() { emu_barrier->arrive_and_wait(); }
static inline unsigned int __float_as_uint(float f) {
    unsigned int u;
    std::memcpy(&u, &f, 4);
    return u;
}
static inline float atomicAdd(float* a, float v) {
    std::lock_guard<std::mutex> g(emu_atomic);
    float old = *a;
    *a = old + v;
    return old;
}

// Few threads per block: the kernels are written for any power of two, and
// real threads meeting at barriers are slow on a CPU.
#define BLOCK 8
#include "kernels.cu"

template <typename... A, std::size_t... I>
static void call_with(void (*k)(A...), void** p, std::index_sequence<I...>) {
    k(*static_cast<A*>(p[I])...);
}

template <typename... A>
static void call(void (*k)(A...), void** p) {
    call_with(k, p, std::index_sequence_for<A...>{});
}

typedef void (*Runner)(void**);

static Runner find(const char* name) {
#define ENTRY(k) \
    if (std::strcmp(name, #k) == 0) return [](void** p) { call(k, p); };
    COUNCIL_KERNELS(ENTRY)
#undef ENTRY
    return nullptr;
}

static std::mutex emu_launch_lock;

// Returns 0 on success, 1 for an unknown kernel name. One launch at a time:
// like a GPU stream, and because the emulated shared memory is global.
extern "C" int aegist_emu_launch(const char* name, unsigned gx, unsigned gy, unsigned bx, void** params) {
    Runner run = find(name);
    if (!run) return 1;
    std::lock_guard<std::mutex> one_at_a_time(emu_launch_lock);
    gridDim = dim3{gx, gy, 1};
    blockDim = dim3{bx, 1, 1};
    std::barrier<> bar(bx);
    emu_barrier = &bar;
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < bx; t++) {
        threads.emplace_back([=, &bar] {
            threadIdx = dim3{t, 0, 0};
            for (unsigned y = 0; y < gy; y++) {
                for (unsigned x = 0; x < gx; x++) {
                    blockIdx = dim3{x, y, 0};
                    run(params);
                    // the next block reuses the shared arrays: wait for everyone
                    bar.arrive_and_wait();
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    emu_barrier = nullptr;
    return 0;
}
