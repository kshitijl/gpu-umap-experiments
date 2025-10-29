#!/usr/bin/env python3
"""
CuPy Example with Custom CUDA Kernel
Demonstrates writing raw CUDA C++ code as a string
"""

import cupy as cp
import numpy as np
from tqdm import tqdm

from utils import read_coo_array

# initialize cuRAND states
init_curand = cp.RawKernel(
    r"""
#include <curand_kernel.h>

extern "C" __global__
void init_curand_states(curandState *states, unsigned long long seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Each thread gets same seed, different sequence number, no offset
    curand_init(seed + idx, 0, 0, &states[idx]);
}
""",
    "init_curand_states",
)

positive_sample_pass = cp.RawKernel(
    r"""
#include <curand_kernel.h>

__forceinline__ float clamp_grad(float value) {
    return fminf(fmaxf(value, -4.0f), 4.0f);
}

extern "C" __global__
void positive_sample_pass(
    const float* pos_x, const float* pos_y,
    float* grad_x, float* grad_y,
    int n_nodes,
    const int* sources, const int* targets, const float* weights,
    int n_edges,
    float a,
    float b,
    float gamma,
    float alpha,
    curandState *states
) {
    extern __shared__ int s[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_edges) {
        return;
    }

    curandState local_state = states[idx];

    int src = sources[idx];
    int dst = targets[idx];
    float src_x = pos_x[src];
    float src_y = pos_y[src];
    float dst_x = pos_x[dst];
    float dst_y = pos_y[dst];

    float dx = src_x - dst_x;
    float dy = src_y - dst_y;
    float rdist = dx * dx + dy * dy;

    float grad_coeff;
    grad_coeff = -2.0f * a * b * weights[idx] * powf(rdist, b - 1.0f);
    grad_coeff /= a * powf(rdist, b) + 1.0f;

    atomicAdd(&grad_x[src], alpha * clamp_grad(dx * grad_coeff));
    atomicAdd(&grad_y[src], alpha * clamp_grad(dy * grad_coeff));

    int negative_sample_count = 10;
    for (int i = 0; i < negative_sample_count; i++) {
        unsigned int dst = curand(&local_state) % n_nodes; // ??
        // printf("dst: %d\n", dst);
        float dst_x = pos_x[dst];
        float dst_y = pos_y[dst];
        float dx = src_x - dst_x;
        float dy = src_y - dst_y;
        float rdist = dx * dx + dy * dy;
        float grad_coeff = 2.0f * gamma * b;
        grad_coeff /= (0.001f + rdist) * (a * powf(rdist, b) + 1.0f);

        atomicAdd(&grad_x[src], alpha * clamp_grad(dx * grad_coeff));
        atomicAdd(&grad_y[src], alpha * clamp_grad(dy * grad_coeff));
    }
}
""",
    "positive_sample_pass",
)

# Helper to get curandState size
_size_kernel = cp.RawKernel(
    r"""
#include <curand_kernel.h>
extern "C" __global__
void get_curand_size(unsigned long long *size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *size = sizeof(curandState);
    }
}
""",
    "get_curand_size",
)


def allocate_curand_states(num_states):
    """Allocate memory for curandState array"""
    size_result = cp.zeros(1, dtype=cp.uint64)
    _size_kernel((1,), (1,), (size_result,))
    cp.cuda.Stream.null.synchronize()
    state_size = int(size_result[0])
    return cp.empty(num_states * state_size, dtype=cp.uint8), state_size


def layout(sources, targets, weights, initial_pos, n_epochs=500):
    assert sources.shape == targets.shape
    n_edges = len(sources)
    n_nodes, dim = initial_pos.shape
    assert dim == 2

    pos_x = cp.asarray(initial_pos[:, 0], dtype=cp.float32)
    pos_y = cp.asarray(initial_pos[:, 1], dtype=cp.float32)

    print(f"Sampling {n_edges:,} edges using custom kernel...")

    # Configure grid and block dimensions
    threads_per_block = 32
    blocks_per_grid = (n_edges + threads_per_block - 1) // threads_per_block
    total_threads = blocks_per_grid * threads_per_block

    print(f"Grid: {blocks_per_grid} blocks × {threads_per_block} threads")

    grad_x = cp.zeros((n_nodes,), dtype=cp.float32)
    grad_y = cp.zeros((n_nodes,), dtype=cp.float32)

    a = cp.float32(1.5769434605754993)
    b = cp.float32(0.8950608781680347)
    gamma = cp.float32(1.0)
    alpha = cp.float32(1.0)

    curand_states, state_size = allocate_curand_states(total_threads)
    print(curand_states, state_size)

    init_curand((1,), (1,), (curand_states, 0))

    print("sources", sources)
    print("targets", targets)

    initial_alpha = alpha

    for n in tqdm(range(n_epochs)):
        cp.cuda.runtime.deviceSynchronize()

        # Launch the kernel
        positive_sample_pass(
            (blocks_per_grid,),  # grid dimensions
            (threads_per_block,),  # block dimensions
            (
                pos_x,
                pos_y,
                grad_x,
                grad_y,
                n_nodes,
                sources,
                targets,
                weights,
                n_edges,
                a,
                b,
                gamma,
                alpha,
                curand_states,
            ),
            shared_mem=64,
        )

        pos_x += grad_x
        pos_y += grad_y
        grad_x.fill(0)
        grad_y.fill(0)

        p = float(n) / float(n_epochs)
        alpha = initial_alpha * (1.0 - p)

    return np.column_stack((cp.asnumpy(pos_x), cp.asnumpy(pos_y)))


def main():
    print("CuPy Custom Kernel Example")
    print(f"GPU ID: {cp.cuda.Device().id}")

    fss = read_coo_array("fss-1e5.arrow")
    assert fss.shape[0] == fss.shape[1]

    sources = cp.asarray(fss.row, dtype=cp.int32)
    targets = cp.asarray(fss.col, dtype=cp.int32)
    weights = cp.asarray(fss.data, dtype=cp.float32)
    assert sources.shape == targets.shape

    # sources = cp.asarray(np.load("sources-1e3.npy"), dtype=cp.int32)
    # targets = cp.asarray(np.load("targets-1e3.npy"), dtype=cp.int32)
    # sources = cp.asarray(np.load("sources-1e5.npy"), dtype=cp.int32)
    # targets = cp.asarray(np.load("targets-1e5.npy"), dtype=cp.int32)

    n_nodes = fss.shape[0]
    n_edges = len(weights)

    print(f"n_nodes: {n_nodes}, n_edges: {n_edges}")
    print("sources", sources)
    print("targets", targets)

    assert np.all(sources < n_nodes)
    assert np.all(targets < n_nodes)

    rand = np.random.RandomState(42)
    initial_pos = np.column_stack((rand.random(n_nodes), rand.random(n_nodes)))
    initial_pos *= 10
    final_pos = layout(sources, targets, weights, initial_pos)

    print("final pos", final_pos[:10])


if __name__ == "__main__":
    main()
