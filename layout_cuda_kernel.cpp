#include <curand_kernel.h>


extern "C" __global__
void init_curand_states(curandState *states, unsigned long long seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Each thread gets same seed, different sequence number, no offset
    curand_init(seed + idx, 0, 0, &states[idx]);
}

extern "C" __global__
void get_sizeof_curand_state(unsigned long long *size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *size = sizeof(curandState);
    }
}

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
    
    // printf("idx: %d\n", idx);

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

    float ux = alpha * clamp_grad(dx * grad_coeff);
    float uy = alpha * clamp_grad(dy * grad_coeff);
    // printf("ux: %f, uy: %f\n", ux, uy);
    atomicAdd(&grad_x[src], ux);
    atomicAdd(&grad_y[src], uy);
    // atomicAdd(&grad_x[dst], -ux);
    // atomicAdd(&grad_y[dst], -uy);

    int negative_sample_count = 3;
    for (int i = 0; i < negative_sample_count; i++) {
        unsigned int dst = curand(&local_state) % n_nodes; // ??
        // printf("[%d/%d]: dst %d\n", idx, i, dst);
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