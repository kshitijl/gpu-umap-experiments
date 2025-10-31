#include <curanddx.hpp>

// typedef unsigned long uint64_t;

// constexpr unsigned int subsequences = 65536;

// using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
//                      curanddx::PhiloxRounds<10>() +
//                      curanddx::SM<1200>() +
//                      curanddx::Thread());

using RNG = decltype(curanddx::Generator<curanddx::pcg>() +
                     curanddx::SM<1200>() +
                     curanddx::Thread());


__device__ __forceinline__ float clamp_grad(float value) {
    return fminf(fmaxf(value, -4.0f), 4.0f);
}

__device__ __forceinline__ unsigned int get_element(const uint4& v, unsigned int idx) {
    switch(idx) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        case 3: return v.w;
        default: return 0;
    }
}

__device__ __host__ inline uint64_t splitmix64(uint64_t seed) {
    uint64_t z = seed * 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

extern "C" __global__
void positive_sample_pass(
    const float* pos_x, const float* pos_y,
    float* grad_x, float* grad_y,
    unsigned int n_nodes,
    const int* sources, const int* targets, const float* weights,
    unsigned int n_edges,
    float a,
    float b,
    float gamma,
    float alpha,
    unsigned int epoch,
    unsigned int n_epochs
) {
    extern __shared__ int s[];

    const unsigned long long        seed   = 1234ULL;
    const typename RNG::offset_type offset = 1ULL;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_edges) {
        return;
    }

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

    // for (int i = 0; i < negative_sample_count; i++) {
    //     uint64_t cycle_size = n_edges * negative_sample_count;
    //     uint64_t seed = cycle_size * epoch + idx * negative_sample_count + i;

    //     auto dst = splitmix64(seed) % n_nodes;
    //     float dst_x = pos_x[dst];
    //     float dst_y = pos_y[dst];
    //     float dx = src_x - dst_x;
    //     float dy = src_y - dst_y;
    //     float rdist = dx * dx + dy * dy;
    //     float grad_coeff = 2.0f * gamma * b;
    //     grad_coeff /= (0.001f + rdist) * (a * powf(rdist, b) + 1.0f);

    //     atomicAdd(&grad_x[src], alpha * clamp_grad(dx * grad_coeff));
    //     atomicAdd(&grad_y[src], alpha * clamp_grad(dy * grad_coeff));
    // }

    // 1650 it/s
    // seed, subsequence, offset
    unsigned int subsequences = n_edges;
    RNG rng(seed, ((offset + idx) % subsequences), ((offset + idx) / subsequences));
    curanddx::uniform_bits<unsigned int> prng;
    for (int i = 0; i < negative_sample_count; i++) {
        unsigned int dst = prng.generate(rng) % n_nodes;
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

    // // 1601 it/s
    // // seed, subsequence, offset
    // RNG rng(seed, ((offset + idx) % subsequences), ((offset + idx) / subsequences));
    // curanddx::uniform_bits<unsigned int> prng;
    // int prng_generations = 1 + (negative_sample_count / 4);
    // for (unsigned int p = 0; p < prng_generations; p++) {
        
    //     uint4 dst_buffer = prng.generate4(rng);
        
    //     for (unsigned int q = 0; q < 4; q++) {
    //         unsigned int i = p * 4 + q;
    //         if (i >= negative_sample_count) {
    //             break;
    //         }

    //         unsigned int dst = get_element(dst_buffer, q) % n_nodes;
    //         float dst_x = pos_x[dst];
    //         float dst_y = pos_y[dst];
    //         float dx = src_x - dst_x;
    //         float dy = src_y - dst_y;
    //         float rdist = dx * dx + dy * dy;
    //         float grad_coeff = 2.0f * gamma * b;
    //         grad_coeff /= (0.001f + rdist) * (a * powf(rdist, b) + 1.0f);
    
    //         atomicAdd(&grad_x[src], alpha * clamp_grad(dx * grad_coeff));
    //         atomicAdd(&grad_y[src], alpha * clamp_grad(dy * grad_coeff));
    //     }
    // }
}