#pragma once
#include "stdio.h"
#include "params.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.cuh"

// to quickly compare program state to cpu version
// I suppose I will want to do a warp_reduce_shufl of some kind
// And then just have thread 0 of each warp print out a hash
// And emulate this in other versions
__device__ __host__ __forceinline__ unsigned test_hash(const uint32_t (&v)[POLY_LEN]) {
    // DBJ hash
    unsigned h = 5381;
    for (int i = 0; i < POLY_LEN; i++) {
        h = (h << 5) + h + v[i];
    }
    return h;
}

__device__ __forceinline__ void test_hash_reduce(Regs &r, const char* txt) {
    unsigned h = 5381;
    #pragma unroll
    for (int i = 0; i < POLY_LEN / NUM_THREADS; i++) {
        h = (h << 5) + h + r[i];
    }
    #pragma unroll
    for (int o = 16; o > 0; o /= 2) {
        unsigned h_prime = __shfl_down_sync(UINT32_MAX, h, o);
        h = (h << 5) + h + h_prime;
    }
    if ((threadIdx.x & 31) == 0) {
        printf("(%d-%s) %u\n", threadIdx.x, txt, h);
    }
}

__global__ void TEST_TEST() {
    Regs r;
    #pragma unroll
    for (int i = 0; i < POLY_LEN / NUM_THREADS; i++) {
        r[i] = i + threadIdx.x * (POLY_LEN / NUM_THREADS);
    }
    test_hash_reduce(r, "test");
    __syncthreads();
}
