#pragma once
#include "params.cuh"

__device__ __forceinline__ void load_twiddles(const uint32_t (&twiddles)[POLY_LEN]) {
    extern __shared__ uint32_t shared_mem[];
    __syncthreads(); // or barrier wait
    #pragma unroll
    for (int i = 0; i < POLY_LEN; i += (NUM_THREADS*NTTS_PER_BLOCK)) {
        int z = i + threadIdx.x + (NUM_THREADS*threadIdx.y);
        shared_mem[z] = twiddles[z];
    }
    __syncthreads(); // barrier arrive
}

__device__ __forceinline__ void load_all_twiddles(const Twiddles* twiddles) {
    extern __shared__ uint32_t shared_mem[];
    #pragma unroll
    for (int i = 0; i < POLY_LEN; i += (NUM_THREADS*NTTS_PER_BLOCK)) {
        int z = i + threadIdx.x + (NUM_THREADS*threadIdx.y);
        shared_mem[z] = twiddles->q0_twiddles[z];
    }
    #pragma unroll
    for (int i = 0; i < POLY_LEN; i += (NUM_THREADS*NTTS_PER_BLOCK)) {
        int z =POLY_LEN + i + threadIdx.x + (NUM_THREADS*threadIdx.y);
        shared_mem[z] = twiddles->q1_twiddles[z];
    }
    #pragma unroll
    for (int i = 0; i < POLY_LEN; i += (NUM_THREADS*NTTS_PER_BLOCK)) {
        int z = 2*POLY_LEN + i + threadIdx.x + (NUM_THREADS*threadIdx.y);
        shared_mem[z] = twiddles->q0_inv_twiddles[z];
    }
    #pragma unroll
    for (int i = 0; i < POLY_LEN; i += (NUM_THREADS*NTTS_PER_BLOCK)) {
        int z = 3*POLY_LEN + i + threadIdx.x + (NUM_THREADS*threadIdx.y);
        shared_mem[z] = twiddles->q1_inv_twiddles[z];
        printf("z = %d, val = %u\n", z, shared_mem[z]);
    }
    __syncthreads();
}