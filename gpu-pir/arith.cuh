#pragma once
#include "params.cuh"
#include "structures.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

typedef unsigned __int128 uint128_t;

__device__ __forceinline__ uint32_t montgomery_reduce(uint64_t u, const uint32_t p, const uint32_t p_inv) {
    uint32_t u_0 = u;
    uint32_t u_1 = u >> 32;
    uint32_t q = u_0 * p_inv;
    uint32_t h = __umulhi(q, p);
    return u_1 + p - h;
}

// input: (0, 14p), (0, p)
// output: (0, 2p)
__device__ __forceinline__ uint32_t montgomery_mult(uint32_t a, const uint32_t w_prime, const uint32_t p, const uint32_t p_inv) {  
    uint64_t u = (uint64_t)(a) * (uint64_t)(w_prime);
    return montgomery_reduce(u, p, p_inv);
}

// mu = 2**(32 + modulus_bits) / modulus, rounded down
__device__ __forceinline__ uint32_t barrett_reduce(uint64_t value, uint32_t mu, uint32_t modulus) {
    uint32_t hi = value >> (MOD_BITS/2); // This shift is why barrett is more expensive than Montgomery, since shift by 32 is "free"
    uint32_t q = __umulhi(hi, mu) + hi;
    return value - q * modulus;
}

__device__ __forceinline__ uint32_t barrett_mult(uint32_t a, uint32_t b, uint32_t mu, uint32_t modulus) {
    uint64_t u = (uint64_t)(a) * (uint64_t)(b);
    return barrett_reduce(u, mu, modulus);
}

// input: (0, 14p) (0, p) (0, 2p)
// output: (0, 2p)
__device__ __forceinline__ void mod_mult_add_regs(Regs &r1, const PolyHalf<uint32_t> &c1, Regs &dst, const uint32_t mod, const uint32_t mont_mod) {
    int j = 0;
    #pragma unroll
    for (int idx = 0; idx < POLY_LEN; idx += NUM_THREADS) {
        int z = idx + threadIdx.x;
        uint32_t out = dst[j] + montgomery_mult(r1[j], c1.data[z], mod, mont_mod);
        out -= (2 * mod) * (out >= 2 * mod);
        dst[j] = out;
        j++;
    }
}

__device__ __forceinline__  uint64_t inv_crt(uint32_t a, uint32_t b) {
    a = montgomery_mult(a, CRT_INV_FOR0, Q0, Q0_MONT);
    a -= Q0 * (a >= Q0);
    b = montgomery_mult(b, CRT_INV_FOR1, Q1, Q1_MONT);
    b -= Q1 * (b >= Q1);
    uint64_t s = ((uint64_t)(a) * Q1) + ((uint64_t)(b) * Q0);
    s -= (MODULUS) * (s >= MODULUS);
    return s;
}

__device__ __forceinline__ void inv_crt_regs(Regs &r_Q1, Regs &r_Q2) {
    #pragma unroll
    for (int j = 0; j < 2 * NUM_PAIRS; j++) {
        uint64_t c = inv_crt(r_Q1[j], r_Q2[j]);
        // high bits
        r_Q1[j] = (uint32_t)(c >> 32);
        // low bits
        r_Q2[j] = (uint32_t)(c);
    }
}

__device__ __host__ uint64_t barrett(uint64_t input, uint64_t const_ratio_1, uint64_t modulus) {
    // const_ratio_1 = floor(2^64 / modulus) - it's a precomputed value
    uint64_t tmp = (uint128_t)(input) * (uint128_t)(const_ratio_1) >> 64;
    // Barrett subtraction
    uint64_t res = input - tmp * modulus;

    // One more subtraction if necessary
    res -= modulus * (res >= modulus);
    return res;
}

__global__ void barrett_raw_u64(uint64_t* a, uint64_t* b, uint64_t* dest_a, uint64_t const_ratio_1, uint64_t modulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dest_a[idx] = barrett(a[idx] + b[idx], const_ratio_1, modulus); // fix later to make the function do what i want
    printf("Thread idx %d computed %lu\n", threadIdx.x, dest_a[idx]);
}