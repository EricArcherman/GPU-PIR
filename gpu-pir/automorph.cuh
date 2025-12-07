#pragma once
#include "params.cuh"
#include "structures.cuh"
#include "classes.cuh"
#include <cuda/std/array>
// to compile: nvcc -arch=sm_86 -cudart static -use_fast_math automorph_test.cu host.cu -o automorph -lcudadevrt -std=c++17
// to run: ./automorph

typedef cuda::std::array<uint64_t, POLY_LEN/NUM_THREADS> PolyLocalFullForm; // Each thread (256 total) holds 8 coefficients of the 2048-term polynomial in a local array (stored as uint64_t)

template<bool global_to_local>
__device__ void move_coeffs(uint64_t* global_coeffs, PolyLocalFullForm& local_coeffs) {
    if constexpr (global_to_local) {
        #pragma unroll
        for (int local_idx = 0; local_idx < POLY_LEN/NUM_THREADS; local_idx++) {
            int global_idx = local_idx * NUM_THREADS + threadIdx.x; // for coalesced memory accesses (@simonq: is this correct?)
            local_coeffs[local_idx] = global_coeffs[global_idx];
        }
    } else {
        #pragma unroll
        for (int local_idx = 0; local_idx < POLY_LEN/NUM_THREADS; local_idx++) {
            int global_idx = local_idx * NUM_THREADS + threadIdx.x;
            global_coeffs[global_idx] = local_coeffs[local_idx];
        }
    }
}

__device__ void device_automorph(PolyLocalFullForm& poly_coeffs, const int t, uint64_t* dest_poly_coeffs) {
    // This array represents shared memory
    extern __shared__ uint32_t shared_mem[]; // we will put poly_coeffs from global into shared memory
    uint64_t* shared_mem_64 = (uint64_t*)shared_mem;

    #pragma unroll
    for (int local_idx = 0; local_idx < POLY_LEN/NUM_THREADS; local_idx++) {
        int coeff = NUM_THREADS * local_idx + threadIdx.x;

        int idx = coeff * t % (POLY_LEN); // the index whether the coefficient is permuted
        int flip = ((coeff * t) / POLY_LEN) % 2; // whether to flip the coefficient as x^n = -1

        shared_mem_64[idx] = flip ? (MODULUS - poly_coeffs[local_idx]) : poly_coeffs[local_idx];
    }
    __syncthreads(); // Make sure all writes are done before reads happen

    #pragma unroll
    for (int local_idx = 0; local_idx < POLY_LEN/NUM_THREADS; local_idx++) {
        int coeff_idx = NUM_THREADS * local_idx + threadIdx.x;
        poly_coeffs[local_idx] = shared_mem_64[coeff_idx];
    }
    __syncthreads(); // Make sure all reads are done before any writes happen in the larger program
}

// the __global__ keyword marks a CUDA kernel, which runs in parallel across many threads; each thread executes the kernel code independently, using its own threadIdx.x to process a unique subset of data
// this function automorphs a single polynomial
__global__ void automorph(uint64_t* poly_coeffs, uint64_t* dest_poly_coeffs, uint64_t* t, int reps) { // remember: automorph happens on each thread
    PolyLocalFullForm poly_local_coeffs;
    // each thread gets its own copy of poly_local_coefficients, which is a local array of 8 uint64_t coefficients
    move_coeffs<true>(poly_coeffs, poly_local_coeffs); // true moves global to local
    for (int i = 0; i < reps; i++) { // NOTE: remove when not testing <-- important!
        device_automorph(poly_local_coeffs, *t, dest_poly_coeffs);
    }
    move_coeffs<false>(dest_poly_coeffs, poly_local_coeffs); // false moves local to global
}