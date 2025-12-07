// edited by Simon Langowski to use montgomery multiplication and improve occupancy
// At one point based on the ntt by
// Can Elgezen and Özgün Özerk, contributed by Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk

#pragma once
#include "params.cuh"
#include "structures.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "testing.cuh"
#include "arith.cuh"
#include "utils.cuh"

__device__  __forceinline__  void shuffle(uint32_t &x, uint32_t &y, int mask) {
    uint32_t dir = threadIdx.x & mask;
    uint32_t send = dir ? x : y;
    uint32_t recv = __shfl_xor_sync(UINT32_MAX, send, mask);
    if (dir) {
        x = recv;
    } else {
        y = recv;
    }
}

__device__ __forceinline__ void butterfly(Regs &r, const int x_idx, const int y_idx, const uint32_t twiddle, const uint32_t modulus, const uint32_t modulus_inv) {
    uint32_t t = montgomery_mult(r[y_idx], twiddle, modulus, modulus_inv);
    r[y_idx] = r[x_idx] - t + 2 * modulus;
    r[x_idx] = r[x_idx] + t;
}

__device__ void ntt_256(Regs &r, const uint32_t modulus, const uint32_t inv_modulus, const uint32_t twiddle_offset=0, const uint32_t scratch_offset=1) {

    extern __shared__ uint32_t shared_mem[];
    // Twiddle values
    uint32_t* twiddles = shared_mem + twiddle_offset;
    // Scratch space for shuffling NTT values
    uint32_t* shared_array = &shared_mem[(threadIdx.y + scratch_offset) * POLY_LEN];

    // put these in constant memory

    // Radix-8 NTT
    // test_hash_reduce(r, "input");
    // stride 1024
    butterfly(r, 0, 4, twiddles[1], modulus, inv_modulus);
    butterfly(r, 1, 5, twiddles[1], modulus, inv_modulus);
    butterfly(r, 2, 6, twiddles[1], modulus, inv_modulus);
    butterfly(r, 3, 7, twiddles[1], modulus, inv_modulus);
    // stride 512
    butterfly(r, 0, 2, twiddles[2], modulus, inv_modulus);
    butterfly(r, 1, 3, twiddles[2], modulus, inv_modulus);
    butterfly(r, 4, 6, twiddles[3], modulus, inv_modulus);
    butterfly(r, 5, 7, twiddles[3], modulus, inv_modulus);
    // stride 256
    butterfly(r, 0, 1, twiddles[4], modulus, inv_modulus);
    butterfly(r, 2, 3, twiddles[5], modulus, inv_modulus);
    butterfly(r, 4, 5, twiddles[6], modulus, inv_modulus);
    butterfly(r, 6, 7, twiddles[7], modulus, inv_modulus);

    // test_hash_reduce(r, "before shared");

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        shared_array[256*i + threadIdx.x] = r[i];
    }
     // so if it goes too fast a warp can write into sm again before everyone has read from it
    __syncthreads(); // barrier wait
    // TODO: perhaps avoid bank conflicts by reading in different ordering and then swapping registers back to correct order
    // read 2 sets of 4 sonsecutive values, spaced by 128
    // each warp has 256 consecutive values within it
    int warp = threadIdx.x >> 5;
    int l = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int x = 4 * l + 256 * warp + i;
        r[i] = shared_array[x];
        r[i + 4] = shared_array[x + 128];
    }

    __syncthreads(); //or barrier arrive
    // TODO: begin memcpy async ?

    // test_hash_reduce(r, "after shared");
    // use shuffle_xor instruction to move values within a warp via butterfly pattern
    #pragma unroll
    for (int i = NUM_RADIX; i < LEN_LOG2 - NUM_RADIX; i++) {
        int twiddle_idx = (1 << i) + (threadIdx.x >> (LEN_LOG2 - NUM_RADIX - i));
        butterfly(r, 0, 4, twiddles[twiddle_idx], modulus, inv_modulus);
        butterfly(r, 1, 5, twiddles[twiddle_idx], modulus, inv_modulus);
        butterfly(r, 2, 6, twiddles[twiddle_idx], modulus, inv_modulus);
        butterfly(r, 3, 7, twiddles[twiddle_idx], modulus, inv_modulus);
        
        // test_hash_reduce(r, "butterfly");
        
        int shift = 1 << (LEN_LOG2 - NUM_RADIX - i - 1);
        shuffle(r[0], r[4], shift);
        shuffle(r[1], r[5], shift);
        shuffle(r[2], r[6], shift);
        shuffle(r[3], r[7], shift);

        // test_hash_reduce(r, "shuffle");

    }
    // can do correction anywhere between (5 and 8) of the 11 iterations
    #pragma unroll
    for (int i = 0; i < 8; i++){
        r[i] -= (8 * modulus) * (r[i] >= (8 * modulus));
    }

    // Radix 8 NTT
    // stride 4
    int twiddle_idx = 256 + threadIdx.x;
    butterfly(r, 0, 4, twiddles[twiddle_idx], modulus, inv_modulus);
    butterfly(r, 1, 5, twiddles[twiddle_idx], modulus, inv_modulus);
    butterfly(r, 2, 6, twiddles[twiddle_idx], modulus, inv_modulus);
    butterfly(r, 3, 7, twiddles[twiddle_idx], modulus, inv_modulus);
    // test_hash_reduce(r, "after 4");
    // stride 2
    int twiddle_base = 512 + threadIdx.x;
    butterfly(r, 0, 2, twiddles[twiddle_base], modulus, inv_modulus);
    butterfly(r, 1, 3, twiddles[twiddle_base], modulus, inv_modulus);
    butterfly(r, 4, 6, twiddles[twiddle_base+256], modulus, inv_modulus);
    butterfly(r, 5, 7, twiddles[twiddle_base+256], modulus, inv_modulus);
    // test_hash_reduce(r, "after 2");
    // stride 1
    twiddle_base = 1024 + threadIdx.x;
    butterfly(r, 0, 1, twiddles[twiddle_base], modulus, inv_modulus);
    butterfly(r, 2, 3, twiddles[twiddle_base+256], modulus, inv_modulus);
    butterfly(r, 4, 5, twiddles[twiddle_base+512], modulus, inv_modulus);
    butterfly(r, 6, 7, twiddles[twiddle_base+(3*256)], modulus, inv_modulus);
    // test_hash_reduce(r, "after ntt_256");
}

__device__ __forceinline__ void intt_butterfly(Regs &r, int x_idx, int y_idx, uint32_t psi, const uint32_t modulus, const uint32_t inv_modulus) {
    uint32_t x = r[x_idx];
    uint32_t y = r[y_idx];
    uint32_t x_prime = x + y;
    x_prime -= modulus * (x_prime >= modulus);
    uint32_t t = x + modulus - y;
    uint32_t y_prime = montgomery_mult(psi, t, modulus, inv_modulus);
    y_prime -= modulus * (y_prime >= modulus);
    r[x_idx] = x_prime;
    r[y_idx] = y_prime;
}

__device__ void intt_256(Regs &r, const uint32_t modulus, const uint32_t inv_modulus, const uint32_t twiddle_offset=0, const uint32_t scratch_offset=1) { // performed on a single polynomial
    // test_hash_reduce(r, "intt_256 input");
    
    extern __shared__ uint32_t shared_mem[];
    uint32_t* twiddles = shared_mem + twiddle_offset;
    uint32_t* shared_array = &shared_mem[(threadIdx.y + scratch_offset) * POLY_LEN];

    // Radix 8 NTT
    // stride 1
    int twiddle_base = 1024 + threadIdx.x;
    intt_butterfly(r, 0, 1, twiddles[twiddle_base], modulus, inv_modulus);
    intt_butterfly(r, 2, 3, twiddles[twiddle_base+256], modulus, inv_modulus);
    intt_butterfly(r, 4, 5, twiddles[twiddle_base+512], modulus, inv_modulus);
    intt_butterfly(r, 6, 7, twiddles[twiddle_base+(3*256)], modulus, inv_modulus);
    // stride 2
    twiddle_base = 512 + threadIdx.x;
    intt_butterfly(r, 0, 2, twiddles[twiddle_base], modulus, inv_modulus);
    intt_butterfly(r, 1, 3, twiddles[twiddle_base], modulus, inv_modulus);
    intt_butterfly(r, 4, 6, twiddles[twiddle_base+256], modulus, inv_modulus);
    intt_butterfly(r, 5, 7, twiddles[twiddle_base+256], modulus, inv_modulus);
    // stride 4
    int twiddle_idx = 256 + 1 * threadIdx.x;
    intt_butterfly(r, 0, 4, twiddles[twiddle_idx], modulus, inv_modulus);
    intt_butterfly(r, 1, 5, twiddles[twiddle_idx], modulus, inv_modulus);
    intt_butterfly(r, 2, 6, twiddles[twiddle_idx], modulus, inv_modulus);
    intt_butterfly(r, 3, 7, twiddles[twiddle_idx], modulus, inv_modulus);
    // test_hash_reduce(r, "after radix");
    
    // use shuffle_xor instruction to move values within a warp via butterfly pattern
    #pragma unroll
    for (int i = LEN_LOG2 - NUM_RADIX - 1; i >= NUM_RADIX; i--) {
        int shift = 1 << (LEN_LOG2 - NUM_RADIX - i - 1);
        shuffle(r[0], r[4], shift);
        shuffle(r[1], r[5], shift);
        shuffle(r[2], r[6], shift);
        shuffle(r[3], r[7], shift);
        int twiddle_idx = (1 << i) + (threadIdx.x >> (LEN_LOG2 - NUM_RADIX - i));
        intt_butterfly(r, 0, 4, twiddles[twiddle_idx], modulus, inv_modulus);
        intt_butterfly(r, 1, 5, twiddles[twiddle_idx], modulus, inv_modulus);
        intt_butterfly(r, 2, 6, twiddles[twiddle_idx], modulus, inv_modulus);
        intt_butterfly(r, 3, 7, twiddles[twiddle_idx], modulus, inv_modulus);
    }

    // test_hash_reduce(r, "after shuffle");

    
    
    // TODO: perhaps avoid bank conflicts by reading in different ordering and then swapping registers back to correct order
    // read 2 sets of 4 sonsecutive values, spaced by 128
    // each warp has 256 consecutive values within it
    int warp = threadIdx.x >> 5;
    int l = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int x = 4 * l + 256 * warp + i;
        shared_array[x] = r[i];
        shared_array[x + 128] = r[i + 4];
    }
    __syncthreads(); // or barrier wait
   
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = shared_array[256*i + threadIdx.x];
    }

    // so if it goes too fast a warp can write into sm again before everyone has read from it
     __syncthreads(); // or barrier arrive
    // test_hash_reduce(r, "after shared");
    // Radix-8 NTT
    // stride 256
    intt_butterfly(r, 0, 1, twiddles[4], modulus, inv_modulus);
    intt_butterfly(r, 2, 3, twiddles[5], modulus, inv_modulus);
    intt_butterfly(r, 4, 5, twiddles[6], modulus, inv_modulus);
    intt_butterfly(r, 6, 7, twiddles[7], modulus, inv_modulus);
    // stride 512
    intt_butterfly(r, 0, 2, twiddles[2], modulus, inv_modulus);
    intt_butterfly(r, 1, 3, twiddles[2], modulus, inv_modulus);
    intt_butterfly(r, 4, 6, twiddles[3], modulus, inv_modulus);
    intt_butterfly(r, 5, 7, twiddles[3], modulus, inv_modulus);
    // stride 1024
    intt_butterfly(r, 0, 4, twiddles[1], modulus, inv_modulus);
    intt_butterfly(r, 1, 5, twiddles[1], modulus, inv_modulus);
    intt_butterfly(r, 2, 6, twiddles[1], modulus, inv_modulus);
    intt_butterfly(r, 3, 7, twiddles[1], modulus, inv_modulus);

    // test_hash_reduce(r, "after intt_256");
}


// __global__ void NTT(uint32_t input[], const uint32_t modulus, const uint32_t inv_modulus, const uint32_t (&twiddles)[POLY_LEN]) {
//     input = &input[blockIdx.x * POLY_LEN];
//     int l = 0;
//     Regs r;
//     #pragma unroll
//     for (int i = 0; i < POLY_LEN; i += NUM_THREADS) {
//         r[l] = input[i + threadIdx.x];
//         l++;
//     }
//     ntt_256(r, modulus, inv_modulus, twiddles);
//     #pragma unroll
//     for (int i = 0; i < 8; i++) {
//         input[threadIdx.x * 8 + i] = r[i];
//     }
// }

// write back in order that will be read
__global__ void NTT_R(uint32_t input[], const uint32_t modulus, const uint32_t inv_modulus, const uint32_t (&twiddles)[POLY_LEN]) {
    load_twiddles(twiddles);
    for (int input_idx = 0; input_idx < REUSE_TEST; input_idx++) {
        uint32_t* test_input = &input[(REUSE_TEST*(NTTS_PER_BLOCK*blockIdx.x + threadIdx.y) + input_idx)* POLY_LEN];
        int l = 0;
        Regs r;
        #pragma unroll
        for (int i = 0; i < POLY_LEN; i += NUM_THREADS) {
            r[l] = test_input[i + threadIdx.x];
            l++;
        }
        ntt_256(r, modulus, inv_modulus);
        l = 0;
        #pragma unroll
        for (int i = 0; i < POLY_LEN; i += NUM_THREADS) {
            test_input[i + threadIdx.x] = r[l];
            l++;
        }
    }
}

// __global__ void INTT(uint32_t input[], const uint32_t modulus, const uint32_t inv_modulus, const uint32_t (&twiddles)[POLY_LEN]) {
//     input = &input[blockIdx.x * POLY_LEN];
//     Regs r;
//     #pragma unroll
//     for (int i = 0; i < 8; i++) {
//         r[i] = input[threadIdx.x * 8 + i];
//         r[i] -= (8 * modulus) * (r[i] >= (8 * modulus));
//         r[i] -= (4 * modulus) * (r[i] >= (4 * modulus));
//         r[i] -= (2 * modulus) * (r[i] >= (2 * modulus));
//         r[i] -= (1 * modulus) * (r[i] >= (1 * modulus));
//     }
//     intt_256(r, modulus, inv_modulus, twiddles);
//     int l = 0;
//     #pragma unroll
//     for (int i = 0; i < POLY_LEN; i += NUM_THREADS) {
//         input[i + threadIdx.x] = r[l];
//         l++;
//     }
// }

__global__ void INTT_R(uint32_t input[], const uint32_t modulus, const uint32_t inv_modulus, const uint32_t (&twiddles)[POLY_LEN]) {
    load_twiddles(twiddles);
    for (int input_idx = 0; input_idx < REUSE_TEST; input_idx++) {
        uint32_t* test_input = &input[(REUSE_TEST*(NTTS_PER_BLOCK*blockIdx.x + threadIdx.y) + input_idx)* POLY_LEN];
        Regs r;
        int l = 0;
        #pragma unroll
        for (int i = 0; i < POLY_LEN; i += NUM_THREADS) {
            r[l] = test_input[i + threadIdx.x];
            r[l] -= (8 * modulus) * (r[l] >= (8 * modulus));
            r[l] -= (4 * modulus) * (r[l] >= (4 * modulus));
            r[l] -= (2 * modulus) * (r[l] >= (2 * modulus));
            r[l] -= (1 * modulus) * (r[l] >= (1 * modulus));
            l++;
        }
        intt_256(r, modulus, inv_modulus);
        l = 0;
        #pragma unroll
        for (int i = 0; i < POLY_LEN; i += NUM_THREADS) {
            test_input[i + threadIdx.x] = r[l];
            l++;
        }
    }
}