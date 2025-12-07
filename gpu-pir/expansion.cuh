#pragma once
#include "params.cuh"
#include "structures.cuh"
#include "automorph.cuh"
#include "gadget.cuh"
#include "ntt_28bit.cuh"
#include "classes.cuh"
#include "arith.cuh"
#include <cuda/std/array>

// if v_i is 4x too big, take 4 blocks to handle one v_i and have each block handle 1/4th of v_i, or have one block handle one v_i and take 1/4 of the data at once (4 rounds, probably easier)
// when a polynomial is in memory it has 2048 coefficients, when it's in a thread it has 8
// rows = a, as+e
// crt_count = moduli (q1_coeffs, q2_coeffs)

/*
inputs to coefficient_expansion:
- v = vector of size (1 << (params.db_dim_1 + 1)), set to zero except v[0], which is the query ciphertexts (meaning has a and as+e, and is rows = 2, cols = 1. row 1 = a, row 2 = as+e)
- params.g() = log2_ceil_usize(params.t_gsw * params.db_dim_2 + (1 << params.db_dim_1))
- params.stop_round() = log2_ceil_usize(params.t_gsw * params.db_dim_2)
- params
- v_w_left = client key; vector of length params.g() (each element is a PolyMatrixNTT with rows = 2, cols = 8) TODO: figure out why it's 2 and 8
    - v_w_left = public_params.v_expansion_left.as_ref().unwrap();
- v_w_right = client key; vector of length params.stop_round() + 1
    - v_w_right = public_params.v_expansion_right.as_ref().unwrap();
- v_neg1 = vector of length params.poly_len_log2 (each element is a PolyMatrixNTT with rows = 1, cols = 1)
- max_bits_to_gen_right = params.t_gsw * params.db_dim_2
*/

const uint64_t v_starting_index_shared_memory = 4 * POLY_LEN; // first 4 * POLY_LEN indices are for twiddles

constexpr uint64_t ceil_log2_uint64(uint64_t x) {
    uint64_t v = 0;
    uint64_t pow = 1;
    while (pow < x) { pow <<= 1; ++v; }
    return v;
}

uint64_t v_to_shared_memory_idx(uint64_t coeff_mod_256, uint64_t coeff_local, uint64_t modulus, uint64_t row, uint64_t i) {
    return v_starting_index_shared_memory + (i * (sizeof(CiphertextNTT) / sizeof(uint32_t)) + row * (sizeof(PolyNTT) / sizeof(uint32_t)) + modulus * POLY_LEN + coeff_local * NUM_THREADS + coeff_mod_256);
}

void load_v_i_to_shared_memory(CiphertextNTT &v_i) {
    extern __shared__ uint32_t shared_mem[];

    uint64_t coeff_mod_256 = threadIdx.x;
    uint64_t i = blockIdx.x;

    #pragma unroll
    for (int coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
        // modulus = 0, row = 0 (q1, a)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 0, i)] = v_i.a.coeffs_q1[coeff_local];
        // modulus = 1, row = 0 (q2, a)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 0, i)] = v_i.a.coeffs_q2[coeff_local];
        // modulus = 0, row = 1 (q1, as+e)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 1, i)] = v_i.as_e.coeffs_q1[coeff_local];
        // modulus = 1, row = 1 (q2, as+e)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 1, i)] = v_i.as_e.coeffs_q2[coeff_local];
    }
    __syncthreads();
}

void get_v_i_from_shared_memory(CiphertextNTT &v_i) {
    extern __shared__ uint32_t shared_mem[];

    uint64_t coeff_mod_256 = threadIdx.x;
    uint64_t i = blockIdx.x;

    #pragma unroll
    for (int coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
        // modulus = 0, row = 0 (q1, a)
        v_i.a.coeffs_q1[coeff_local] = shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 0, i)];
        // modulus = 1, row = 0 (q2, a)
        v_i.a.coeffs_q2[coeff_local] = shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 0, i)];
        // modulus = 0, row = 1 (q1, as+e)
        v_i.as_e.coeffs_q1[coeff_local] = shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 1, i)];
        // modulus = 1, row = 1 (q2, as+e)
        v_i.as_e.coeffs_q2[coeff_local] = shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 1, i)];
    }
    __syncthreads();
}

const uint64_t g = ceil_log2_uint64(T_GSW * V2 + (1ULL << V1));
const uint64_t stop_round = ceil_log2_uint64(T_GSW * V2);
const uint64_t max_bits_to_gen_right = T_GSW * V2;

__device__ __host__ void action_expand(CiphertextNTT &v_i, PublicParameters *public_parameters, const Twiddles &twiddles, uint64_t r) {
    uint64_t i = blockIdx.x;
    
    uint64_t t = (POLY_LEN / (1 << r)) + 1;

    if (stop_round > 0 && r > stop_round && (i % 2) == 1) {
        return;
    }
    if (stop_round > 0 && r == stop_round && (i % 2) == 1 && (i / 2) >= max_bits_to_gen_right) {
        return;
    }

    CiphertextRaw ct;
    CiphertextRaw ct_auto; // automorphed ct
    PolyRaw ct_auto_1; // second row (as+e part) of automorphed ct in raw form
    PolyNTT ct_auto_1_ntt; // second row (as+e part) of automorphed ct in ntt form
    CiphertextNTT w_times_ginv_ct;

    cudaMemset(ct.a.coeffs.data(), 0, sizeof(ct.a.coeffs));
    cudaMemset(ct.as_e.coeffs.data(), 0, sizeof(ct.as_e.coeffs));
    cudaMemset(ct_auto.a.coeffs.data(), 0, sizeof(ct_auto.a.coeffs));
    cudaMemset(ct_auto.as_e.coeffs.data(), 0, sizeof(ct_auto.as_e.coeffs));
    cudaMemset(ct_auto_1.coeffs.data(), 0, sizeof(ct_auto_1.coeffs));
    cudaMemset(ct_auto_1_ntt.coeffs_q1.data(), 0, sizeof(ct_auto_1_ntt.coeffs_q1));
    cudaMemset(ct_auto_1_ntt.coeffs_q2.data(), 0, sizeof(ct_auto_1_ntt.coeffs_q2));
    cudaMemset(w_times_ginv_ct.a.coeffs_q1.data(), 0, sizeof(w_times_ginv_ct.a.coeffs_q1));
    cudaMemset(w_times_ginv_ct.a.coeffs_q2.data(), 0, sizeof(w_times_ginv_ct.a.coeffs_q2));
    cudaMemset(w_times_ginv_ct.as_e.coeffs_q1.data(), 0, sizeof(w_times_ginv_ct.as_e.coeffs_q1));
    cudaMemset(w_times_ginv_ct.as_e.coeffs_q2.data(), 0, sizeof(w_times_ginv_ct.as_e.coeffs_q2));

    CiphertextHalf* w_0;
    CiphertextHalf* w_1;
    uint64_t gadget_dim;

    if(r != 0 && (i % 2) == 0) {
        w_0 = &public_parameters->left_expansion[0][r];
        w_1 = &public_parameters->left_expansion[1][r];
        gadget_dim = T_EXP_LEFT;
    }
    else {
        w_0 = &public_parameters->right_expansion[0][r];
        w_1 = &public_parameters->right_expansion[1][r];
        gadget_dim = T_EXP_RIGHT;
    }

    ct.a = to_raw(v_i.a);
    ct.as_e = to_raw(v_i.as_e);
    
    auto ct_a = ct.a.coeffs;
    auto ct_as_e = ct.as_e.coeffs;

    device_automorph(ct_a, t, ct_auto.a.coeffs.data());
    device_automorph(ct_as_e, t, ct_auto.as_e.coeffs.data());

    // CiphertextHalf is a and as+e for one modulus
    w_times_ginv_ct = gadget_inv_rdim1(ct_auto.a, w_0, w_1, gadget_dim); // ct_auto.a because rdim = 1

    ct_auto_1.coeffs = ct_auto.as_e.coeffs;
    ct_auto_1_ntt = to_ntt(ct_auto_1);

    // v_i.a does not have anything from ct_auto_1_ntt added to it
    for(uint64_t coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
        // first modulus
        uint64_t sum = v_i.a.coeffs_q1[coeff_local] + w_times_ginv_ct.a.coeffs_q1[coeff_local];
        v_i.a.coeffs_q1[coeff_local] = barrett(sum, BARRETT_CR_1[0], MODULI[0]);
        
        // second modulus
        sum = v_i.a.coeffs_q2[coeff_local] + w_times_ginv_ct.a.coeffs_q2[coeff_local];
        v_i.a.coeffs_q2[coeff_local] = barrett(sum, BARRETT_CR_1[1], MODULI[1]);
    }

    // v_i.as_e has the as+e part of ct_auto_1_ntt added to it
    for(uint64_t coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
        uint64_t sum = v_i.as_e.coeffs_q1[coeff_local] + w_times_ginv_ct.as_e.coeffs_q1[coeff_local] + ct_auto_1_ntt.coeffs_q1[coeff_local];
        v_i.as_e.coeffs_q1[coeff_local] = barrett(sum, BARRETT_CR_1[0], MODULI[0]);

        sum = v_i.as_e.coeffs_q2[coeff_local] + w_times_ginv_ct.as_e.coeffs_q2[coeff_local] + ct_auto_1_ntt.coeffs_q2[coeff_local];
        v_i.as_e.coeffs_q2[coeff_local] = barrett(sum, BARRETT_CR_1[1], MODULI[1]);
    }
}

// size of v_i = 2 * 1 * POLY_LEN * 8 bytes = 32.768KB (one v_i fits into a block)
// GPU version: NVIDIA GeForce RTX 3090 (100KB = 102400 of shared memory)
// action_expand is essentially the kernel, other functions (device_coefficient_expansion, etc) are wrappers that load the twiddles, keys, etc
__device__ __host__ uint64_t device_coefficient_expansion(CiphertextNTT v_i, PublicParameters *public_parameters, const Twiddles &twiddles, PolyNTT& v_neg1_r, int r) { // fix sizing of v_w_left and v_w_right
    // we will put twiddles (32KB) and v_i (32.768KB) into shared memory, 8 more KB reserved for NTT operations
    extern __shared__ uint32_t shared_mem[];

    uint64_t coeff_mod_256 = threadIdx.x;
    uint64_t i = blockIdx.x; // FIX: MAYBE NEEDS TO BE blockIdx.x + num_in
    uint64_t num_in = 1ULL << r;

    // load_v_i_to_shared_memory(v_i); not necessary: already done in coefficient_expansion_r

    #pragma unroll
    for (uint64_t coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
        // v[i].a.coeffs_q1[coeff_local] = barrett(v_neg1[r].coeffs_q1[coeff_local] * v[i - num_in].a.coeffs_q1[coeff_local], MODULI_BARRETT[0], MODULI[0]);
        // v[i].a.coeffs_q2[coeff_local] = barrett(v_neg1[r].coeffs_q2[coeff_local] * v[i - num_in].a.coeffs_q2[coeff_local], MODULI_BARRETT[1], MODULI[1]);
        // v[i].as_e.coeffs_q1[coeff_local] = barrett(v_neg1[r].coeffs_q1[coeff_local] * v[i - num_in].as_e.coeffs_q1[coeff_local], MODULI_BARRETT[0], MODULI[0]);
        // v[i].as_e.coeffs_q2[coeff_local] = barrett(v_neg1[r].coeffs_q2[coeff_local] * v[i - num_in].as_e.coeffs_q2[coeff_local], MODULI_BARRETT[1], MODULI[1]);

        // modulus = 0, row = 0 (q1, a)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 0, i)] =
            barrett(v_neg1_r.coeffs_q1[coeff_local] *
                    shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 0, i - num_in)],
                    MODULI_BARRETT[0], MODULI[0]);

        // modulus = 1, row = 0 (q2, a)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 0, i)] =
            barrett(v_neg1_r.coeffs_q2[coeff_local] *
                    shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 0, i - num_in)],
                    MODULI_BARRETT[1], MODULI[1]);

        // modulus = 0, row = 1 (q1, as+e)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 1, i)] =
            barrett(v_neg1_r.coeffs_q1[coeff_local] *
                    shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 1, i - num_in)],
                    MODULI_BARRETT[0], MODULI[0]);

        // modulus = 1, row = 1 (q2, as+e)
        shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 1, i)] =
            barrett(v_neg1_r.coeffs_q2[coeff_local] *
                    shared_mem[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 1, i - num_in)],
                    MODULI_BARRETT[1], MODULI[1]);
    }
    __syncthreads();

    get_v_i_from_shared_memory(v_i);

    action_expand(v_i, public_parameters, twiddles, r); // action_expand doesn't use shared memory, just updates v_i

    load_v_i_to_shared_memory(v_i);
}

// one dimension
__global__ void coefficient_expansion_r(uint32_t* v, PublicParameters* public_parameters, uint32_t* v_neg1, uint32_t r) {
    Twiddles twiddles;
    load_all_twiddles(&twiddles);

    uint64_t coeff_mod_256 = threadIdx.x;

    uint64_t num_in = 1ULL << r;
    uint64_t num_out = 2 * num_in;

    // load v into shared memory
    #pragma unroll
    for (uint64_t i = 0; i < num_out; i++) {
        CiphertextNTT v_i;
        // get v_i from v
        for(uint64_t coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
            // modulus = 0, row = 0 (q1, a)
            v_i.a.coeffs_q1[coeff_local] = v[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 0, i) - v_starting_index_shared_memory];
            // modulus = 1, row = 0 (q2, a)
            v_i.a.coeffs_q2[coeff_local] = v[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 0, i) - v_starting_index_shared_memory];
            // modulus = 0, row = 1 (q1, as+e)
            v_i.as_e.coeffs_q1[coeff_local] = v[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 0, 1, i) - v_starting_index_shared_memory];
            // modulus = 1, row = 1 (q2, as+e)
            v_i.as_e.coeffs_q2[coeff_local] = v[v_to_shared_memory_idx(coeff_mod_256, coeff_local, 1, 1, i) - v_starting_index_shared_memory];
        }
        load_v_i_to_shared_memory(v_i);
    }

    CiphertextNTT v_i;
    get_v_i_from_shared_memory(v_i);

    // get v_neg1_r from v_neg1 (which has all v_neg1, not specific to r)
    PolyNTT v_neg1_r;
    for(uint64_t coeff_local = 0; coeff_local < COEFFS_PER_THREAD; coeff_local++) {
        v_neg1_r.coeffs_q1[coeff_local] = ((uint32_t*)v_neg1)[r * POLY_LEN + coeff_local * NUM_THREADS + coeff_mod_256];
        v_neg1_r.coeffs_q2[coeff_local] = ((uint32_t*)v_neg1)[(r + g) * POLY_LEN + coeff_local * NUM_THREADS + coeff_mod_256];
    }

    device_coefficient_expansion(v_i, public_parameters, twiddles, v_neg1_r, r);
}