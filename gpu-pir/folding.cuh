#pragma once

#include "params.cuh"
#include "structures.cuh"
#include "gadget.cuh"
#include "arith.cuh"
#include "utils.cuh"

__device__ void fold_ciphertexts(const Twiddles &twiddles, Ciphertext accumulators[], const CiphertextHalf (&v_folding)[V2][2][2*T_GSW], const CiphertextHalf (&v_folding_neg)[V2][2][2*T_GSW], const int num_per, const int i, const int cur_dim) {
    Regs acc_a_q0 = {0};
    Regs acc_as_e_q0 = {0};
    load_twiddles(twiddles.q0_twiddles);
    inv_gadget_ntt_mult_rdim2(twiddles, accumulators[num_per + i], acc_a_q0, acc_as_e_q0, v_folding[V2 - 1 - cur_dim][0], 2*T_GSW, Q0, Q0_MONT);
    inv_gadget_ntt_mult_rdim2(twiddles, accumulators[i], acc_a_q0, acc_as_e_q0, v_folding_neg[V2 - 1 - cur_dim][0], 2*T_GSW, Q0, Q0_MONT);
    
    // likely will require spill to local memory of q0 reg values...
    Regs acc_a_q1 = {0};
    Regs acc_as_e_q1 = {0};
    load_twiddles(twiddles.q1_twiddles);
    inv_gadget_ntt_mult_rdim2(twiddles, accumulators[num_per + i], acc_a_q1, acc_as_e_q1, v_folding[V2 - 1 - cur_dim][1], 2*T_GSW, Q1, Q1_MONT);
    inv_gadget_ntt_mult_rdim2(twiddles, accumulators[i], acc_a_q1, acc_as_e_q1, v_folding_neg[V2 - 1 - cur_dim][1], 2*T_GSW, Q1, Q1_MONT);
    
    from_ntt_regs(twiddles, acc_a_q0, acc_a_q1);
    write_regs(accumulators[i].polys[0], acc_a_q0, acc_a_q1);
    from_ntt_regs(twiddles, acc_as_e_q0, acc_as_e_q1);
    write_regs(accumulators[i].polys[1], acc_as_e_q0, acc_as_e_q1);
}

const bool REUSE_PARAMS = false;

__global__ 
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
void FOLD1(const Twiddles* twiddles, FoldingQueryStorage *query, const int cur_dim) {
    int input_idx = NTTS_PER_BLOCK*blockIdx.x + threadIdx.y;
    int i = blockIdx.y;
    int num_per = 1 << (V2 - cur_dim - 1);
    int param_idx = input_idx;
    if (REUSE_PARAMS) {
        // test what happens if both NTTs operate on the same query
        param_idx = NTTS_PER_BLOCK*blockIdx.x;
    }
    fold_ciphertexts(*twiddles, query[input_idx].accumulators[0], query[param_idx].v_folding, query[param_idx].v_folding_neg, num_per, i, cur_dim);
}

__device__ void fold_all_ciphertexts(const Twiddles &twiddles, Ciphertext accumulators[], const CiphertextHalf (&v_folding)[V2][2][2*T_GSW], const CiphertextHalf (&v_folding_neg)[V2][2][2*T_GSW]) {
    int num_per = 1 << V2;
    for (int cur_dim = 0; cur_dim < V2; cur_dim++) {
        num_per = num_per / 2;
        for (int i = 0; i < num_per; i++) {
            fold_ciphertexts(twiddles, accumulators, v_folding, v_folding_neg, num_per, i, cur_dim);
        }
    }
}

__global__
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
void FOLD_CIPHERTEXTS(const Twiddles* twiddles, FoldingQueryStorage *query) {
    int input_idx = NTTS_PER_BLOCK*blockIdx.x + threadIdx.y;
    int param_idx = input_idx;
    if (REUSE_PARAMS) {
        // test what happens if both NTTs operate on the same query
        param_idx = NTTS_PER_BLOCK*blockIdx.x;
    }
    fold_all_ciphertexts(*twiddles, query[input_idx].accumulators[0], query[param_idx].v_folding, query[param_idx].v_folding_neg);
}