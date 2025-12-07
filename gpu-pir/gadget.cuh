#pragma once
#include "params.cuh"
#include "structures.cuh"
#include "ntt_28bit.cuh"
#include "arith.cuh"
#include "utils.cuh"

__device__ __forceinline__ void from_ntt_regs(const Twiddles &twiddles, Regs &q1_values, Regs &q2_values) {
    // reduce (0,2p) to (0,p)
    #pragma unroll
    for (int i = 0; i < 2 * NUM_PAIRS; i++) {
        q1_values[i] -= Q0 * (q1_values[i] >= Q0);
        q2_values[i] -= Q1 * (q2_values[i] >= Q1);
    }
    load_twiddles(twiddles.q0_inv_twiddles);
    intt_256(q1_values, Q0, Q0_MONT);
    // test_hash_reduce(q1_values, "q0 intt");
    load_twiddles(twiddles.q1_inv_twiddles);
    intt_256(q2_values, Q1, Q1_MONT);
    // test_hash_reduce(q2_values, "q2 intt");
    inv_crt_regs(q1_values, q2_values);
    //test_hash_reduce(q1_values, "q1 crt");
    // test_hash_reduce(q2_values, "q2 crt");
}

__device__ __forceinline__ void write_regs(Poly<uint32_t> &dst, Regs &q1_values, Regs &q2_values) {
    int i = 0;
    #pragma unroll
    for (int idx = 0; idx < POLY_LEN; idx += NUM_THREADS) {
        int z = idx + threadIdx.x;
        dst.crt[0].data[z] = q1_values[i];
        dst.crt[1].data[z] = q2_values[i];
        i++;
    }
}

__device__ __forceinline__ int get_bits_per(const int dim) {
    return (MOD_BITS + dim - 1) / dim;
}

// multiplication matrix
// expansion: 2 by t_exp_left or t_exp_right
// conversion: 2 by 2t_conv
// folding: 2 by 2t_gsw
// times
// input matrix
// expansion: 1 by 1 -> t_exp_left (or right) by 1
// conversion: 2 by 1 -> 2t_conv by 1
// folding: 2 by 1 -> 2t_gsw by 1
// equals and add to
// output accumulator
// expansion: 2 by 1
// conversion: 2 by 1
// folding: 2 by 1
__device__ void inv_gadget_ntt_mult1(const Twiddles &twiddles, const Poly<uint32_t> &input, Regs &acc_a, Regs &acc_as_e, const CiphertextHalf mat[], const int j, const int rdim, const int num_elems, const int bits_per, const uint32_t mask, const uint32_t q, const uint32_t q_inv) {
    // let the compiler decide to either unroll this loop or inline ntt
    for (int k = 0; k < num_elems; k++) {
        int row = j + k * rdim;
        int shift = k * bits_per;
        Regs r;
        int i = 0;
        #pragma unroll
        for (int idx = 0; idx < POLY_LEN; idx += NUM_THREADS) {
            // possible to load only the part of the input needed rather than all 64 bits
            // TODO: is it okay to change input?
            int z = idx + threadIdx.x;
            uint32_t high = input.crt[0].data[z];
            uint32_t low = input.crt[1].data[z];
            uint64_t combined = ((uint64_t)(high) << 32) + (uint64_t)(low);
            combined = combined >> shift;
            combined = combined & mask;
            r[i] = (uint32_t)(combined);
            i++;
        }
        // test_hash_reduce(r, "input");
        ntt_256(r, q, q_inv);
        // test_hash_reduce(r, "ntt");
        mod_mult_add_regs(r, mat[row].polys[0], acc_a, q, q_inv);
        // test_hash_reduce(acc_a, "add");
        mod_mult_add_regs(r, mat[row].polys[1], acc_as_e, q, q_inv);
    }
}

__device__ __forceinline__ void inv_gadget_ntt_mult_rdim2(const Twiddles &twiddles, const Ciphertext &input, Regs &acc_a, Regs &acc_as_e, const CiphertextHalf mul[], const int mx, const int q, const int q_inv) {
    // these divisions should be inlined away
    const int rdim = 2;
    const int num_elems = mx / rdim;
    const int bits_per = get_bits_per(num_elems);
    const uint32_t mask = (1 << bits_per) - 1;
    inv_gadget_ntt_mult1(twiddles, input.polys[0], acc_a, acc_as_e, mul, 0, rdim, num_elems, bits_per, mask, q, q_inv);
    inv_gadget_ntt_mult1(twiddles, input.polys[1], acc_a, acc_as_e, mul, 1, rdim, num_elems, bits_per, mask, q, q_inv);
}

__device__ __forceinline__ void inv_gadget_ntt_mult_rdim1(const Twiddles &twiddles, const Poly<uint32_t> &input, Regs &acc_a, Regs &acc_as_e, const CiphertextHalf mul[], const int mx, const int q, const int q_inv) { // q, q_inv are moduli
    // these divisions should be inlined away
    const int rdim = 1;
    const int num_elems = mx / rdim;
    const int bits_per = get_bits_per(num_elems);
    const uint32_t mask = (1 << bits_per) - 1;
    inv_gadget_ntt_mult1(twiddles, input, acc_a, acc_as_e, mul, 0, rdim, num_elems, bits_per, mask, q, q_inv);
}