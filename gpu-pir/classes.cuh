#include "structures.cuh"
#include "params.cuh"
#include "utils.cuh"
#include "ntt_28bit.cuh"
#include "arith.cuh"
#include "gadget.cuh"
#include <cuda/std/array>

class PolyNTT {
    public:
    cuda::std::array<uint32_t, COEFFS_PER_THREAD> coeffs_q1;
    cuda::std::array<uint32_t, COEFFS_PER_THREAD> coeffs_q2;
    
    __device__ PolyNTT add(const PolyNTT &other) const {
        PolyNTT out;
        for (int i = 0; i < COEFFS_PER_THREAD; i++) {
            out.coeffs_q1[i] = coeffs_q1[i] + other.coeffs_q1[i];
            out.coeffs_q2[i] = coeffs_q2[i] + other.coeffs_q2[i];
            out.coeffs_q1[i] -= (Q0) * (out.coeffs_q1[i] >= (Q0));
            out.coeffs_q2[i] -= (Q1) * (out.coeffs_q2[i] >= (Q1));
        }
        return out;
    }

    __device__ PolyNTT mult(const PolyNTT &other) const {
        PolyNTT out;
        for (int i = 0; i < COEFFS_PER_THREAD; i++) {
            out.coeffs_q1[i] = barrett_mult(coeffs_q1[i], other.coeffs_q1[i], Q0, Q0_BARRETT);
            out.coeffs_q2[i] = barrett_mult(coeffs_q2[i], other.coeffs_q2[i], Q1, Q1_BARRETT);
            out.coeffs_q1[i] -= (Q0) * (out.coeffs_q1[i] >= (Q0));
            out.coeffs_q2[i] -= (Q1) * (out.coeffs_q2[i] >= (Q1));
        }
        return out;
    }

    __device__ PolyNTT neg() const {
        PolyNTT out;
        for (int i = 0; i < COEFFS_PER_THREAD; i++) {
            out.coeffs_q1[i] = (Q0) - out.coeffs_q1[i];
            out.coeffs_q2[i] = (Q1) - out.coeffs_q2[i];
        }
        return out;
    }
};

typedef struct PolyRaw {
    cuda::std::array<uint64_t, COEFFS_PER_THREAD> coeffs;
} PolyRaw;

typedef struct CiphertextNTT {
    PolyNTT a;
    PolyNTT as_e;
} CiphertextNTT;

typedef struct CiphertextRaw {
    PolyRaw a;
    PolyRaw as_e;
} CiphertextRaw;


// __device__ PolyNTT to_ntt(const PolyRaw &poly_raw) {
//     PolyNTT out;

//     ntt_256(out.coeffs_q1, Q0, Q0_MONT, 0, TOTAL_TWIDDLES);
//     ntt_256(out.coeffs_q2, Q1, Q1_MONT, 1, TOTAL_TWIDDLES);
//     return out;
// }

__device__ PolyRaw to_raw(const PolyNTT &poly_ntt) {
    PolyNTT tmp;
    tmp.coeffs_q1 = poly_ntt.coeffs_q1;
    tmp.coeffs_q2 = poly_ntt.coeffs_q2;
    // #pragma unroll
    // for (int i = 0; i < COEFFS_PER_THREAD; i++) {
    //     tmp.coeffs_q1[i] -= Q0 * (tmp.coeffs_q1[i] >= Q0);
    //     tmp.coeffs_q2[i] -= Q1 * (tmp.coeffs_q2[i] >= Q1);
    // }
    
    Regs tmp_regs_q1;
    Regs tmp_regs_q2;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        tmp_regs_q1[j] = tmp.coeffs_q1[j];
        tmp_regs_q2[j] = tmp.coeffs_q2[j];
    }
    intt_256(tmp_regs_q1, Q0, Q0_MONT, 2, TOTAL_TWIDDLES);
    intt_256(tmp_regs_q2, Q1, Q1_MONT, 3, TOTAL_TWIDDLES);
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        tmp.coeffs_q1[j] = tmp_regs_q1[j];
        tmp.coeffs_q2[j] = tmp_regs_q2[j];
    }
    PolyRaw out;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs[j] = inv_crt(tmp.coeffs_q1[j], tmp.coeffs_q2[j]);
    }
    return out;
}

// old:
/*
__device__ PolyRaw to_raw(const PolyNTT &poly_ntt) {
    PolyNTT tmp;
    tmp.coeffs_q1 = poly_ntt.coeffs_q1;
    tmp.coeffs_q2 = poly_ntt.coeffs_q2;
    // #pragma unroll
    // for (int i = 0; i < COEFFS_PER_THREAD; i++) {
    //     tmp.coeffs_q1[i] -= Q0 * (tmp.coeffs_q1[i] >= Q0);
    //     tmp.coeffs_q2[i] -= Q1 * (tmp.coeffs_q2[i] >= Q1);
    // }
    intt_256(tmp.coeffs_q1, Q0, Q0_MONT, 2, TOTAL_TWIDDLES);
    intt_256(tmp.coeffs_q2, Q1, Q1_MONT, 3, TOTAL_TWIDDLES);
    PolyRaw out;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs[j] = inv_crt(tmp.coeffs_q1[j], tmp.coeffs_q2[j]);
    }
    return out;
}
*/

__device__ CiphertextRaw to_raw_ct(const CiphertextNTT &ciphertext_ntt) {
    CiphertextRaw out;
    out.a = to_raw(ciphertext_ntt.a);
    out.as_e = to_raw(ciphertext_ntt.as_e);
    return out;
}

// corrects all redundant values for a given modulus, allowing matching to CPU code
template<int modulus>
__device__ __forceinline__ uint32_t correct_all(uint32_t redundant_value) {
    #pragma unroll
    for (int i = 0; i < (32 - (MOD_BITS / 2)); i++) {
        redundant_value -= (modulus << i) * (redundant_value >= (modulus << i));
    }
    return redundant_value;
}

__device__ PolyNTT to_ntt(const PolyRaw &poly_raw) {
    PolyNTT out;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs_q1[j] = barrett_reduce(poly_raw.coeffs[j], MODULI_BARRETT[0], MODULI[0]);
        out.coeffs_q2[j] = barrett_reduce(poly_raw.coeffs[j], MODULI_BARRETT[1], MODULI[1]);
        // out.coeffs_q1[j] -= (Q0) * (out.coeffs_q1[j] >= (Q0));
        // out.coeffs_q2[j] -= (Q1) * (out.coeffs_q2[j] >= (Q1));
    }

    Regs regs_q1;
    Regs regs_q2;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        regs_q1[j] = out.coeffs_q1[j];
        regs_q2[j] = out.coeffs_q2[j];
    }

    ntt_256(regs_q1, Q0, Q0_MONT, 0, TOTAL_TWIDDLES);
    ntt_256(regs_q2, Q1, Q1_MONT, 1, TOTAL_TWIDDLES);

    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs_q1[j] = regs_q1[j];
        out.coeffs_q2[j] = regs_q2[j];
    }

    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs_q1[j] = correct_all<Q0>(out.coeffs_q1[j]);
        out.coeffs_q2[j] = correct_all<Q1>(out.coeffs_q2[j]);
    }
    return out;
}

// old:
/*
__device__ PolyNTT to_ntt(const PolyRaw &poly_raw) {
    PolyNTT out;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs_q1[j] = barrett_reduce(poly_raw.coeffs[j], Q0_BARRETT, Q0);
        out.coeffs_q2[j] = barrett_reduce(poly_raw.coeffs[j], Q1_BARRETT, Q1);
        // out.coeffs_q1[j] -= (Q0) * (out.coeffs_q1[j] >= (Q0));
        // out.coeffs_q2[j] -= (Q1) * (out.coeffs_q2[j] >= (Q1));
    }
    ntt_256(out.coeffs_q1, Q0, Q0_MONT, 0, TOTAL_TWIDDLES);
    ntt_256(out.coeffs_q2, Q1, Q1_MONT, 1, TOTAL_TWIDDLES);
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.coeffs_q1[j] = correct_all<Q0>(out.coeffs_q1[j]);
        out.coeffs_q2[j] = correct_all<Q1>(out.coeffs_q2[j]);
    }
    return out;
}
*/

__device__ CiphertextNTT to_ntt_ct(const CiphertextRaw &ciphertext_raw) {
    CiphertextNTT out;
    out.a = to_ntt(ciphertext_raw.a);
    out.as_e = to_ntt(ciphertext_raw.as_e);
    return out;
}

// We could save an addition by setting acc_a and acc_as_e to some value we want to add to
__device__ CiphertextNTT gadget_inv_rdim1(PolyRaw &input, const CiphertextHalf mul0[], const CiphertextHalf mul1[], const int log_base_bits) {
    Poly<uint32_t> input_in_high_low_crt;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        input_in_high_low_crt.crt[0].data[j] = (uint32_t)(input.coeffs[j] >> 32);
        input_in_high_low_crt.crt[1].data[j] = (uint32_t)(input.coeffs[j]);
    }
    Twiddles twiddles;
    Regs acc_a_q1 = {0};
    Regs acc_as_e_q1 = {0};
    Regs acc_a_q2 = {0};
    Regs acc_as_e_q2 = {0};
    inv_gadget_ntt_mult_rdim1(twiddles, input_in_high_low_crt, acc_a_q1, acc_as_e_q1, mul0, log_base_bits, Q0, Q0_MONT);
    inv_gadget_ntt_mult_rdim1(twiddles, input_in_high_low_crt, acc_a_q2, acc_as_e_q2, mul1, log_base_bits, Q1, Q1_MONT);
    CiphertextNTT out;
    #pragma unroll
    for (int j = 0; j < COEFFS_PER_THREAD; j++) {
        out.a.coeffs_q1[j] = acc_a_q1[j];
        out.a.coeffs_q2[j] = acc_a_q2[j];
        out.as_e.coeffs_q1[j] = acc_as_e_q1[j];
        out.as_e.coeffs_q2[j] = acc_as_e_q2[j];
        out.a.coeffs_q1[j] = correct_all<MODULI[0]>(out.a.coeffs_q1[j]);
        out.a.coeffs_q2[j] = correct_all<MODULI[1]>(out.a.coeffs_q2[j]);   
        out.as_e.coeffs_q1[j] = correct_all<MODULI[0]>(out.as_e.coeffs_q1[j]);
        out.as_e.coeffs_q2[j] = correct_all<MODULI[1]>(out.as_e.coeffs_q2[j]);
    }
    return out;
}