#pragma once

#include "params.cuh"
#include "structures.cuh"
#include "gadget.cuh"
#include "arith.cuh"
#include "utils.cuh"

// each thread responsible for one row of database (happens in blocks of 256 threads)
// block.y responsible for coefficients
// block.z selects for CRT half (hi or lo) and part of ciphertext (a or as_e) 

// short barrett reduction (x mod n) for 32-bit input
__device__ __forceinline__ uint32_t short_barrett(uint32_t x, uint32_t n, uint32_t mu) {
    // TODO: handle when x > 2^32 (e.g., when accumulator gets too large)

    uint32_t q = (uint64_t(x) * mu) >> 32; // uint64_t casting to avoid overflow
    uint32_t r = x - q * n;
    if (r >= n) {
        r -= n;
    }
    return r;
}

// manages dispatching work for each thread
__global__ void db_mul_kernel(
    const Database<Poly<uint32_t>>* db,
    const FirstDim* first_dim,
    FoldingInputs* folding_inputs
) {
    // each thread handles one column of database; coefficients handled by block
    // might need to switch uint128_t
    // todo: make a, as_e indexable, distribute over blocks (so blockIdx.z goes 0..4 instead of 0..2)

    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x; // threadIdx.x goes 0..256; blockIdx.x = 0 for now

    // bounds check
    if (col >= (1 << V2_temp)) {
        return;
    }

    uint32_t modulus = blockIdx.z % 2; // 0 for Q0; 1 for Q1 (fix tech debt)!
    uint32_t ct_half = blockIdx.z / 2; // ct_half: 0 for a; 1 for as_e
    uint32_t coeff = blockIdx.y; // 0..POLY_LEN

    // bounds check
    if (coeff >= POLY_LEN) {
        return;
    }

    uint32_t num_rows = 1 << V1;

    uint128_t acc = 0;

    for (int row = 0; row < num_rows; ++row) {
        // todo: combine a, as_e
        const uint32_t value_acc  = first_dim->selectors[row].polys[ct_half].crt[modulus].data[coeff];
        const uint32_t db_value   = db->col[col].row[row].crt[modulus].data[coeff];

        acc += uint128_t(value_acc) * uint128_t(db_value);
    }

    acc = acc % uint128_t(modulus == 0 ? Q0 : Q1); // replace with barrett

    folding_inputs->accumulators[col].polys[ct_half].crt[modulus].data[coeff] = uint32_t(acc);
    // vastly unoptimized now;
    // need to optimize later:
    // - optimize data loading for: poly, coeff, data, row;
    // 
}