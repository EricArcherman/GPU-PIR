#pragma once
#include "params.cuh"
#include <assert.h>
#include <cuda/std/array>
// #include <stdexcept>

// To coerce cuda to use registers instead of memory for an array with constant indexing
typedef struct Regs {
    uint32_t r0;
    uint32_t r1;
#if (NUM_THREADS <= 512)
    uint32_t r2;
    uint32_t r3;
#endif
#if (NUM_THREADS <= 256)
    uint32_t r4;
    uint32_t r5;
    uint32_t r6;
    uint32_t r7;
#endif
#if (NUM_THREADS <= 128)
    uint32_t r8;
    uint32_t r9;
    uint32_t r10;
    uint32_t r11;
    uint32_t r12;
    uint32_t r13;
    uint32_t r14;
    uint32_t r15;
#endif
// this is to encourage register use with constant indexing and should all be inlined away, allowing nicer array-like code
    __device__ __forceinline__ uint32_t& operator[](int i) {
        if (i == 0) {
            return r0;
        } else if (i == 1) {
            return r1;
#if (NUM_THREADS <= 512)
        } else if (i == 2) {
            return r2;
        } else if (i == 3) {
            return r3;
#endif
#if (NUM_THREADS <= 256)
        } else if (i == 4) {
            return r4;
        } else if (i == 5) {
            return r5;
        } else if (i == 6) {
            return r6;
        } else if (i == 7) {
            return r7;
#endif
#if (NUM_THREADS <= 128)
        } else if (i == 8) {
            return r8;
        } else if (i == 9) {
            return r9;
        } else if (i == 10) {
            return r10;
        } else if (i == 11) {
            return r11;
        } else if (i == 12) {
            return r12;
        } else if (i == 13) {
            return r13;
        } else if (i == 14) {
            return r14;
        } else if (i == 15) {
            return r15;
#endif
        } else {
            // Destined to cause problems
            assert(false);
            // throw std::invalid_argument("Out of bounds");
        }
    }
} Regs;

typedef struct Regs64 {
    uint64_t r0;
    uint64_t r1;
#if (NUM_THREADS <= 512)
    uint64_t r2;
    uint64_t r3;
#endif
#if (NUM_THREADS <= 256)
    uint64_t r4;
    uint64_t r5;
    uint64_t r6;
    uint64_t r7;
#endif
#if (NUM_THREADS <= 128)
    uint64_t r8;
    uint64_t r9;
    uint64_t r10;
    uint64_t r11;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
#endif
// this is to encourage register use with constant indexing and should all be inlined away, allowing nicer array-like code
    __device__ __forceinline__ uint64_t& operator[](int i) {
        if (i == 0) {
            return r0;
        } else if (i == 1) {
            return r1;
#if (NUM_THREADS <= 512)
        } else if (i == 2) {
            return r2;
        } else if (i == 3) {
            return r3;
#endif
#if (NUM_THREADS <= 256)
        } else if (i == 4) {
            return r4;
        } else if (i == 5) {
            return r5;
        } else if (i == 6) {
            return r6;
        } else if (i == 7) {
            return r7;
#endif
#if (NUM_THREADS <= 128)
        } else if (i == 8) {
            return r8;
        } else if (i == 9) {
            return r9;
        } else if (i == 10) {
            return r10;
        } else if (i == 11) {
            return r11;
        } else if (i == 12) {
            return r12;
        } else if (i == 13) {
            return r13;
        } else if (i == 14) {
            return r14;
        } else if (i == 15) {
            return r15;
#endif
        } else {
            // Destined to cause problems
            assert(false);
            // throw std::invalid_argument("Out of bounds");
        }
    }
} Regs64;

template<typename T>
typedef struct alignas(POLY_LEN) PolyHalf {
    T data[POLY_LEN];
} PolyHalf;

template<typename T>
typedef struct alignas(POLY_LEN) Poly {
    PolyHalf<T> crt[2]; // indexable by modulus
} Poly;

typedef struct alignas(POLY_LEN) Ciphertext {
    Poly<uint32_t> polys[2]; // polys[0] for 'a', polys[1] for 'as_e'
} Ciphertext;

typedef struct alignas(POLY_LEN) CiphertextHalf {
    PolyHalf<uint32_t> polys[2]; // data[0] for 'a', data[1] for 'as_e'
} CiphertextHalf;

// To be removed in optimized version
typedef struct alignas(POLY_LEN) QueryScratch {
    Ciphertext expansion_scratch[V1];
} QueryScratch;

typedef struct alignas(POLY_LEN) PublicParameters {
    CiphertextHalf left_expansion[2][T_EXP_LEFT];
    CiphertextHalf right_expansion[2][T_EXP_RIGHT];
    Ciphertext conversion[2*T_CONV];
} PublicParameters;

typedef struct alignas(POLY_LEN) FoldingQueryStorage {
    Ciphertext accumulators[BATCH_SIZE][1 << V2];
    CiphertextHalf v_folding[V2][2][2*T_GSW];
    CiphertextHalf v_folding_neg[V2][2][2*T_GSW];
} FoldingQueryStorage;

typedef struct alignas(POLY_LEN) QueryStorage {
    Ciphertext query;
    PublicParameters public_parameters;
    QueryScratch scratch;
    FoldingQueryStorage folding_query;
} QueryStorage;

typedef struct alignas(POLY_LEN) FoldingInputs {
    Ciphertext accumulators[1 << V2_temp]; // tech debt: should not be V2_temp; accumulators are the accumulated and reduced dot products of selector ciphertexts and database
} FoldingInputs; // batching query processing for db_mul would be faster (for later)

typedef struct alignas(POLY_LEN) FirstDim {
    Ciphertext selectors[1 << V1]; // FirstDim is an array of Ciphertexts that do the selecting
} FirstDim;

template<typename T>
class alignas(POLY_LEN) DatabaseCol {
    public:
    T row[1 << V1];
};

template<typename T>
class alignas(POLY_LEN) Database { // OldDatabase: Database<Poly<uint32_t>>
    public:
    DatabaseCol<T> col[1 << V2_temp]; // tech debt: should not be V2_temp (like unify the 4 in dbmul with 2 in folding)
};

template<typename T>
class alignas(POLY_LEN) FastDB {
    public:
    Poly<Database<uint32_t>> db;
};

typedef struct alignas(POLY_LEN) Twiddles {
    uint32_t q0_twiddles[POLY_LEN];
    uint32_t q1_twiddles[POLY_LEN];
    uint32_t q0_inv_twiddles[POLY_LEN];
    uint32_t q1_inv_twiddles[POLY_LEN];
} Twiddles;