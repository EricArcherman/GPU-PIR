#pragma once
#include <stdint.h>
// Change based on GPU
#define TOTAL_THREADS 1536
const int WARP_SIZE = 32;

// #include<algorithm>
#define MAX(x,y) ((x) > (y) ? x : y)
#define NUM_THREADS 256
#define NTTS_PER_BLOCK 2
#define REUSE_TEST 25
#define MAX_THREADS_PER_BLOCK (NUM_THREADS * NTTS_PER_BLOCK)
#define MIN_BLOCKS_PER_MP (TOTAL_THREADS / MAX_THREADS_PER_BLOCK)
const dim3 THREAD_DIMENSIONS(NUM_THREADS, NTTS_PER_BLOCK, 1);

const uint32_t POLY_LEN = 2048;
const uint32_t LEN_LOG2 = 11;
static_assert((1 << LEN_LOG2) == POLY_LEN);

const int SHARED_MEM_PER_BLOCK = (1 + NTTS_PER_BLOCK) * POLY_LEN * sizeof(uint32_t);

const int NUM_PAIRS = POLY_LEN / NUM_THREADS / 2;
const int NUM_RADIX = 3;
static_assert((1 << NUM_RADIX) == NUM_PAIRS * 2);
const uint32_t Q0 = 268369921; // moduli[0] (lo) (tech debt: should be indexable)
const uint32_t Q0_ROOT = 66687;
const uint32_t Q0_ROOT_INV = 181947619;
const uint32_t Q1 = 249561089; // moduli[1] (hi) (tech debt: should be indexable), but IDK how?! bc then we need to pass to gpu
const uint32_t Q1_ROOT = 158221;
const uint32_t Q1_ROOT_INV = 88293783;
const uint32_t MODULI[] = {Q0, Q1};

// Bezout coefficient * INTT scaling * Montgomery factor
const uint32_t CRT_INV_FOR0 = 88833249;
const uint32_t CRT_INV_FOR1 = 166953760;
const uint32_t Q0_MONT = 4026597377;
const uint32_t Q1_MONT = 4045406209;
const uint32_t Q0_RSQUARED = 234877184;
const uint32_t Q1_RSQUARED = 148369395;
const uint64_t MODULUS = (uint64_t)Q0 * Q1;
const uint32_t MOD_BITS = 56;
const uint32_t PT_MODULUS = 256;
const int PT_BYTES = 1;

const uint32_t Q0_BARRETT = (uint32_t)(((uint64_t)(1) << (32 + MOD_BITS/2)) / Q0);
const uint32_t Q1_BARRETT = (uint32_t)(((uint64_t)(1) << (32 + MOD_BITS/2)) / Q1);
const uint32_t MODULI_BARRETT[] = {Q0_BARRETT, Q1_BARRETT};

const int N = 1; // implementing all nxn matrices in the paper as single element (1x1 matrix).
const int CT_ROWS = 2 * N;
const int CT_COLS = 1 * N;
const int PT_ROWS = 1 * N;
const int PT_COLS = 1 * N;

// Parameter that can change

// const int T_CONV = 4;
// const int T_GSW = 7;
// const int T_EXP_LEFT = 16;
// const int T_EXP_RIGHT = 56;
// const int V1 = 10;
// const int V2 = 4;

// fast expansion test params (V1 is log2(num_rows) and V2/V2_temp is log2(num_cols) for folding/db_mul respectively)
const int T_CONV = 4;
const int T_GSW = 8;
const int T_EXP_LEFT = 8;
const int T_EXP_RIGHT = 8;
const int V1 = 6;
const int V2 = 2; // note(eric): your db_mul test code has shape 6x4 (i.e., V2=4). fix later [good litmus test].
const int V2_temp = 4; // temp variable for testing

const int DB_COLS = 1 << V1;
const int DB_ROWS = 1 << V2_temp;

// probably don't change
const int BATCH_SIZE = 4;
const int BATCH_ROOT = 2;

// Barrett constants
const uint64_t BARRETT_CR_1[] = {68736257792ULL, 73916747789ULL}; // floor(2^64 / Q1), floor(2^64 % Q1)

const int COEFFS_PER_THREAD = POLY_LEN / NUM_THREADS;
const int TOTAL_TWIDDLES = 4; // forward and inverse for each modulus