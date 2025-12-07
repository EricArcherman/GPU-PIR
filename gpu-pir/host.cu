#include "host.h"
#include "params.cuh"
#include "structures.cuh"
#include <cstdint>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include "stdio.h"

// coded by Can Elgezen and Özgün Özerk
// contributed by Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk

unsigned modpow64(unsigned a, unsigned b, unsigned mod)  // calculates (<a> ** <b>) mod <mod>
{
    unsigned res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        unsigned long long t64 = (unsigned long long)a * a;
        a = t64 % mod;
        if (b & 1)
        {
            unsigned long long r64 = (unsigned long long)a * res;
            res = r64 % mod;
        }

    }
    return res;
}

unsigned long long bitReverse(unsigned long long a, int bit_length)  // reverses the bits for twiddle factor calculation
{
    unsigned long long res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    return res;
}

std::random_device dev;  // uniformly distributed integer random number generator that produces non-deterministic random numbers
std::mt19937_64 rng(dev());  // pseudo-random generator of 64 bits with a state size of 19937 bits

void randomArray64(unsigned a[], int n, unsigned q)
{
    std::uniform_int_distribution<unsigned> randnum(0, q - 1);  // uniformly distributed random integers on the closed interval [a, b] according to discrete probability

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void fillTablePsi64(unsigned psi, unsigned q, unsigned psiinv, unsigned psiTable[], unsigned psiinvTable[], unsigned int n)  // twiddle factors computation
{
    for (int i = 0; i < n; i++)
    {
        psiTable[i] = modpow64(psi, bitReverse(i, log2(n)), q);
        psiinvTable[i] = modpow64(psiinv, bitReverse(i, log2(n)), q);
    }
}

void toMont(unsigned a[], unsigned q, unsigned int n){
    for (int i = 0; i < n; i++) {
        a[i] = ((uint64_t)(a[i]) << 32) % q;
    }
}

void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void restride_twiddles(uint32_t (&og)[POLY_LEN]) {
    uint32_t n[POLY_LEN];
    for (int tid = 0; tid < 256; tid++) {
        int twiddle_idx = 512 + 2 * tid;
        int twiddle_base = 512 + tid;
        n[twiddle_base] = og[twiddle_idx];
        n[twiddle_base + 256] = og[twiddle_idx + 1];
    }
    for (int tid = 0; tid < 256; tid++) {
        int twiddle_idx = 1024 + 4 * tid;
        int twiddle_base = 1024 + tid;
        n[twiddle_base] = og[twiddle_idx];
        n[twiddle_base + 256] = og[twiddle_idx + 1];
        n[twiddle_base + 512] = og[twiddle_idx + 2];
        n[twiddle_base + (3*256)] = og[twiddle_idx+3];
    }

    for (int i = 512; i < POLY_LEN; i++) {
        og[i] = n[i];
    }
}

void generate_twiddles(Twiddles &twiddles) {
    fillTablePsi64(Q0_ROOT, Q0, Q0_ROOT_INV, twiddles.q0_twiddles, twiddles.q0_inv_twiddles, POLY_LEN);
    fillTablePsi64(Q1_ROOT, Q1, Q1_ROOT_INV, twiddles.q1_twiddles, twiddles.q1_inv_twiddles, POLY_LEN);
    toMont(twiddles.q0_twiddles, Q0, POLY_LEN);
    toMont(twiddles.q0_inv_twiddles, Q0, POLY_LEN);
    toMont(twiddles.q1_twiddles, Q1, POLY_LEN);
    toMont(twiddles.q1_inv_twiddles, Q1, POLY_LEN);
    restride_twiddles(twiddles.q0_twiddles);
    restride_twiddles(twiddles.q0_inv_twiddles);
    restride_twiddles(twiddles.q1_twiddles);
    restride_twiddles(twiddles.q1_inv_twiddles);
}

Twiddles* gen_twiddles_gpu() {
    Twiddles cpu_twiddles;
    generate_twiddles(cpu_twiddles);

    Twiddles* gpu_twiddles;

    cudaMalloc(&gpu_twiddles, sizeof(Twiddles));
    CHECK_LAST_CUDA_ERROR();

    cudaMemcpy(gpu_twiddles, &cpu_twiddles, sizeof(Twiddles), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();
    return gpu_twiddles;
}

void rearrange(PolyHalf<uint32_t> &p) {
    PolyHalf<uint32_t> out;
    for (int thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
        int i = 0;
        for (int idx = 0; idx < POLY_LEN; idx += NUM_THREADS) {
            int z = idx + thread_idx;
            out.data[z] = p.data[thread_idx*8 + i];
            i++;
        }
    }
    for (int i = 0; i < POLY_LEN; i++) {
        p.data[i] = out.data[i];
    }
}

void ntt_to_gpu_form(Poly<uint32_t> &p, const uint64_t data[], const bool montgomery) {
    for (int i = 0; i < POLY_LEN; i++) {
        p.crt[0].data[i] = uint32_t(data[i]);
        p.crt[1].data[i] = uint32_t(data[i + POLY_LEN]);
    }
    if (montgomery) {
        toMont(p.crt[0].data, Q0, POLY_LEN);
        toMont(p.crt[1].data, Q1, POLY_LEN);
        rearrange(p.crt[0]);
        rearrange(p.crt[1]);
    }
}

void raw_to_gpu_form(Poly<uint32_t> &p, const uint64_t data[]) {
    for (int i = 0; i < POLY_LEN; i++) {
        p.crt[0].data[i] = data[i] >> 32;
        p.crt[1].data[i] = data[i] & ((((uint64_t)1 << 32) - 1));
    }
}

std::vector<std::vector<Poly<uint32_t>>> group_by(const std::vector<Poly<uint32_t>> &v, int grouping) {
    std::vector<std::vector<Poly<uint32_t>>> grouped;
    for (int i = 0; i < v.size(); i += grouping) {
        std::vector<Poly<uint32_t>> g;
        for (int j = i; j < i + grouping; j++) {
            g.push_back(v[j]);
        }
        grouped.push_back(g);
    }
    return grouped;
}

std::vector<Ciphertext> to_ciphertexts(const std::vector<Poly<uint32_t>> &v) {
    std::vector<Ciphertext> cts;
    int len_half = v.size() / 2 ;
    for (int i = 0; i < v.size()/2; i++) {
        Ciphertext c;
        c.polys[0] = v[i];
        c.polys[1] = v[i + len_half];
        cts.push_back(c);
    }
    return cts;
}

std::vector<CiphertextHalf> to_ciphertext_halfs(const std::vector<Ciphertext>& v) {
    std::vector<CiphertextHalf> bycrt;
    for (int i = 0; i < v.size(); i++) {
        CiphertextHalf c;
        c.polys[0] = v[i].polys[0].crt[0];
        c.polys[1] = v[i].polys[1].crt[0];
        bycrt.push_back(c);
    }
    for (int i = 0; i < v.size(); i++) {
        CiphertextHalf c;
        c.polys[0] = v[i].polys[0].crt[1];
        c.polys[1] = v[i].polys[1].crt[1];
        bycrt.push_back(c);
    }
    return bycrt;
}

std::vector<Poly<uint32_t>> load_polys(std::string fn, bool crt_form, bool montgomery) {
    std::vector<Poly<uint32_t>> polys;
    FILE * f;
    f = fopen(fn.c_str(), "rb");
    if (f == nullptr) {
        fprintf(stderr, "Error opening %s\n", fn.c_str());
        return polys;
    }
    if (crt_form) {
        uint64_t tmp[2 * POLY_LEN];
        while (true) {
            int read = fread(tmp, sizeof(tmp), 1, f);
            if (read <= 0) {
                break;
            }
            Poly<uint32_t> p;
            ntt_to_gpu_form(p, tmp, montgomery);
            polys.push_back(p);
        }
    } else {
        uint64_t tmp[POLY_LEN];
        while(true) {
            int read = fread(tmp, sizeof(tmp), 1, f);
            if (read <= 0) {
                break;
            }
            Poly<uint32_t> p;
            raw_to_gpu_form(p, tmp);
            polys.push_back(p);
        }
    }
    fclose(f);
    return polys;
}

std::vector<std::vector<Ciphertext>> group_to_ciphertext(const std::vector<Poly<uint32_t>> &polys, int grouping) {
    grouping *= 2;
    std::vector<std::vector<Poly<uint32_t>>> grouped = group_by(polys, grouping);
    std::vector<std::vector<Ciphertext>> m;
    for (int i = 0; i < grouped.size(); i++) {
        std::vector<Ciphertext> converted = to_ciphertexts(grouped[i]);
        m.push_back(converted);
    }
    return m;
}

std::vector<Ciphertext> flatten(const std::vector<std::vector<Ciphertext>> &input) {
    std::vector<Ciphertext> flattened;
    for (int i = 0; i < input.size(); i++) {
        flattened.push_back(input[i][0]);
    }
    return flattened;
}