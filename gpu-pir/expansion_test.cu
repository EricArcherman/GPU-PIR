#include "params.cuh"
#include <iostream>
#include <chrono>
using std::cout;
using std::endl;

#include "ntt_28bit.cuh"
#include "host.h"
#include "testing.cuh"
#include "automorph.cuh"
#include "structures.cuh"
#include "automorph.cuh"
#include "gadget.cuh"
#include "expansion.cuh"
#include <cuda/std/array>

#define check 1 // set to 0 if there is no need to check for the correctness of operations

void cpu_coefficient_expansion_reference(uint32_t* v, PublicParameters* public_parameters, uint32_t* v_neg1) {
    
}

// runs num_per blocks and does all of the folds for a power of 2 in parallel
void parallel_expand(int num_blocks, uint32_t* v, PublicParameters* public_parameters, uint32_t* v_neg1) {
    int num_per = 1;
    for (uint32_t r = 0; r < g; r++) {
        num_per = num_per / 2;
        dim3 block_dims(num_blocks/NTTS_PER_BLOCK, num_per, 1);
        coefficient_expansion_r<<<block_dims, THREAD_DIMENSIONS, SHARED_MEM_PER_BLOCK, 0>>>(v, public_parameters, v_neg1, r);
    }
}

int main() {
    uint32_t* v; // CPU pointer
    cudaMallocHost(&v, sizeof(CiphertextNTT) * (1<<(V1 + 1)));
    CHECK_LAST_CUDA_ERROR();

    PublicParameters* public_parameters;
    cudaMallocHost(&public_parameters, sizeof(PublicParameters));
    CHECK_LAST_CUDA_ERROR();

    uint32_t* v_neg1;
    cudaMallocHost(&v_neg1, sizeof(PolyNTT) * LEN_LOG2);
    CHECK_LAST_CUDA_ERROR();

    // for (int i = 0; i < num_elements; i++) {
    //     v[i] = (uint64_t) i; // fill the values of the inputs
    // }

    uint32_t* d_v; // device pointer
    cudaMalloc(&d_v, sizeof(CiphertextNTT) * (1<<(V1 + 1)));
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_v, v, sizeof(CiphertextNTT) * (1<<(V1 + 1)), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    PublicParameters* d_public_parameters;
    cudaMalloc(&d_public_parameters, sizeof(PublicParameters));
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_public_parameters, public_parameters, sizeof(PublicParameters), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    uint32_t* d_v_neg1;
    cudaMalloc(&d_v_neg1, sizeof(PolyNTT) * LEN_LOG2);
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_v_neg1, v_neg1, sizeof(PolyNTT) * LEN_LOG2, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    this means we have finished the data movement from host to device, and can start the kernel calls on the GPU
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    // const uint64_t g = ceil(log2(static_cast<double>(T_GSW * V2 + (1 << V1))));
    // const uint64_t stop_round = ceil(log2(static_cast<double>(T_GSW * V2)));
    // const uint64_t max_bits_to_gen_right = T_GSW * V2;

    auto start = std::chrono::high_resolution_clock::now();
    
    parallel_expand(1<<(V1 - 1), d_v, d_public_parameters, d_v_neg1); // check the first argument

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Coefficient Expansion GPU: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros\n" << std::endl;

    cudaMemcpy(v, d_v, sizeof(CiphertextNTT) * (1<<(V1 + 1)), cudaMemcpyDeviceToHost); // copies processed data back from device to host
    CHECK_LAST_CUDA_ERROR();

    bool correct = 1;
    if (check)
    {
        uint64_t* cpu_v;
        cudaMallocHost(&cpu_v, sizeof(uint64_t) * 2 * 1 * POLY_LEN * 2 * (1<<(V1 + 1)));
        CHECK_LAST_CUDA_ERROR();
        auto start = std::chrono::high_resolution_clock::now();
        cpu_coefficient_expansion_reference(v, public_parameters, v_neg1);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Coefficient Expansion CPU: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros\n" << std::endl;
        for (int i = 0; i < 2 * 1 * POLY_LEN * 2 * (1<<(V1 + 1)); i++) {
            if (v[i] != cpu_v[i]) {
                correct = 0;
                std::cout << "at " << i << " got " << v[i] << " instead of " << cpu_v[i] << std::endl;
                break;
            }
        }
    }

    if (correct)
        cout << "Coefficient expansion GPU matches CPU." << endl;
    else
        cout << "Coefficient expansion GPU does not match CPU." << endl;
    CHECK_LAST_CUDA_ERROR();

    return 0;
}