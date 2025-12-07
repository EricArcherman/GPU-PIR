#include "params.cuh"
#include <iostream>
#include <chrono>
using std::cout;
using std::endl;

#include "ntt_28bit.cuh"
#include "host.h"
#include "testing.cuh"

#define check 1 // set to 0 if there is no need to check for the correctness of operations


// to compile: nvcc -arch=sm_86 -cudart static --machine 64 -use_fast_math -O2 barrett_test.cu host.cu -o barrett_test -lcudadevrt
// to run: ./barrett_test
// to grant permissions to a new file: chmod a+rw barrett_test.cu

int main()
{
    const int num_blocks = 1;

    const int num_threads= 256 * num_blocks; // choose 256 threads per block

    uint64_t* a;
    cudaMallocHost(&a, sizeof(uint64_t) * num_threads);
    CHECK_LAST_CUDA_ERROR();

    uint64_t* b;
    cudaMallocHost(&b, sizeof(uint64_t) * num_threads);
    CHECK_LAST_CUDA_ERROR();
    
    for (int i = 0; i < num_threads; i++)
    {
        a[i] = (uint64_t) rand() * (uint64_t) rand();
        b[i] = (uint64_t) rand() * (uint64_t) rand();
    }

    uint64_t* res_a;
    cudaMallocHost(&res_a, sizeof(uint64_t) * num_threads);
    CHECK_LAST_CUDA_ERROR();

    for (int i = 0; i < num_threads; i++)
        res_a[i] = 0;

    uint64_t* d_a;
    cudaMalloc(&d_a, sizeof(uint64_t) * num_threads);
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_a, a, sizeof(uint64_t) * num_threads, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    uint64_t* d_b;
    cudaMalloc(&d_b, sizeof(uint64_t) * num_threads);
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_b, b, sizeof(uint64_t) * num_threads, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    uint64_t* d_res_a;
    cudaMalloc(&d_res_a, sizeof(uint64_t) * num_threads);
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_res_a, res_a, sizeof(uint64_t) * num_threads, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    /****************************************************************
    BEGIN
    Kernel Calls
    */

    auto start = std::chrono::high_resolution_clock::now();
    barrett_raw_u64<<<num_blocks, num_threads, 0, 0>>>(d_a, d_b, d_res_a, const_ratio_1, MODULUS);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Barrett: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros" << std::endl;

    cudaMemcpy(res_a, d_res_a, sizeof(uint64_t) * num_threads, cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();

    bool correct = 1;
    if (check) //check the correctness of results
    {
        for (int i = 0; i < num_threads; i++)
        {
            uint64_t expected = barrett_raw_u64(a[i] + b[i], const_ratio_1, MODULUS);
            if (res_a[i] != expected)
            {
                correct = 0;
                std::cout << "at " << i << " got " << res_a[i] << " instead of " << expected << std::endl;
                break;
            }
        }
    }

    if (correct)
        cout << "\nBarrett is working correctly." << endl;
    else
        cout << "\nBarrett is not working correctly." << endl;
    CHECK_LAST_CUDA_ERROR();
    return 0;
}