// coded by Can Elgezen and Özgün Özerk
// contributed by Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk

#include "params.cuh"
#include <iostream>
#include <chrono>
using std::cout;  // yes we are that lazy
using std::endl;  // :)

#include "ntt_28bit.cuh"
#include "host.h"
#include "testing.cuh"

#define check 1 // set to 0 if there is no need to check for the correctness of operations

int main()
{
    const int num_blocks = 100000;
    const int POLY_SIZE = sizeof(uint32_t) * POLY_LEN;

    // cudaFuncSetAttribute(NTT, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    // CHECK_LAST_CUDA_ERROR();
    // cudaFuncSetAttribute(INTT, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    // CHECK_LAST_CUDA_ERROR();
    cudaFuncSetAttribute(NTT_R, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    CHECK_LAST_CUDA_ERROR();
    cudaFuncSetAttribute(INTT_R, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    CHECK_LAST_CUDA_ERROR();

    Twiddles* gpu_twiddles = gen_twiddles_gpu();

    unsigned* a;
    cudaMallocHost(&a, sizeof(unsigned) * POLY_LEN * num_blocks);
    randomArray64(a, POLY_LEN * num_blocks, Q0); //fill array with random numbers between 0 and q - 1
    CHECK_LAST_CUDA_ERROR();

    // fill first block with deterministic value
    std::vector<Poly> test = load_polys("./test_data/raw_poly.dat", false, false);
    for (int i = 0; i < POLY_LEN; i++) {
        uint64_t c = ((uint64_t)(test[0].crt[0].data[i]) << 32) + test[0].crt[1].data[i];
        a[i] = c % Q0;
    }

    unsigned* res_a;
    cudaMallocHost(&res_a, sizeof(unsigned) * POLY_LEN * num_blocks);
    CHECK_LAST_CUDA_ERROR();

    unsigned* d_a;
    cudaMalloc(&d_a, POLY_SIZE * num_blocks);
    CHECK_LAST_CUDA_ERROR();

    cudaMemcpy(d_a, a, POLY_SIZE * num_blocks, cudaMemcpyHostToDevice);
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
    NTT_R<<<num_blocks/(NTTS_PER_BLOCK * REUSE_TEST), THREAD_DIMENSIONS, SHARED_MEM_PER_BLOCK, 0>>>(d_a, Q0, Q0_MONT, gpu_twiddles->q0_twiddles);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "NTT: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    INTT_R<<<num_blocks/(NTTS_PER_BLOCK * REUSE_TEST), THREAD_DIMENSIONS, SHARED_MEM_PER_BLOCK, 0>>>(d_a, Q0, Q0_MONT, gpu_twiddles->q0_inv_twiddles);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << "INTT: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros" << std::endl; 
    
    cudaMemcpy(res_a, d_a, POLY_SIZE * num_blocks, cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();
    bool correct = 1;
    if (check) //check the correctness of results
    {
        for (int i = 0; i < POLY_LEN * num_blocks; i++)
        {
            // Corrected for during inverse CRT
            uint64_t renormalized = res_a[i];
            renormalized *= 268238881;
            renormalized %= Q0;
            if (a[i] != renormalized)
            {
                correct = 0;
                std::cout << "at " << i << " got " << renormalized << " instead of " << a[i] << std::endl;
                break;
            }
        }
    }

    if (correct)
        cout << "\nNTT and INTT are working correctly." << endl;
    else
        cout << "\nNTT and INTT are not working correctly." << endl;
    cudaFreeHost(a); cudaFreeHost(res_a);  
    cudaFree(d_a);
    CHECK_LAST_CUDA_ERROR();
    return 0;
}


