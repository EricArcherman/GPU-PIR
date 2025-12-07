#include "params.cuh"
#include <iostream>
#include <chrono>
using std::cout;
using std::endl;

#include "ntt_28bit.cuh"
#include "host.h"
#include "testing.cuh"
#include "automorph.cuh"

#define check 1 // set to 0 if there is no need to check for the correctness of operations

void cpu_automorph_reference(uint64_t* input, uint64_t* output, int t, int poly_len, int reps) { // reps just repeats the same calculation a bunch of times (no chaining)
    for (int j = 0; j < reps; j++) {
        for (int i = 0; i < poly_len; i++) {
            int new_idx = ((uint64_t)i * t) % poly_len;
            int sign = (((uint64_t)i * t) / poly_len) % 2;
            output[new_idx] = sign ? (MODULUS - input[i]) : input[i];
        }
    }
}

int main() {
    // const int num_blocks = 8; (we don't actually have 8 blocks)
    // const int num_elements = 256 * num_blocks;
    const int num_elements = 2048; // numelements is the number of coefficients in the polynomial

    uint64_t* poly_coefficients; // pointer to where the coefficients of the polynomial are stored
    cudaMallocHost(&poly_coefficients, sizeof(uint64_t) * num_elements); // allocate sizeof(uint64_t) * num_elements of memory to store the polynomial coefficients on the host device (CPU)
    CHECK_LAST_CUDA_ERROR();

    uint64_t* t; // pointer to where the automorphism parameter is stored
    cudaMallocHost(&t, sizeof(uint64_t)); // allocate sizeof(uint64_t) of memory to store the automorphism parameter on the host device (CPU)
    CHECK_LAST_CUDA_ERROR();
    *t = 3; // automorphism parameter (f(x) -> f(x^3) mod x^MODULUS - 1) (3 was arbitrarily chosen)

    for (int i = 0; i < num_elements; i++) {
        poly_coefficients[i] = (uint64_t) i; // fill the polynomial coefficients with the values from 0 to num_elements - 1 (this helps for testing, random values also work)
    }

    uint64_t* res_poly_coefficients; // pointer to where the resulting polynomial coefficients will be stored
    cudaMallocHost(&res_poly_coefficients, sizeof(uint64_t) * num_elements); // allocate sizeof(uint64_t) * num_elements of memory to store the resultant polynomial's coefficients on the host device (CPU)
    CHECK_LAST_CUDA_ERROR();

    for (int i = 0; i < num_elements; i++) {
        res_poly_coefficients[i] = (uint64_t) 0; // initialize to 0
    }

    uint64_t* d_poly_coefficients; // device pointer
    cudaMalloc(&d_poly_coefficients, sizeof(uint64_t) * num_elements); // cudaMalloc allocates memory on the device (GPU)
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_poly_coefficients, poly_coefficients, sizeof(uint64_t) * num_elements, cudaMemcpyHostToDevice); // cudaMemcpy COPIES from host to device (destination pointer, source pointer, number of bytes to copy, direction of copy - in this case from Host to Device)
    CHECK_LAST_CUDA_ERROR();

    uint64_t* d_t; // device pointer to automorphism parameter, the (d_) prefix generally stands for device
    cudaMalloc(&d_t, sizeof(uint64_t));
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_t, t, sizeof(uint64_t), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    uint64_t* d_res_poly_coefficients;
    cudaMalloc(&d_res_poly_coefficients, sizeof(uint64_t) * num_elements);
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(d_res_poly_coefficients, res_poly_coefficients, sizeof(uint64_t) * num_elements, cudaMemcpyHostToDevice);
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
    const int reps = 1 << 20;

    auto start = std::chrono::high_resolution_clock::now();
    const uint32_t SHARED_MEMORY_SIZE = sizeof(uint64_t) * POLY_LEN; // shared memory size need only be large enough to hold the polynomial coefficients
    automorph<<<1, 256, SHARED_MEMORY_SIZE, 0>>>(d_poly_coefficients, d_res_poly_coefficients, d_t, reps); // runs the automorph kernel on 1 block, 256 threads per block, shared memory size of SHARED_MEMORY_SIZE bytes of dynamic shared memory, stream 0 (default stream)
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Automorph GPU: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros\n" << std::endl;

    cudaMemcpy(res_poly_coefficients, d_res_poly_coefficients, sizeof(uint64_t) * num_elements, cudaMemcpyDeviceToHost); // copies processed data back from device to host
    CHECK_LAST_CUDA_ERROR();

    bool correct = 1;
    if (check)
    {
        uint64_t* cpu_res_poly_coefficients;
        cudaMallocHost(&cpu_res_poly_coefficients, sizeof(uint64_t) * num_elements);
        CHECK_LAST_CUDA_ERROR();
        auto start = std::chrono::high_resolution_clock::now();
        cpu_automorph_reference(poly_coefficients, cpu_res_poly_coefficients, *t, num_elements, reps);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Automorph CPU: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros\n" << std::endl;
        for (int i = 0; i < num_elements; i++) {
            if (res_poly_coefficients[i] != cpu_res_poly_coefficients[i]) {
                correct = 0;
                std::cout << "at " << i << " got " << res_poly_coefficients[i] << " instead of " << cpu_res_poly_coefficients[i] << std::endl;
                break;
            }
        }
    }

    if (correct)
        cout << "Automorph GPU matches CPU." << endl;
    else
        cout << "Automorph GPU does not match CPU." << endl;
    CHECK_LAST_CUDA_ERROR();

    // std::cout << "Original coefficients:" << std::endl;
    // for (int i = 0; i < num_elements; i++) {
    //     std::cout << poly_coefficients[i] << " ";
    // }
    // std::cout << std::endl << "Result coefficients:" << std::endl;
    // for (int i = 0; i < num_elements; i++) {
    //     std::cout << res_poly_coefficients[i] << " ";
    // }
    // std::cout << std::endl;
    return 0;
}