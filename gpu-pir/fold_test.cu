#include "params.cuh"
#include "structures.cuh"
#include "testing.cuh"
#include "host.h"
#include "folding.cuh"
#include <vector>
#include <iostream>
#include <chrono>

// to compile: nvcc -arch=sm_86 -cudart static --machine 64 -use_fast_math -O2 fold_test.cu host.cu -o fold_test -lcudadevrt
// to run: ./fold_test
// to grant permissions to a new file: chmod a+rw fold_test.cu

void test_test() {
    TEST_TEST<<<1, NUM_THREADS, 0, 0>>>();
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
}

// runs num_per blocks and does all of the folds for a power of 2 in parallel
void parallel_fold(int num_blocks, const Twiddles* gpu_twiddles, FoldingQueryStorage* query_on_gpu) {
    int num_per = 1 << V2;
    for (int cur_dim = 0; cur_dim < V2; cur_dim++) {
        num_per = num_per / 2;
        dim3 block_dims(num_blocks/NTTS_PER_BLOCK, num_per, 1);
        FOLD1<<<block_dims, THREAD_DIMENSIONS, SHARED_MEM_PER_BLOCK, 0>>>(gpu_twiddles, query_on_gpu, cur_dim);
    }
}

void test_folding() {
    const bool PARALLEL = true;
    const int num_blocks = 1000;
    std::vector<Ciphertext> test_input = flatten(group_to_ciphertext(load_polys("./test_data/client_fold_input.dat", false, false), 1));
    std::vector<std::vector<Ciphertext>> test_keys = group_to_ciphertext(load_polys("./test_data/client_folding_keys.dat", true, true), 2*T_GSW);
    std::vector<std::vector<Ciphertext>> test_keys_neg = group_to_ciphertext(load_polys("./test_data/client_folding_keys_neg.dat", true, true), 2*T_GSW);
    std::vector<Ciphertext> test_output = flatten(group_to_ciphertext(load_polys("./test_data/client_fold_output.dat", false, false),1));
    assert(test_keys.size() == V2);
    assert(test_keys[0].size() == (2 * T_GSW));
    assert(test_input.size() == (1 << V2));

    cudaFuncSetAttribute(FOLD_CIPHERTEXTS, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    CHECK_LAST_CUDA_ERROR();

    const Twiddles* gpu_twiddles = gen_twiddles_gpu();

    FoldingQueryStorage query;
    for (int i = 0; i < test_input.size(); i++) {
        query.accumulators[0][i] = test_input[i];
        // printf("input-host %u\n", test_hash(query.accumulators[0][i].polys[0].crt[0].data));
    }
    for (int i = 0; i < V2; i++) {
        for (int j = 0; j < (2 * T_GSW); j++) {
            query.v_folding[i][0][j].polys[0] = test_keys[i][j].polys[0].crt[0];
            query.v_folding[i][1][j].polys[0] = test_keys[i][j].polys[0].crt[1];
            query.v_folding[i][0][j].polys[1] = test_keys[i][j].polys[1].crt[0];
            query.v_folding[i][1][j].polys[1] = test_keys[i][j].polys[1].crt[1];

            query.v_folding_neg[i][0][j].polys[0] = test_keys_neg[i][j].polys[0].crt[0];
            query.v_folding_neg[i][1][j].polys[0] = test_keys_neg[i][j].polys[0].crt[1];
            query.v_folding_neg[i][0][j].polys[1] = test_keys_neg[i][j].polys[1].crt[0];
            query.v_folding_neg[i][1][j].polys[1] = test_keys_neg[i][j].polys[1].crt[1];
        }
    }

    FoldingQueryStorage* query_on_gpu;
    cudaMalloc(&query_on_gpu, sizeof(FoldingQueryStorage) * num_blocks);
    CHECK_LAST_CUDA_ERROR();
    for (int i = 0; i < num_blocks; i++) {
        cudaMemcpy(&query_on_gpu[i], &query, sizeof(FoldingQueryStorage), cudaMemcpyHostToDevice);
    }
    auto start = std::chrono::high_resolution_clock::now();
    if (PARALLEL) {
        parallel_fold(num_blocks, gpu_twiddles, query_on_gpu);
    } else {
        FOLD_CIPHERTEXTS<<<num_blocks/NTTS_PER_BLOCK, THREAD_DIMENSIONS, SHARED_MEM_PER_BLOCK, 0>>>(gpu_twiddles, query_on_gpu);
    }
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Fold: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "micros" << std::endl;

    for (int i = 0; i < num_blocks; i++) {
        cudaMemcpy(&query.accumulators[0][0], &query_on_gpu[i].accumulators[0][0], sizeof(Ciphertext), cudaMemcpyDeviceToHost);
        CHECK_LAST_CUDA_ERROR();
        if (memcmp(&query.accumulators[0][0], &test_output[0], sizeof(Ciphertext)) == 0) {
           
        } else {
            printf("Fail\n");
            return;
        }
    }
    printf("Pass\n");
}

int main() {
    test_folding(); // 13438micros, 313240 executions of fold_ciphertexts
} // graph time per iteration for different parameters (database size)