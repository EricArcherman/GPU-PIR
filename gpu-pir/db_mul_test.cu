#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <chrono> // For timing

#include "params.cuh"
#include "structures.cuh"
#include "db_mul.cuh"
#include "host.h"

// to compile: nvcc -arch=sm_86 -cudart static --machine 64 -use_fast_math -O2 db_mul_test.cu host.cu -o db_mul_test -lcudadevrt
// to run: ./db_mul_test
// to grant permissions to a new file: chmod a+rw db_mul_test.cu

// precompute (short) barrett ratio
inline uint32_t precompute_short_barrett(uint32_t modulus) {
    // floor( 2^32 / modulus )
    return static_cast<uint32_t>((1ULL << 32) / modulus);
}

void load_db(const std::string& filename, Database<Poly<uint32_t>>& db) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Could not open database file: %s\n", filename.c_str());
        std::exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(&db), sizeof(Database<Poly<uint32_t>>));
    if (!in.good() && !in.eof()) {
        fprintf(stderr, "Error reading database file: %s\n", filename.c_str());
        std::exit(EXIT_FAILURE);
    }
}

void load_first_dim(const std::string& filename, FirstDim& first_dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Could not open first dimension file: %s\n", filename.c_str());
        std::exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(&first_dim), sizeof(FirstDim));
    if (!in.good() && !in.eof()) {
        fprintf(stderr, "Error reading first dimension file: %s\n", filename.c_str());
        std::exit(EXIT_FAILURE);
    }
}

void load_folding_inputs(const std::string& filename, FoldingInputs& folding_inputs) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Could not open folding inputs file: %s\n", filename.c_str());
        std::exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(&folding_inputs), sizeof(FoldingInputs));
    if (!in.good() && !in.eof()) {
        fprintf(stderr, "Error reading folding inputs file: %s\n", filename.c_str());
        std::exit(EXIT_FAILURE);
    }
}

void test_db_mul() {
    std::string data = "data";
    std::string prefix = "eric_";

    // load data from files
    std::string db_file = data + "/" + prefix + "db" + ".dat";
    std::string first_dim_file = data + "/" + prefix + "first_dim" + ".dat";
    std::string output_file = data + "/" + prefix + "folding_inputs" + ".dat";

    Database<Poly<uint32_t>>* db = new Database<Poly<uint32_t>>(); // cpp way to allocate on heap
    FirstDim* first_dim = new FirstDim();
    FoldingInputs* correct_folding_inputs = new FoldingInputs();
    FoldingInputs* computed_folding_inputs = new FoldingInputs();

    load_db(db_file, *db);
    load_first_dim(first_dim_file, *first_dim);
    load_folding_inputs(output_file, *correct_folding_inputs);

    /****************************************************************
    BEGIN
    Memory Movement
    */

    // allocate memory on GPU
    Database<Poly<uint32_t>>* d_db;
    cudaMalloc(&d_db, sizeof(Database<Poly<uint32_t>>));
    CHECK_LAST_CUDA_ERROR();

    FirstDim* d_first_dim;
    cudaMalloc(&d_first_dim, sizeof(FirstDim));
    CHECK_LAST_CUDA_ERROR();

    FoldingInputs* d_folding_inputs;
    cudaMalloc(&d_folding_inputs, sizeof(FoldingInputs));
    CHECK_LAST_CUDA_ERROR();

    // copy data to GPU
    cudaMemcpy(d_db, db, sizeof(Database<Poly<uint32_t>>), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    cudaMemcpy(d_first_dim, first_dim, sizeof(FirstDim), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    /*
    END
    Memory Movement
    ****************************************************************/

    /****************************************************************
    BEGIN
    Kernel Calls
    */

    dim3 block_dim, grid_dim;
    if (1 << V2_temp < 256) { // can go up to 1024
        block_dim = dim3(1 << V2_temp);
        grid_dim = dim3(1, POLY_LEN, 2); // 2 * 2 for Q0, Q1, a, as_e
    } else { // does not happen for our database size right now; todo: fix&stress test (fix later when necessary)
        block_dim = dim3(256);
        grid_dim = dim3((1 << V2_temp) / block_dim.x, POLY_LEN, 2);
    }

    // precompute barrett ratios
    uint32_t barrett_ratio_1 = precompute_short_barrett(Q1);
    uint32_t barrett_ratio_0 = precompute_short_barrett(Q0); // fix: tech debt (Q0, Q1, but them in an array)

    // todo: add barrett reduction

    // Timing the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    db_mul_kernel<<<grid_dim, block_dim>>>(
        d_db,
        d_first_dim,
        d_folding_inputs
    );

    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "db_mul_kernel GPU execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*
    END
    Kernel Calls
    ****************************************************************/

    /****************************************************************
    BEGIN
    Copy results from GPU to host
    */

    cudaMemcpy(computed_folding_inputs, d_folding_inputs, sizeof(FoldingInputs), cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();

    /*
    END
    Copy results from GPU to host
    ****************************************************************/

    // todo: check correctness

    bool correct = 1;
    for (int i = 0; i < (1 << V2_temp); i++) {
        for (int j = 0; j < POLY_LEN; j++) {
            if (computed_folding_inputs->accumulators[i].polys[0].crt[0].data[j] != correct_folding_inputs->accumulators[i].polys[0].crt[0].data[j]) {
                correct = 0;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[0].crt[0].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[0].crt[0].data[j] << std::endl;
                break;
            }
            if (computed_folding_inputs->accumulators[i].polys[1].crt[0].data[j] != correct_folding_inputs->accumulators[i].polys[1].crt[0].data[j]) {
                correct = 0;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[1].crt[0].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[1].crt[0].data[j] << std::endl;
                break;
            }
            if (computed_folding_inputs->accumulators[i].polys[0].crt[1].data[j] != correct_folding_inputs->accumulators[i].polys[0].crt[1].data[j]) {
                correct = 0;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[0].crt[1].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[0].crt[1].data[j] << std::endl;
                break;
            }
            if (computed_folding_inputs->accumulators[i].polys[1].crt[1].data[j] != correct_folding_inputs->accumulators[i].polys[1].crt[1].data[j]) {
                correct = 0;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[1].crt[1].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[1].crt[1].data[j] << std::endl;
                break;
            }
        }
    }

    if (correct) {
        std::cout << "db_mul is working correctly." << std::endl;
    } else {
        std::cout << "db_mul is not working correctly." << std::endl;
    }

    // Clean up heap allocations
    delete db;
    delete first_dim;
    delete correct_folding_inputs;
    delete computed_folding_inputs;
}

void cpu_db_mul_comparison() {
    std::string data = "data";
    std::string prefix = "eric_";

    // load data from files
    std::string db_file = data + "/" + prefix + "db" + ".dat";
    std::string first_dim_file = data + "/" + prefix + "first_dim" + ".dat";
    std::string output_file = data + "/" + prefix + "folding_inputs" + ".dat";

    Database<Poly<uint32_t>>* db = new Database<Poly<uint32_t>>();
    FirstDim* first_dim = new FirstDim();
    FoldingInputs* correct_folding_inputs = new FoldingInputs();
    FoldingInputs* computed_folding_inputs = new FoldingInputs();

    load_db(db_file, *db);
    load_first_dim(first_dim_file, *first_dim);
    load_folding_inputs(output_file, *correct_folding_inputs);

    // Initialize computed_folding_inputs to zero
    memset(computed_folding_inputs, 0, sizeof(FoldingInputs));

    // Timing the CPU execution
    auto start = std::chrono::high_resolution_clock::now();

    // CPU implementation of db_mul
    // Iterate over all columns
    for (uint32_t col = 0; col < (1 << V2_temp); ++col) {
        // Iterate over all coefficients
        for (uint32_t coeff = 0; coeff < POLY_LEN; ++coeff) {
            // Iterate over both moduli (0 for Q0, 1 for Q1)
            for (uint32_t modulus = 0; modulus < 2; ++modulus) {
                uint32_t num_rows = 1 << V1;
                
                // Use 128-bit arithmetic to match GPU implementation and avoid overflow
                unsigned __int128 acc_a = 0;
                unsigned __int128 acc_as_e = 0;

                // Compute dot products over all rows
                for (uint32_t row = 0; row < num_rows; ++row) {
                    const uint32_t value_a = (modulus == 0) 
                        ? first_dim->selectors[row].polys[0].crt[0].data[coeff]
                        : first_dim->selectors[row].polys[0].crt[1].data[coeff];
                    const uint32_t value_as_e = (modulus == 0)
                        ? first_dim->selectors[row].polys[1].crt[0].data[coeff]
                        : first_dim->selectors[row].polys[1].crt[1].data[coeff];
                    const uint32_t db_value = (modulus == 0)
                        ? db->col[col].row[row].crt[0].data[coeff]
                        : db->col[col].row[row].crt[1].data[coeff];

                    acc_a += static_cast<unsigned __int128>(value_a) * static_cast<unsigned __int128>(db_value);
                    acc_as_e += static_cast<unsigned __int128>(value_as_e) * static_cast<unsigned __int128>(db_value);
                }

                // Reduce modulo Q0 or Q1
                unsigned __int128 mod = (modulus == 0) ? Q0 : Q1;
                uint32_t a = static_cast<uint32_t>(acc_a % mod);
                uint32_t as_e = static_cast<uint32_t>(acc_as_e % mod);

                // Store results
                if (modulus == 0) {
                    computed_folding_inputs->accumulators[col].polys[0].crt[0].data[coeff] = a;
                    computed_folding_inputs->accumulators[col].polys[1].crt[0].data[coeff] = as_e;
                } else {
                    computed_folding_inputs->accumulators[col].polys[0].crt[1].data[coeff] = a;
                    computed_folding_inputs->accumulators[col].polys[1].crt[1].data[coeff] = as_e;
                }
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double milliseconds = duration.count() / 1000.0;
    std::cout << "db_mul CPU execution time: " << milliseconds << " ms" << std::endl;

    // Check correctness
    bool correct = true;
    for (int i = 0; i < (1 << V2_temp); i++) {
        for (int j = 0; j < POLY_LEN; j++) {
            if (computed_folding_inputs->accumulators[i].polys[0].crt[0].data[j] != correct_folding_inputs->accumulators[i].polys[0].crt[0].data[j]) {
                correct = false;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[0].crt[0].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[0].crt[0].data[j] << std::endl;
                break;
            }
            if (computed_folding_inputs->accumulators[i].polys[1].crt[0].data[j] != correct_folding_inputs->accumulators[i].polys[1].crt[0].data[j]) {
                correct = false;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[1].crt[0].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[1].crt[0].data[j] << std::endl;
                break;
            }
            if (computed_folding_inputs->accumulators[i].polys[0].crt[1].data[j] != correct_folding_inputs->accumulators[i].polys[0].crt[1].data[j]) {
                correct = false;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[0].crt[1].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[0].crt[1].data[j] << std::endl;
                break;
            }
            if (computed_folding_inputs->accumulators[i].polys[1].crt[1].data[j] != correct_folding_inputs->accumulators[i].polys[1].crt[1].data[j]) {
                correct = false;
                std::cout << "at " << i << ", " << j << " got " << computed_folding_inputs->accumulators[i].polys[1].crt[1].data[j] << " instead of " << correct_folding_inputs->accumulators[i].polys[1].crt[1].data[j] << std::endl;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        std::cout << "CPU db_mul is working correctly." << std::endl;
    } else {
        std::cout << "CPU db_mul is not working correctly." << std::endl;
    }

    // Clean up heap allocations
    delete db;
    delete first_dim;
    delete correct_folding_inputs;
    delete computed_folding_inputs;
}

int main() {
    test_db_mul();
    cpu_db_mul_comparison(); // this is lowk bloat
    return 0;
}