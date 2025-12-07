#pragma once
#include "params.cuh"
#include "structures.cuh"
#include <vector>
#include <string>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line);

Twiddles* gen_twiddles_gpu();
void randomArray64(unsigned a[], int n, unsigned q);
std::vector<Poly<uint32_t>> load_polys(std::string fn, bool crt_form, bool montgomery);
std::vector<std::vector<Ciphertext>> group_to_ciphertext(const std::vector<Poly<uint32_t>> &polys, int grouping);
std::vector<Ciphertext> flatten(const std::vector<std::vector<Ciphertext>> &input);