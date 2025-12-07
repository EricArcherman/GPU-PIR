#include "params.cuh"
#include <stdio.h>

constexpr uint32_t barrett_reduce(uint64_t value, uint32_t mu, uint32_t modulus) {
    uint32_t hi = value >> (MOD_BITS/2);
    // __umulhi is a 64-bit multiplication that returns the high 32 bits of the product, which does not exist on CPU
    uint32_t q = (((uint64_t)(hi) * (uint64_t)(mu)) >> 32) + hi;
    uint32_t r = value - q * modulus;
    if (r >= modulus) {
        r -= modulus;
    }
    return r;
}

const uint64_t test_value = 1000000000000000000;

static_assert(barrett_reduce(test_value, Q0_BARRETT, Q0) == test_value % Q0);
static_assert(barrett_reduce(test_value, Q1_BARRETT, Q1) == test_value % Q1);

int main() {
    printf("Compiled successfully");
    return 0;
}