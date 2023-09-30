#include "fast_rand.h"

// IF MSVC
#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_rdrand32_step)
#else
#include <immintrin.h>
#endif

// Get a random value using RDRAND.
uint32_t rdrand_fast() {
    unsigned int val;
    while (!_rdrand32_step(&val)) {};  // Keep trying until successful
    return val;
}

