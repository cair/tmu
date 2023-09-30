#include "fast_rand.h"

static uint64_t const multiplier = 6364136223846793005u;
static uint64_t mcg_state = 0xcafef00dd15ea5e5u;

void pcg32_seed(uint64_t seed) {
    mcg_state = seed;
}

uint32_t pcg32_fast() {
    uint64_t x = mcg_state;
    unsigned int count = (unsigned int)(x >> 61);	// 61 = 64 - 3
    mcg_state = x * multiplier;
    return (uint32_t)((x ^ x >> 22) >> (22 + count));	// 22 = 32 - 3 - 7
}

