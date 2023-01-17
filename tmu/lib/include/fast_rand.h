//
// Created by Yasser Phoulady on 2019-04-01.
//

#ifndef C_BITWISE_TSETLIN_MACHINE_FAST_RAND_H
#define C_BITWISE_TSETLIN_MACHINE_FAST_RAND_H

#include <stdint.h>

static uint64_t const multiplier = 6364136223846793005u;
static uint64_t       mcg_state  = 0xcafef00dd15ea5e5u;

inline static uint32_t pcg32_fast() {
    uint64_t x = mcg_state;
    unsigned int count = (unsigned int) (x >> 61);	// 61 = 64 - 3

    mcg_state = x * multiplier;
    return (uint32_t) ((x ^ x >> 22) >> (22 + count));	// 22 = 32 - 3 - 7
}

//static void pcg32_fast_init(uint64_t seed) {
//    mcg_state = 2 * seed + 1;
//    pcg32_fast();
//}

#define FAST_RAND_MAX UINT32_MAX
#define fast_rand() pcg32_fast()

// Box–Muller transform
inline static int normal(double mean, double variance) {
    double u1 = (double) (fast_rand() + 1) / ((double) FAST_RAND_MAX + 1), u2 = (double) fast_rand() / FAST_RAND_MAX; // u1 in (0, 1] and u2 in [0, 1]
    double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
    return (int) round(mean + sqrt(variance) * n1);
}

inline static int binomial(int n, double p) {
    return normal(n * p, n * p * (1 - p));
}

// Knuth's random Poisson-distributed number
inline static int poisson(double lambda) {
    int k = 0;
    double l = exp(-lambda), p = 1;
    while (p > l) {
        ++k;
        p *= (double) fast_rand() / FAST_RAND_MAX;
    }
    return k - 1;
}

#endif //C_BITWISE_TSETLIN_MACHINE_FAST_RAND_H
