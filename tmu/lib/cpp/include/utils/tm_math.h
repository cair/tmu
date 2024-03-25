//
// Created by per on 3/3/24.
//

#ifndef TUMLIBPP_TM_MATH_H
#define TUMLIBPP_TM_MATH_H
#include <vector>
#include <span>
#include <algorithm>

class TMMath {
    public:


    template<typename T>
    static T clamp(T value, T low, T high) {
        return std::max(low, std::min(value, high));
    }


    static void aRange(size_t num_samples, bool shuffle, std::vector<int>& sample_indices) {
        // Resize the vector to hold all indices
        sample_indices.resize(num_samples);

        // Fill the vector with consecutive integers starting from 0
        std::iota(sample_indices.begin(), sample_indices.end(), 0);

        // Shuffle the indices if required
        if (shuffle) {
            // Use a random device to seed a random number generator
            std::random_device rd;
            std::mt19937 g(rd());

            std::shuffle(sample_indices.begin(), sample_indices.end(), g);
        }
    }


    template<typename T, typename U>
    static const std::vector<T> elementwise_multiply(const tcb::span<const T> span1, const tcb::span<const U> span2) {
        assert(span1.size() == span2.size());

        std::vector<T> result(span1.size());

        for (std::size_t i = 0; i < span1.size(); ++i) {
            result[i] = span1[i] * span2[i];
        }

        return result;
    }


};

#endif //TUMLIBPP_TM_MATH_H
