//
// Created by per on 3/2/24.
//

#ifndef TUMLIBPP_TM_WEIGHT_BANK_H
#define TUMLIBPP_TM_WEIGHT_BANK_H

#include <span>
#include "tm_memory.h"
#include "tm_clause_dense.h"
#include <assert.h>
#include <iostream>
extern "C" {
    #include "WeightBank.h"
}

class TMWeightBankPresets {



public:
    static std::span<int32_t> positive_one_negative_minus_one(std::span<int32_t>& weights, std::size_t number_of_clauses) {
        assert (weights.size() == number_of_clauses);

        // Fill weights with 1 and -1
        for (int i = 0; i < number_of_clauses / 2; i++) {
            weights[i] = 1;
            weights[i + (number_of_clauses / 2)] = -1;
        }

        return weights;
    }

};


template<class T>
class TMWeightBank: public std::enable_shared_from_this<TMWeightBank<T>> {



public:
    std::span<int32_t> weights;

    TMWeightBank(){};

    void initialize(TMMemory<T>& memory, std::size_t number_of_clauses) {

        std::span<uint32_t> originalSpan = memory.getSegment(number_of_clauses);
        std::span<int32_t> reinterpretedSpan(reinterpret_cast<int32_t*>(originalSpan.data()), originalSpan.size());
        weights = reinterpretedSpan;

        TMWeightBankPresets::positive_one_negative_minus_one(weights, number_of_clauses);
    }

    std::size_t getRequiredMemorySize(std::size_t number_of_clauses) {
        return number_of_clauses;
    }

    void increment(std::span<uint32_t> clause_output, float update_p, std::span<uint32_t>& clause_active, bool positive_weights){

        wb_increment(
                weights.data(),
                weights.size(),
            clause_output.data(),
            update_p,
            clause_active.data(),
            positive_weights
        );
    }

    void decrement(std::span<uint32_t> clause_output, float update_p, std::span<uint32_t>& clause_active, bool negative_weights){

        wb_decrement(
            weights.data(),
            weights.size(),
            clause_output.data(),
            update_p,
            clause_active.data(),
            negative_weights
        );
    }

    std::span<int32_t> getWeights() {
        return weights;
    }



};

#endif //TUMLIBPP_TM_WEIGHT_BANK_H
