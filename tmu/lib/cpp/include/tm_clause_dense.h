//
// Created by per on 3/1/24.
//

#ifndef TUMLIBPP_TM_CLAUSE_DENSE_H
#define TUMLIBPP_TM_CLAUSE_DENSE_H

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <tl/optional.hpp>
#include <memory>
#include <iostream>
#include "tm_memory.h"

extern "C" {
    #include "ClauseBank.h"
    #include "Tools.h"
}

template<class T>
class TMClauseBankDense: public std::enable_shared_from_this<TMClauseBankDense<T>>{


public:
    float d;
    float s;
    bool boost_true_positive_feedback;
    bool reuse_random_feedback;
    std::size_t number_of_clauses;
    std::size_t number_of_state_bits;
    std::size_t number_of_state_bits_ind;
    std::size_t batch_size;
    std::size_t max_included_literals;
    std::size_t number_of_features;
    std::size_t number_of_literals;

    std::size_t number_of_ta_chunks;
    std::size_t number_of_patches;
    std::tuple<int, int, int> dim;
    std::tuple<int, int> patch_dim;


    bool incremental_clause_evaluation_initialized = false;
    bool incremental;


    // Memory Segments
    tcb::span<T> clause_output;
    tcb::span<T> clause_output_batch;
    tcb::span<T> clause_and_target;
    tcb::span<T> clause_output_patchwise;
    tcb::span<T> feedback_to_ta;
    tcb::span<T> output_one_patches;
    tcb::span<T> literal_clause_count;
    tcb::span<T> type_ia_feedback_counter;
    tcb::span<T> literal_clause_map;
    tcb::span<T> literal_clause_map_pos;
    tcb::span<T> false_literals_per_clause;
    tcb::span<T> previous_xi;
    tcb::span<T> clause_bank;
    tcb::span<T> actions;
    tcb::span<T> clause_bank_ind;

private:

    int seed;


public:

    TMClauseBankDense(
        float _s,
        float _d,
        bool _boost_true_positive_feedback,
        bool _reuse_random_feedback,
        std::vector<int> X_shape,
        tl::optional<std::vector<int>> _patch_dim,
        tl::optional<std::size_t> _max_included_literals,
        std::size_t _number_of_clauses,
        std::size_t _number_of_state_bits,
        std::size_t _number_of_state_bits_ind,
        std::size_t _batch_size,
        bool _incremental,
        int _seed = 0
    )
    : s(_s)
    , d(_d)
    , boost_true_positive_feedback(_boost_true_positive_feedback)
    , reuse_random_feedback(_reuse_random_feedback)
    , number_of_clauses(_number_of_clauses)
    , number_of_state_bits(_number_of_state_bits)
    , number_of_state_bits_ind(_number_of_state_bits_ind)
    , batch_size(_batch_size)
    , incremental(_incremental)
    , seed(_seed)
    {


        // Validate and set the dimensions based on X_shape.
        dim = TMClauseBankDense::getDim(X_shape);
        patch_dim = TMClauseBankDense::getPatchDim(_patch_dim, dim);



        // Calculate the number of features.
        number_of_features = std::get<0>(patch_dim) * std::get<1>(patch_dim) * std::get<2>(dim) +
                             (std::get<0>(dim) - std::get<0>(patch_dim)) +
                             (std::get<1>(dim) - std::get<1>(patch_dim));

        // Calculate the number of literals.
        number_of_literals = number_of_features * 2;

        // Calculate the number of patches.
        number_of_patches = (std::get<0>(dim) - std::get<0>(patch_dim) + 1) *
                            (std::get<1>(dim) - std::get<1>(patch_dim) + 1);

        // Set maximum included literals or use the total number of literals.
        max_included_literals = _max_included_literals.value_or(number_of_literals);

        // Calculate the number of ternary association chunks.
        number_of_ta_chunks = (number_of_literals - 1) / (sizeof(T) * 8) + 1; // Assuming
    }

    static std::tuple<int, int> getPatchDim(const tl::optional<std::vector<int>> _patch_dim, const std::tuple<int, int, int> dim){
        // Set patch dimensions.
        if(!_patch_dim){
            return std::make_tuple(std::get<0>(dim) * std::get<1>(dim) * std::get<2>(dim), 1);
        }else{
            return std::make_tuple(_patch_dim.value()[0], _patch_dim.value()[1]);
        }
    }

    static std::tuple<int, int, int> getDim(const std::vector<int> X_shape){
        if (X_shape.size() == 2) {
            return {X_shape[1], 1, 1};
        } else if (X_shape.size() == 3) {
            return {X_shape[1], X_shape[2], 1};
        } else if (X_shape.size() == 4) {
            return {X_shape[1], X_shape[2], X_shape[3]};
        } else {
            throw std::invalid_argument("X_shape must be a 2D, 3D, or 4D tensor");
        }
    }


    std::size_t getRequiredMemorySize() const {
        return calculateTotalMemorySize();
    }

    bool initialize(TMMemory<T>& memory) {
        // Allocate memory segments directly for each data structure.
        clause_output = memory.getSegment(calculateClauseOutputSize());
        clause_output_batch = memory.getSegment(calculateClauseOutputBatchSize());
        clause_and_target = memory.getSegment(calculateClauseAndTargetSize());
        clause_output_patchwise = memory.getSegment(calculateClauseOutputPatchwiseSize());
        feedback_to_ta = memory.getSegment(calculateFeedbackToTaSize());
        output_one_patches = memory.getSegment(calculateOutputOnePatchesSize());
        literal_clause_count = memory.getSegment(calculateLiteralClauseCountSize());
        type_ia_feedback_counter = memory.getSegment(calculateTypeIaFeedbackCounterSize());
        literal_clause_map = memory.getSegment(calculateLiteralClauseMapSize());
        literal_clause_map_pos = memory.getSegment(calculateLiteralClauseMapPosSize());
        false_literals_per_clause = memory.getSegment(calculateFalseLiteralsPerClauseSize());
        previous_xi = memory.getSegment(calculatePreviousXiSize());
        clause_bank = memory.getSegment(calculateClauseBankSize());
        actions = memory.getSegment(calculateActionsSize());
        clause_bank_ind = memory.getSegment(calculateClauseTaChunksStateBitsIndSize());


        initializeClauses();

        return true; // Assuming all allocations are successful
    }

    void initializeClauses(){
        // Set all bits to 1 except the last bit in each "chunk" of the 1D array
        for (int i = 0; i < number_of_clauses; ++i) {
            for (int j = 0; j < number_of_ta_chunks; ++j) {
                for (int k = 0; k < number_of_state_bits - 1; ++k) { // Up to the second last
                    int index = i * (number_of_ta_chunks * number_of_state_bits) + j * number_of_state_bits + k;
                    clause_bank[index] = ~uint32_t(0); // Set all bits to 1
                }
                // Set the last bit to 0
                int last_bit_index = i * (number_of_ta_chunks * number_of_state_bits) + j * number_of_state_bits + (number_of_state_bits - 1);
                clause_bank[last_bit_index] = 0;
            }
        }

        // Sett all bits to 1 for independent clauses
        std::fill(clause_bank_ind.begin(), clause_bank_ind.end(), ~0);
    }

    std::size_t getEncodedXiSize(const std::vector<int>& X_shape) const {
        return X_shape.at(0) * number_of_patches * number_of_ta_chunks;
    }

    const std::vector<uint32_t> prepare_X(
            const tcb::span<uint32_t>& x,
            const std::vector<int>& X_shape

    ){
        std::vector<uint32_t> encoded_X(X_shape.at(0) * number_of_patches * number_of_ta_chunks) ;

        tmu_encode(
                x.data(),
                encoded_X.data(),
                X_shape.at(0),
                std::get<0>(dim),
                std::get<1>(dim),
                std::get<2>(dim),
                std::get<0>(patch_dim),
                std::get<1>(patch_dim),
                1, // TODO
                0 // TODO
        );

        return std::move(encoded_X);
    }


    // Assuming we have a method to calculate the position in clause_bank
    size_t calculatePosition(size_t clause, size_t ta_chunk) const {
        return clause * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits;
    }

    void setTAState(size_t clause, size_t ta, unsigned int state) {
        size_t ta_chunk = ta / 32;
        size_t chunk_pos = ta % 32;
        size_t pos = calculatePosition(clause, ta_chunk);
        for (size_t b = 0; b < number_of_state_bits; ++b) {
            if (state & (1 << b)) {
                clause_bank[pos + b] |= (1 << chunk_pos);
            } else {
                clause_bank[pos + b] &= ~(1 << chunk_pos);
            }
        }
    }

    unsigned int getTAState(size_t clause, size_t ta) {
        size_t ta_chunk = ta / 32;
        size_t chunk_pos = ta % 32;
        size_t pos = calculatePosition(clause, ta_chunk);
        unsigned int state = 0;
        for (size_t b = 0; b < number_of_state_bits; ++b) {
            if (clause_bank[pos + b] & (1 << chunk_pos)) {
                state |= (1 << b);
            }
        }
        return state;
    }

    bool get_ta_action(size_t clause, size_t ta) {
        size_t ta_chunk = ta / 32;
        size_t chunk_pos = ta % 32;
        size_t pos = calculatePosition(clause, ta_chunk) + number_of_state_bits - 1;
        return (clause_bank[pos] & (1 << chunk_pos)) > 0;
    }

    tcb::span<T> calculateClauseOutputsPredict() {
        return clause_output;
    }

    std::size_t calculateClauseBankSize() const {
        return number_of_clauses * number_of_ta_chunks * number_of_state_bits;
    }

    std::size_t calculateClauseOutputBatchSize() const {
        return number_of_clauses * batch_size;
    }




    void type_i_feedback(
            float update_p,
            const tcb::span<T>& clause_active,
            const tcb::span<T>& literal_active,
            const tcb::span<T>& encoded_xi
    ){

        cb_type_i_feedback(
                clause_bank.data(),
                feedback_to_ta.data(),
                output_one_patches.data(),
                number_of_clauses,
                number_of_literals,
                number_of_state_bits,
                number_of_patches,
                update_p,
                s,
                boost_true_positive_feedback,
                reuse_random_feedback,
                max_included_literals,
                clause_active.data(),
                literal_active.data(),
                encoded_xi.data()
        );

        incremental_clause_evaluation_initialized = false;

    }

    void type_ii_feedback(
            float update_p,
            const tcb::span<T>& clause_active,
            const tcb::span<T>& literal_active,
            const tcb::span<T>& encoded_xi
    ){
        cb_type_ii_feedback(
            clause_bank.data(),
            output_one_patches.data(),
            number_of_clauses,
            number_of_literals,
            number_of_state_bits,
            number_of_patches,
            update_p,
            clause_active.data(),
            literal_active.data(),
            encoded_xi.data()
        );

        incremental_clause_evaluation_initialized = false;
    }

    void type_iii_feedback(
            float update_p,
            const tcb::span<T>& clause_active,
            const tcb::span<T>& literal_active,
            const tcb::span<T>& encoded_X_train,
            bool target
    ){
        cb_type_iii_feedback(
                clause_bank.data(),
                clause_bank_ind.data(),
                clause_and_target.data(),
                output_one_patches.data(),
                number_of_clauses,
                number_of_literals,
                number_of_state_bits,
                number_of_state_bits_ind,
                number_of_patches,
                update_p,
                d,
                clause_active.data(),
                literal_active.data(),
                encoded_X_train.data(), // TODO
                target
        );

        incremental_clause_evaluation_initialized = false;
    }

    void calculate_clause_outputs_update(
            const tcb::span<T>& literal_active,
            const tcb::span<T>& encoded_xi
    ){
        cb_calculate_clause_outputs_update(
                clause_bank.data(),
                number_of_clauses,
                number_of_literals,
                number_of_state_bits,
                number_of_patches,
                clause_output.data(),
                literal_active.data(),
                encoded_xi.data()
        );

    }
    
    
    const tcb::span<T> calculate_clause_outputs_predict(
            const tcb::span<T>& encoded_xi,
            std::size_t sample_index,
            std::size_t n_items
    ){

        if(!incremental){
            cb_calculate_clause_outputs_predict(
                    clause_bank.data(),
                    number_of_clauses,
                    number_of_literals,
                    number_of_state_bits,
                    number_of_patches,
                    clause_output.data(),
                    encoded_xi.data()
            );

            return clause_output;
        }

        if(!incremental_clause_evaluation_initialized){

            cb_initialize_incremental_clause_calculation(
                    clause_bank.data(),
                    literal_clause_map.data(),
                    literal_clause_map_pos.data(),
                    false_literals_per_clause.data(),
                    number_of_clauses,
                    number_of_literals,
                    number_of_state_bits,
                    previous_xi.data()
            );

            incremental_clause_evaluation_initialized = true;
        }

        if(sample_index % batch_size == 0){
            cb_calculate_clause_outputs_incremental_batch(
                    literal_clause_map.data(),
                    literal_clause_map_pos.data(),
                    false_literals_per_clause.data(),
                    number_of_clauses,
                    number_of_literals,
                    number_of_patches,
                    clause_output_batch.data(),
                    previous_xi.data(),
                    encoded_xi.data(),
                    std::min(batch_size, n_items - sample_index)

            );
        }

        size_t start_index = (sample_index % batch_size) * number_of_clauses;

        // Create a span that represents the slice of the clause outputs for the entry 'e'
        tcb::span<T> cl(
                clause_output_batch.data() + start_index,
                number_of_clauses
        );

        return cl;
    }


private:
    std::size_t calculateClauseOutputSize() const {
        return number_of_clauses;
    }

    std::size_t calculateClauseAndTargetSize() const {
        return number_of_clauses * number_of_ta_chunks;
    }

    std::size_t calculateClauseOutputPatchwiseSize() const {
        return number_of_clauses * number_of_patches;
    }

    std::size_t calculateFeedbackToTaSize() const {
        return number_of_ta_chunks;
    }

    std::size_t calculateOutputOnePatchesSize() const {
        return number_of_patches;
    }

    std::size_t calculateLiteralClauseCountSize() const {
        return number_of_literals;
    }

    std::size_t calculateTypeIaFeedbackCounterSize() const {
        return number_of_clauses;
    }

    std::size_t calculateLiteralClauseMapSize() const {
        return number_of_literals * number_of_clauses;
    }

    std::size_t calculateLiteralClauseMapPosSize() const {
        return number_of_literals;
    }

    std::size_t calculateFalseLiteralsPerClauseSize() const {
        return number_of_clauses * number_of_patches;
    }

    std::size_t calculatePreviousXiSize() const {
        return number_of_ta_chunks * number_of_patches;
    }

    std::size_t calculateActionsSize() const {
        return number_of_ta_chunks;
    }

    std::size_t calculateClauseTaChunksStateBitsIndSize() const {
        return number_of_clauses * number_of_ta_chunks * number_of_state_bits_ind;
    }

    std::size_t calculateTotalMemorySize() const {
        return
            calculateClauseOutputSize() +
            calculateClauseOutputBatchSize() +
            calculateClauseAndTargetSize() +
            calculateClauseOutputPatchwiseSize() +
            calculateFeedbackToTaSize() +
            calculateOutputOnePatchesSize() +
            calculateLiteralClauseCountSize() +
            calculateTypeIaFeedbackCounterSize() +
            calculateLiteralClauseMapSize() +
            calculateLiteralClauseMapPosSize() +
            calculateFalseLiteralsPerClauseSize() +
            calculatePreviousXiSize() +
            calculateClauseBankSize() +
            calculateActionsSize() +
            calculateClauseTaChunksStateBitsIndSize();
    }

    void calculate_literal_clause_frequency(){
        // clause_active
        // TODO
    }

    void calculate_clause_outputs_predict(){
        // TODO
    }

    void included_literals(){
        // TODO
    }

    void get_literals(){
        // TODO
    }

    void number_of_include_actions(){
        // TODO
    }

    void prepare_Xi(){
        // TODO
    }

    void prepare_X_autoencoder(){
        // TODO
    }

    void produce_autoencoder_example(){
        // TODO
    }

};


#endif //TUMLIBPP_TM_CLAUSE_DENSE_H
