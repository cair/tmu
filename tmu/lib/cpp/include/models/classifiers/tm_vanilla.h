//
// Created by per on 3/2/24.
//

#ifndef TUMLIBPP_TM_VANILLA_H
#define TUMLIBPP_TM_VANILLA_H
#include <cmath>
#include "utils/sparse_clause_container.h"
#include "tm_weight_bank.h"
#include "tm_clause_dense.h"
#include "utils/tm_math.h"

extern "C" {
    #include "fast_rand.h"
}

template<class Type>
class TMVanillaClassifier {

public:


    int T;
    uint32_t number_of_clauses;
    bool confidence_driven_updating;
    std::span<int32_t> positive_clauses;
    std::span<int32_t> negative_clauses;

    bool weighted_clauses;
    bool type_i_feedback;
    bool type_ii_feedback;
    bool type_iii_feedback;
    bool type_i_p;
    bool type_ii_p;
    float type_i_ii_ratio;
    int seed;
    std::optional<std::size_t> max_included_literals;
    float s;
    float d;
    float clause_drop_p = 0.0; // TODO
    float literal_drop_p = 0.0; // TODO
    bool feature_negation = true; // TODO

    bool boost_true_positive_feedback;
    bool reuse_random_feedback;
    std::optional<std::vector<int>> patch_dim;
    int32_t number_of_state_bits;
    int32_t number_of_state_bits_ind;
    int32_t batch_size;
    bool incremental;


    SparseClauseContainer<TMClauseBankDense<Type>> clause_banks;
    SparseClauseContainer<TMWeightBank<Type>> weight_banks;
    TMMemory<uint32_t> memory;


    bool _is_initialized = false;

    TMVanillaClassifier(
            int _T,
            float _s,
            float _d,
            uint32_t _number_of_clauses,
            bool _confidence_driven_updating,
            bool _weighted_clauses,
            bool _type_i_feedback,
            bool _type_ii_feedback,
            bool _type_iii_feedback,
            float _type_i_ii_ratio,
            std::optional<std::size_t> _max_included_literals,
            bool _boost_true_positive_feedback,
            bool _reuse_random_feedback,
            std::optional<std::vector<int>> _patch_dim,
            int32_t _number_of_state_bits,
            int32_t _number_of_state_bits_ind,
            int32_t _batch_size,
            bool _incremental,
            int _seed
    )
    : T(_T)
    , s(_s)
    , d(_d)
    , number_of_clauses(_number_of_clauses)
    , confidence_driven_updating(_confidence_driven_updating)
    , weighted_clauses(_weighted_clauses)
    , type_i_feedback(_type_i_feedback)
    , type_ii_feedback(_type_ii_feedback)
    , type_iii_feedback(_type_iii_feedback)
    , type_i_ii_ratio(_type_i_ii_ratio)
    , max_included_literals(_max_included_literals)
    , seed(_seed)
    , clause_banks(_seed)
    , weight_banks(_seed)
    , boost_true_positive_feedback(_boost_true_positive_feedback)
    , reuse_random_feedback(_reuse_random_feedback)
    , patch_dim(_patch_dim)
    , number_of_state_bits(_number_of_state_bits)
    , number_of_state_bits_ind(_number_of_state_bits_ind)
    , batch_size(_batch_size)
    , incremental(_incremental)
    , memory()


    {
        if(type_i_ii_ratio >= 1.0){
            type_i_p = 1.0;
            type_ii_p = 1.0 / type_i_ii_ratio;
        }else{
            type_i_p = type_i_ii_ratio;
            type_ii_p = 1.0;
        }

    }

    void initialize(std::vector<int> X_shape){
        static_assert(sizeof(unsigned int) == sizeof(int32_t), "Sizes of unsigned int and int32_t must be the same.");

        auto unsigned_positive_clauses = memory.getSegment(number_of_clauses);
        auto unsigned_negative_clauses = memory.getSegment(number_of_clauses);
        positive_clauses = std::span<int32_t>(reinterpret_cast<int32_t*>(unsigned_positive_clauses.data()), unsigned_positive_clauses.size());
        negative_clauses = std::span<int32_t>(reinterpret_cast<int32_t*>(unsigned_negative_clauses.data()), unsigned_negative_clauses.size());
    }


    void init_after(){
        // Fill the first half of positive_clauses with 1s
        std::fill(positive_clauses.begin(), positive_clauses.begin() + number_of_clauses / 2, 1);

        // Fill the second half of negative_clauses with 1s
        std::fill(negative_clauses.begin() + number_of_clauses / 2, negative_clauses.end(), 1);

    }

    std::size_t get_required_memory_size(){
        auto mem_size = 0;

        // Negative clauses
        mem_size += number_of_clauses;

        // Positive clauses
        mem_size += number_of_clauses;


        return mem_size;
    }

    std::vector<uint32_t> mechanism_literal_active() {

        auto item = *clause_banks.begin();
        auto number_of_literals = item->number_of_literals;;

        size_t number_of_ta_chunks = (number_of_literals + 31) / 32; // Calculate the number of 32-bit chunks
        std::vector<uint32_t> literal_active(number_of_ta_chunks, 0); // Initialize with zeros

        std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Distribution for random floats between 0 and 1

        // Iterate over each literal to determine its activation based on the drop probability
        for (size_t k = 0; k < number_of_literals; ++k) {

            auto rng = fast_rand() / static_cast<float>(UINT32_MAX);

            if (rng >= literal_drop_p) { // Literal remains active
                size_t ta_chunk = k / 32;
                size_t chunk_pos = k % 32;
                literal_active[ta_chunk] |= (1 << chunk_pos);
            }
        }

        // If feature negation is not applied, clear the corresponding bits for the second half of literals
        if (!feature_negation) {
            for (size_t k = number_of_literals / 2; k < number_of_literals; ++k) {
                size_t ta_chunk = k / 32;
                size_t chunk_pos = k % 32;
                literal_active[ta_chunk] &= ~(1 << chunk_pos);
            }
        }

        return literal_active;
    }



    std::vector<uint32_t> mechanism_clause_active() {
        size_t weight_banks_size = weight_banks.size();
        size_t total_elements = weight_banks_size * number_of_clauses;
        std::vector<uint32_t> clause_active(total_elements, 0);

        // TODO
        // TODO
        // TODO
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 eng(rd()); // Seed the generator
        std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range

        for (size_t idx = 0; idx < total_elements; ++idx) {
            // Generate random float between 0 and 1
            float random_value = distr(eng);
            // Compare with clause_drop_p and set to 1 if the random value is greater or equal
            clause_active[idx] = static_cast<uint32_t>(random_value >= clause_drop_p);
        }

        return clause_active;
    }

    float mechanism_compute_update_probabilities(bool is_target, int class_sum) {
        // Confidence-driven updating method
        if (confidence_driven_updating) {
            return (T - std::abs(class_sum)) / T;
        }

        // Compute based on whether the class is the target or not
        if (is_target) {
            return (T - class_sum) / (2.0 * T);
        }


        return (T + class_sum) / (2.0 * T);
    }



    void mechanism_feedback(
            bool is_target,
            uint32_t target,
            std::span<Type>& clause_output,
            float update_p,
            std::span<Type>& clause_active,
            std::span<Type>& literal_active,
            std::span<Type>& encoded_xi
    ) {

        auto& clause_a = (is_target) ? positive_clauses : negative_clauses;
        auto& clause_b = (is_target) ? negative_clauses : positive_clauses;

        auto clause_active_target = clause_active.subspan(
                target * number_of_clauses,
                number_of_clauses
        );

        auto clause_active_elem_a = TMMath::elementwise_multiply<uint32_t, int32_t>(clause_active_target, clause_a);
        auto clause_active_elem_b = TMMath::elementwise_multiply<uint32_t, int32_t>(clause_active_target, clause_b);

        auto clause_active_elem_a_span = std::span<uint32_t>(clause_active_elem_a.data(), clause_active_elem_a.size());
        auto clause_active_elem_b_span = std::span<uint32_t>(clause_active_elem_b.data(), clause_active_elem_b.size());


        if(weighted_clauses){

            if(is_target) {
                weight_banks[target]->increment(
                        clause_output, // clause_output
                        update_p, // update_p
                        clause_active, // clause_active
                        false // positive_weights
                );

            }else{

                weight_banks[target]->decrement(
                        clause_output, // clause_output
                        update_p, // update_p
                        clause_active, // clause_active
                        false // negative_weights
                );

            }


        }


        if(type_i_feedback){

            clause_banks[target]->type_i_feedback(
                    update_p * type_i_p, // update_p
                    clause_active_elem_a_span, // clause_active
                    literal_active, // literal_active
                    encoded_xi // encoded_X_train
            );
        }

        if(type_ii_feedback){
            clause_banks[target]->type_ii_feedback(
                    update_p * type_ii_p, // update_p
                    clause_active_elem_b_span, // clause_active
                    literal_active, // literal_active
                    encoded_xi // encoded_X_train
            );
        }

        if(type_iii_feedback){
            clause_banks[target]->type_iii_feedback(
                    update_p, // update_p
                    clause_active_elem_a_span, // clause_active
                    literal_active, // literal_active
                    encoded_xi, // encoded_X_train
                    true // target
            );

            clause_banks[target]->type_iii_feedback(
                    update_p, // update_p
                    clause_active_elem_b_span, // clause_active
                    literal_active, // literal_active
                    encoded_xi, // encoded_X_train
                    false // target
            );
        }

    }

    int mechanism_clause_sum(
            uint32_t target,
            std::span<Type>& clause_active,
            std::span<Type>& literal_active,
            std::span<Type>& encoded_xi
    ){
        auto clause_bank = clause_banks[target];

        clause_bank->calculate_clause_outputs_update(
                literal_active,
                encoded_xi
        );

        std::span<int32_t> clause_weights = weight_banks[target]->weights;

        std::span<uint32_t> clause_active_target = clause_active.subspan(
                target * number_of_clauses,
                number_of_clauses
        );

        auto ca_mul_clause_weights = TMMath::elementwise_multiply<uint32_t , int32_t>(
                clause_active_target,
                clause_weights
        );

        auto class_sum = std::inner_product(
                ca_mul_clause_weights.begin(), ca_mul_clause_weights.end(),
                clause_bank->clause_output.begin(),
                0 // Initial sum value
        );

        class_sum = std::clamp(class_sum, -static_cast<int32_t>(T), static_cast<int32_t>(T));


        return class_sum;
    }

    float _fit_sample_target(
            int class_sum,
            std::span<Type>& clause_outputs,
            bool is_target_class,
            uint32_t target,
            std::span<Type>& clause_active,
            std::span<Type>& literal_active,
            std::span<Type>& encoded_xi
    ){


        auto update_p = mechanism_compute_update_probabilities(
                is_target_class,
                class_sum
        );

        mechanism_feedback(
                is_target_class,
                target,
                clause_outputs,
                update_p,
                clause_active,
                literal_active,
                encoded_xi
        );

        return update_p;
    }

    void _fit_sample(
        std::span<Type>& clause_active,
        std::span<Type>& literal_active,
        std::span<Type>& encoded_xi,
        uint32_t target,
        std::optional<uint32_t> not_target = std::nullopt
    ){
        auto class_sum = mechanism_clause_sum(
                target,
                clause_active,
                literal_active,
                encoded_xi
        );

        auto& clause_outputs = clause_banks[target]->clause_output;

        float update_p_target = _fit_sample_target(
                class_sum,
                clause_outputs,
                true,
                target,
                clause_active,
                literal_active,
                encoded_xi
        );


        if(!not_target.has_value()){
            return; // TODO Dict
        }

        auto class_sum_not = mechanism_clause_sum(
                not_target.value(),
                clause_active,
                literal_active,
                encoded_xi
        );
        auto clause_outputs_not = clause_banks[not_target.value()]->clause_output;

        auto update_p_not_target = _fit_sample_target(
                class_sum_not,
                clause_outputs_not,
                false,
                not_target.value(),
                clause_active,
                literal_active,
                encoded_xi
        );

        return; // TODO dict

    }

    void init(std::span<Type>& y, std::vector<int>& X_shape){
        if(_is_initialized){
            return;
        }
        _is_initialized = true;


        // Get unique classes from y (Set)
        std::set<int> unique_classes(y.begin(), y.end());

        // Convert to vector
        std::vector<int> cls(unique_classes.begin(), unique_classes.end());


        // Set up template functions
        clause_banks.template_instance = std::make_shared<TMClauseBankDense<Type>>(
                s,
                d,
                boost_true_positive_feedback,
                reuse_random_feedback,
                X_shape,
                patch_dim,
                max_included_literals,
                number_of_clauses,
                number_of_state_bits,
                number_of_state_bits_ind,
                batch_size,
                incremental,
                seed
        );
        clause_banks.populate(cls);

        weight_banks.template_instance = std::make_shared<TMWeightBank<Type>>();
        weight_banks.populate(cls);

        /// Setup memory
        auto num_classes = weight_banks.size();
        auto mem_size = get_required_memory_size();
        for(auto class_id : weight_banks.get_classes()){
            auto weight_bank = weight_banks[class_id];
            auto clause_bank = clause_banks[class_id];
            mem_size += weight_bank->getRequiredMemorySize(number_of_clauses);
            mem_size += clause_bank->getRequiredMemorySize();
        }
        memory.reserve(mem_size);


        for(auto class_id : weight_banks.get_classes()){
            auto clause_bank = clause_banks[class_id];
            auto weight_bank = weight_banks[class_id];
            clause_bank->initialize(memory);
            weight_bank->initialize(memory, number_of_clauses);
        }

        initialize(X_shape);
        init_after();

    }


    void fit(
            std::span<Type>& y,
            std::span<Type>& encoded_X_train,
            std::vector<int>& X_shape,
            bool shuffle
    ){

        // TODO init( should be here         init(y, X_shape);

        auto num_features = X_shape.at(1);

        auto clause_active_matrix = mechanism_clause_active();
        auto clause_active = std::span(clause_active_matrix.data(), clause_active_matrix.size());

        auto literal_active_vector = mechanism_literal_active();
        auto literal_active = std::span(literal_active_vector.data(), literal_active_vector.size());

        std::vector<int> sample_indices(X_shape.at(0));
        TMMath::aRange(
                X_shape.at(0),
                shuffle,
                sample_indices
        );

        for(auto i = 0; i < sample_indices.size(); i++){
            auto sample_idx = sample_indices[i];
            auto target = y[sample_idx];
            auto not_target = weight_banks.sample({target});


            auto encoded_xi = encoded_X_train.subspan(
                    sample_idx * num_features,
                    num_features
            );

            _fit_sample(
                    clause_active,
                    literal_active,
                    encoded_xi,
                    target,
                    not_target
            );
        }

    }

    std::vector<int> predict_compute_class_sums(
            std::span<Type>& encoded_xi,
            int sample_index,
            int num_items,
            bool clip_class_sum) {

        std::vector<int> class_sums;
        class_sums.reserve(weight_banks.size());

        for (auto& class_id : weight_banks.get_classes()) {
            auto weight_bank = weight_banks[class_id];
            auto clause_bank = clause_banks[class_id];
            std::span<int32_t> weights = weight_bank->weights;

            // Assuming calculate_clause_outputs_predict returns a std::span<uint32_t>
            std::span<uint32_t> clause_outputs = clause_bank->calculate_clause_outputs_predict(
                    encoded_xi,
                    sample_index,
                    num_items
            );

            // Pre-compute the minimum size to avoid checking bounds inside the loop
            size_t minSize = std::min(clause_outputs.size(), weights.size());

            // Dot product for class sum
            int class_sum = 0;
            for (size_t i = 0; i < minSize; ++i) {
                class_sum += static_cast<int64_t>(clause_outputs[i]) * weights[i];
            }

            if (clip_class_sum) {
                // Assuming T is defined and provided
                class_sum = std::clamp(class_sum, -T, T);
            }

            class_sums.push_back(class_sum);
        }

        return class_sums;
    }




    inline std::pair<std::vector<int>, std::optional<std::vector<std::vector<int>>>> predict(
            std::span<Type>& encoded_X_test,
            const std::vector<int>& X_shape,
            bool clip_class_sum = false,
            bool return_class_sum = false) {

        auto num_items = X_shape.at(0);
        auto num_features = X_shape.at(1);

        std::vector<int> argmax_indices(num_items, 0);
        std::optional<std::vector<std::vector<int>>> optional_class_sums;

        if (return_class_sum) {
            // Initialize class_sums_matrix only if needed
            optional_class_sums.emplace(num_items, std::vector<int>(weight_banks.size(), 0));
        }

        for (int sample_index = 0; sample_index < num_items; ++sample_index) {
            auto encoded_xi = encoded_X_test.subspan(sample_index * num_features, num_features);
            auto class_sums_vector = predict_compute_class_sums(
                    encoded_xi,
                    sample_index,
                    num_items,
                    clip_class_sum
            );

            // Compute argmax
            int argmax_index = std::distance(
                    class_sums_vector.begin(),
                    std::max_element(
                            class_sums_vector.begin(),
                            class_sums_vector.end()
                    )
            );
            argmax_indices[sample_index] = argmax_index;

            // Store the class sums if requested
            if (return_class_sum) {
                (*optional_class_sums)[sample_index] = std::move(class_sums_vector);
            }
        }

        // Return both argmax_indices and optionally class_sums_matrix
        return {std::move(argmax_indices), std::move(optional_class_sums)};
    }

};

#endif //TUMLIBPP_TM_VANILLA_H
