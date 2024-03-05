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
#include <tcb/span.hpp>
#include <tl/optional.hpp>

extern "C" {
    #include "fast_rand.h"
}

template<class Type>
class TMVanillaClassifier {

public:


    int T;
    uint32_t number_of_clauses;
    bool confidence_driven_updating;
    tcb::span<int32_t> positive_clauses;
    tcb::span<int32_t> negative_clauses;

    // data views
    std::vector<uint32_t> encoded_X_train_cached;
    std::vector<uint32_t> encoded_X_test_vector;
    std::vector<uint32_t> encoded_X_train_shape;
    std::vector<uint32_t> encoded_X_test_shape;

    bool weighted_clauses;
    bool type_i_feedback;
    bool type_ii_feedback;
    bool type_iii_feedback;
    bool type_i_p;
    bool type_ii_p;
    float type_i_ii_ratio;
    int seed;
    tl::optional<std::size_t> max_included_literals;
    float s;
    float d;
    float clause_drop_p = 0.0; // TODO
    float literal_drop_p = 0.0; // TODO
    bool feature_negation = true; // TODO

    bool boost_true_positive_feedback;
    bool reuse_random_feedback;
    tl::optional<std::vector<int>> patch_dim;
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
            tl::optional<std::size_t> _max_included_literals,
            bool _boost_true_positive_feedback,
            bool _reuse_random_feedback,
            tl::optional<std::vector<int>> _patch_dim,
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

    void initialize(std::vector<uint32_t> X_shape){
        static_assert(sizeof(unsigned int) == sizeof(int32_t), "Sizes of unsigned int and int32_t must be the same.");

        auto unsigned_positive_clauses = memory.getSegment(number_of_clauses);
        auto unsigned_negative_clauses = memory.getSegment(number_of_clauses);
        positive_clauses = tcb::span<int32_t>(reinterpret_cast<int32_t*>(unsigned_positive_clauses.data()), unsigned_positive_clauses.size());
        negative_clauses = tcb::span<int32_t>(reinterpret_cast<int32_t*>(unsigned_negative_clauses.data()), unsigned_negative_clauses.size());
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



    const std::vector<uint32_t> mechanism_clause_active() {
        size_t weight_banks_size = weight_banks.size();
        size_t total_elements = weight_banks_size * number_of_clauses;
        std::vector<uint32_t> clause_active(total_elements, 0);

        for (size_t idx = 0; idx < total_elements; ++idx) {
            // Generate random float between 0 and 1
            // generates a random number between 0 and 2^32 - 1
            const auto random_value = fast_rand() / static_cast<float>(UINT32_MAX);

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
            const tcb::span<Type>& clause_output,
            float update_p,
            const tcb::span<Type>& clause_active,
            const tcb::span<Type>& literal_active,
            const tcb::span<Type>& encoded_xi
    ) {

        const auto& clause_a = (is_target) ? positive_clauses : negative_clauses;
        const auto& clause_b = (is_target) ? negative_clauses : positive_clauses;

        const auto clause_active_target = clause_active.subspan(
                target * number_of_clauses,
                number_of_clauses
        );

        auto clause_active_elem_a = TMMath::elementwise_multiply<uint32_t, int32_t>(clause_active_target, clause_a);
        auto clause_active_elem_b = TMMath::elementwise_multiply<uint32_t, int32_t>(clause_active_target, clause_b);

        const auto clause_active_elem_a_span = tcb::span<uint32_t>(clause_active_elem_a.data(), clause_active_elem_a.size());
        const auto clause_active_elem_b_span = tcb::span<uint32_t>(clause_active_elem_b.data(), clause_active_elem_b.size());


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
            const tcb::span<Type>& clause_active,
            const tcb::span<Type>& literal_active,
            const tcb::span<Type>& encoded_xi
    ){
        auto clause_bank = clause_banks[target];

        clause_bank->calculate_clause_outputs_update(
                literal_active,
                encoded_xi
        );

        const tcb::span<int32_t> clause_weights = weight_banks[target]->weights;

        const tcb::span<uint32_t> clause_active_target = clause_active.subspan(
                target * number_of_clauses,
                number_of_clauses
        );

        const auto ca_mul_clause_weights = TMMath::elementwise_multiply<uint32_t , int32_t>(
                clause_active_target,
                clause_weights
        );

        int32_t class_sum = std::inner_product(
                ca_mul_clause_weights.begin(), ca_mul_clause_weights.end(),
                clause_bank->clause_output.begin(),
                0 // Initial sum value
        );

        class_sum = TMMath::clamp(class_sum, -static_cast<int32_t>(T), static_cast<int32_t>(T));


        return class_sum;
    }

    float _fit_sample_target(
            int class_sum,
            const tcb::span<Type>& clause_outputs,
            bool is_target_class,
            uint32_t target,
            const tcb::span<Type>& clause_active,
            const tcb::span<Type>& literal_active,
            const tcb::span<Type>& encoded_xi
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
        const tcb::span<Type>& clause_active,
        const tcb::span<Type>& literal_active,
        const tcb::span<Type>& encoded_xi,
        uint32_t target,
        tl::optional<uint32_t> not_target = tl::nullopt
    ){
        auto class_sum = mechanism_clause_sum(
                target,
                clause_active,
                literal_active,
                encoded_xi
        );

        const auto& clause_outputs = clause_banks[target]->clause_output;

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
        const auto clause_outputs_not = clause_banks[not_target.value()]->clause_output;

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

    void init(
        const tcb::span<Type>& y,
        const tcb::span<Type>& x,
        const std::vector<int32_t>& X_shape
    ){
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

        mem_size += clause_banks.begin()->get()->getEncodedXiSize(X_shape);
        mem_size += y.size();

        memory.reserve(mem_size);

        auto number_of_ta_chunks = clause_banks.begin()->get()->number_of_ta_chunks;
        auto encoded_x_vector = clause_banks.begin()->get()->prepare_X(
                x,
                X_shape
        );

        // Retrieve memory segment for the dataset
        //encoded_X_train_cached = memory.getSegment(encoded_x_vector.size()); // TODO not using cache
        encoded_X_train_shape = std::vector<uint32_t>({
            static_cast<uint32_t>(X_shape.at(0)),
            static_cast<uint32_t>(number_of_ta_chunks)
        });
        encoded_X_train_cached = std::vector<uint32_t>(encoded_x_vector.size());

        // Copy the encoded_x_vector to the memory
        std::copy(encoded_x_vector.begin(), encoded_x_vector.end(), encoded_X_train_cached.begin());


        for(auto class_id : weight_banks.get_classes()){
            auto clause_bank = clause_banks[class_id];
            auto weight_bank = weight_banks[class_id];
            clause_bank->initialize(memory);
            weight_bank->initialize(memory, number_of_clauses);
        }

        initialize(encoded_X_train_shape);
        init_after();

    }


    void fit(
            const tcb::span<Type>& y,
            const tcb::span<Type>& x,
            const std::vector<int32_t>& X_shape,
            bool shuffle
    ){

        init(y, x,X_shape);


        const auto encoded_X = tcb::span<Type>(encoded_X_train_cached.data(), encoded_X_train_cached.size());
        const auto& encoded_X_shape = encoded_X_train_shape;
        const auto num_features = encoded_X_shape.at(1);

        auto clause_active_matrix = mechanism_clause_active();
        const auto clause_active = tcb::span<uint32_t>(clause_active_matrix.data(), clause_active_matrix.size());

        auto literal_active_vector = mechanism_literal_active();
        const auto literal_active = tcb::span<uint32_t>(literal_active_vector.data(), literal_active_vector.size());

        std::vector<int> sample_indices(encoded_X_shape.at(0));
        TMMath::aRange(
                encoded_X_shape.at(0),
                shuffle,
                sample_indices
        );

        for(auto i = 0; i < sample_indices.size(); i++){
            const auto sample_idx = sample_indices[i];
            const auto target = y[sample_idx];
            const auto not_target = weight_banks.sample({target});


            const auto encoded_xi = encoded_X.subspan(
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
            const tcb::span<Type>& encoded_xi,
            int sample_index,
            int num_items,
            bool clip_class_sum) {
        std::vector<int> class_sums;
        class_sums.reserve(weight_banks.size()); // Assuming weight_banks is accessible and defined.

        for (const auto& class_id : weight_banks.get_classes()) { // Assuming get_classes() is a method returning class IDs.
            const auto& weight_bank = weight_banks[class_id]; // Use const ref to avoid copies.
            const auto& clause_bank = clause_banks[class_id]; // Use const ref to avoid copies.
            const tcb::span<int32_t>& weights = weight_bank->weights; // Assuming weights is accessible and defined.

            // Assuming calculate_clause_outputs_predict returns a tcb::span<uint32_t> and is correctly defined.
            const tcb::span<uint32_t> clause_outputs = clause_bank->calculate_clause_outputs_predict(
                    encoded_xi,
                    sample_index,
                    num_items
            );

            int class_sum = std::inner_product(
                    clause_outputs.begin(),
                    clause_outputs.begin() + std::min(clause_outputs.size(), weights.size()),
                    weights.begin(),
                    0);

            if (clip_class_sum) {
                // Assuming T is defined and TMMath::clamp is correctly implemented.
                class_sum = TMMath::clamp(class_sum, -T, T);
            }

            class_sums.push_back(class_sum);
        }

        return class_sums;
    }


    const tcb::span<Type> getEncodedTestData(
            const tcb::span<Type>& x,
            const std::vector<int32_t>& X_shape
    ){

        if(encoded_X_test_vector.size() != 0){
           return tcb::span<Type>(encoded_X_test_vector.data(), encoded_X_test_vector.size());
        }

        auto& clause_bank = *clause_banks.begin();
        auto number_of_ta_chunks = clause_bank->number_of_ta_chunks;

        encoded_X_test_vector = clause_bank->prepare_X(
                x,
                X_shape
        );

        encoded_X_test_shape = std::vector<uint32_t>(
        {
            static_cast<uint32_t>(X_shape.at(0)),
            static_cast<uint32_t>(number_of_ta_chunks)
        });

        return tcb::span<Type>(encoded_X_test_vector.data(), encoded_X_test_vector.size());
    }


    const std::pair<std::vector<int>, tl::optional<std::vector<std::vector<int>>>> predict(
            const tcb::span<Type>& X_test,
            const std::vector<int32_t >& X_shape,
            bool clip_class_sum = false,
            bool return_class_sum = false) {


        const auto encoded_X_test = getEncodedTestData(X_test, X_shape);
        const auto num_items = encoded_X_test_shape.at(0);
        const auto num_features = encoded_X_test_shape.at(1);

        std::vector<int> argmax_indices(num_items, 0);
        tl::optional<std::vector<std::vector<int>>> optional_class_sums;

        if (return_class_sum) {
            // Initialize class_sums_matrix only if needed
            optional_class_sums.emplace(num_items, std::vector<int>(weight_banks.size(), 0));
        }

        for (int sample_index = 0; sample_index < num_items; ++sample_index) {
            const auto encoded_xi = encoded_X_test.subspan(sample_index * num_features, num_features);

            const auto class_sums_vector = predict_compute_class_sums(
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
        return {
            std::move(argmax_indices),
            std::move(optional_class_sums)
        };
    }

};

#endif //TUMLIBPP_TM_VANILLA_H
