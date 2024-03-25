#include "tm_clause_dense.h"
#include "utils/tm_dataset.h"
#include <chrono>
#include <vector>
#include <tcb/span.hpp>
#include "models/classifiers/tm_vanilla.h"
#include <cstdint>
#include <type_traits>
static_assert(sizeof(uint32_t) == sizeof(unsigned int), "uint32_t is not the same size as unsigned int");
static_assert(std::is_unsigned<uint32_t>::value, "uint32_t is not unsigned");


void perform_epoch(
        TMVanillaClassifier<uint32_t>& classifier,
        int epoch,
        const tcb::span<uint32_t>& y_train_span,
        const tcb::span<uint32_t>& x_train_span,
        const std::vector<int32_t>& x_train_shape,
        const tcb::span<uint32_t>& x_test_span,
        const std::vector<int32_t>& x_test_shape,
        const tcb::span<uint32_t>& y_test_span
) {
    auto timer_train_start = std::chrono::high_resolution_clock::now();
    classifier.fit(
            y_train_span,
            x_train_span,
            x_train_shape,
            true
    );
    auto timer_train_end = std::chrono::high_resolution_clock::now();

    auto timer_pred_start = std::chrono::high_resolution_clock::now();
    auto predict_result= classifier.predict(x_test_span, x_test_shape);
    auto y_pred = std::get<0>(predict_result);
    auto y_pred_shape = std::get<1>(predict_result);

    std::vector<int> results(y_pred.size());
    std::transform(y_pred.begin(), y_pred.end(), y_test_span.begin(), results.begin(),
                   [](int pred, int actual) -> int { return pred == actual ? 1 : 0; });

    int correct = std::accumulate(results.begin(), results.end(), 0);

    double accuracy = static_cast<double>(correct) / static_cast<double>(y_test_span.size()) * 100.0;
    auto timer_pred_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_train = timer_train_end - timer_train_start;
    std::chrono::duration<double, std::milli> elapsed_pred = timer_pred_end - timer_pred_start;

    std::cout << "Epoch: " << epoch << " | Accuracy: " << accuracy << "% | "
              << "Training Time: " << elapsed_train.count() / 1000 << "s | "
              << "Prediction Time: " << elapsed_pred.count() / 1000 << "s\n";
}


// Function to perform the training and testing of the TMVanillaClassifier
void train_and_test_classifier() {
    // Initialize the TMVanillaClassifier with predefined parameters
    auto NUM_EPOCHS = 10;
    TMVanillaClassifier<uint32_t> classifier(
            5000, // T
            10.0, // s
            200.0, // d
            2000, // number_of_clauses
            false, // _confidence_driven_updating
            true, // _weighted_clauses
            true, // type_1
            true, // type_2
            false, // type_3
            1.0,  // _type_i_ii_ratio
            tl::nullopt, // max_included_literals
            true,  // boost_true_positive_feedback
            true, // reuse_random_feedback // TODO - set to true
            tl::nullopt, // patch_dim
            8, // number of state bits
            8, // number of state bits ind
            100, // batch_size
            true,
            42 // _seed
    );

    // Load datasets
    auto x_train_result = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/x_train.txt");
    auto y_train_result = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/y_train.txt");
    auto x_test_result = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/x_test.txt");
    auto y_test_result = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/y_test.txt");

    // Assuming read_dataset_from_txt returns a pair or tuple that can be accessed by std::get
    auto x_train = std::get<0>(x_train_result);
    auto y_train = std::get<0>(y_train_result);
    auto x_test = std::get<0>(x_test_result);
    auto y_test = std::get<0>(y_test_result);

    auto x_train_shape = std::get<1>(x_train_result);
    auto y_train_shape = std::get<1>(y_train_result);
    auto x_test_shape = std::get<1>(x_test_result);
    auto y_test_shape = std::get<1>(y_test_result);

    // Prepare data spans
    auto x_train_span = tcb::span<uint32_t>(x_train.data(), x_train.size());
    auto y_train_span = tcb::span<uint32_t>(y_train.data(), y_train.size());
    auto x_test_span = tcb::span<uint32_t>(x_test.data(), x_test.size());
    auto y_test_span = tcb::span<uint32_t>(y_test.data(), y_test.size());


    // Training and Testing loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        perform_epoch(
                classifier,
                epoch,
                y_train_span,
                x_train_span,
                x_train_shape,
                x_test_span,
                x_test_shape,
                y_test_span
        );
    }
}


int main() {
    try {
        train_and_test_classifier();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }

    return 0;
}