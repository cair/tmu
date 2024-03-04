#include "tm_clause_dense.h"
#include "models/classifiers/tm_vanilla.h"
#include "utils/tm_dataset.h"
#include <chrono>
#include <vector>
#include <span>


void perform_epoch(TMVanillaClassifier<uint32_t>& classifier, int epoch,
                   std::span<uint32_t>& y_train_span, std::span<uint32_t>& encoded_X_span,
                   std::vector<int32_t>& encoded_X_train_shape, std::span<uint32_t>& encoded_X_test_span,
                   std::vector<int32_t>& encoded_X_test_shape, std::span<uint32_t>& y_test_span) {
    auto timer_train_start = std::chrono::high_resolution_clock::now();
    classifier.fit(y_train_span, encoded_X_span, encoded_X_train_shape, true);
    auto timer_train_end = std::chrono::high_resolution_clock::now();

    auto timer_pred_start = std::chrono::high_resolution_clock::now();
    auto [y_pred, class_sum] = classifier.predict(encoded_X_test_span, encoded_X_test_shape);

    int correct = std::transform_reduce(
            y_pred.begin(), y_pred.end(), // Range of the first container
            y_test_span.begin(),          // Start of the second container
            0,                            // Initial value for the reduction
            std::plus<>(),                // Reducer (sums up the counts)
            [](int pred, int actual) { return pred == actual ? 1 : 0; } // Transformer
    );

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
            std::nullopt, // max_included_literals
            true,  // boost_true_positive_feedback
            true, // reuse_random_feedback // TODO - set to true
            std::nullopt, // patch_dim
            8, // number of state bits
            8, // number of state bits ind
            100, // batch_size
            true,
            42 // _seed
    );

    // Load datasets
    auto [x_train, x_shape] = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/x_train.txt");
    auto [y_train, y_shape] = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/y_train.txt");
    auto [x_test, x_test_shape] = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/x_test.txt");
    auto [y_test, y_test_shape] = TMDataset::read_dataset_from_txt("/mnt/disk/git/code/tmu/tmu/lib/cpp/tests/y_test.txt");

    // Prepare data spans
    auto x_train_span = std::span<uint32_t>(x_train.data(), x_train.size());
    auto y_train_span = std::span<uint32_t>(y_train.data(), y_train.size());
    auto x_test_span = std::span<uint32_t>(x_test.data(), x_test.size());
    auto y_test_span = std::span<uint32_t>(y_test.data(), y_test.size());

    // Determine the shapes
    auto X_shape = std::vector<int32_t>({std::get<0>(x_shape), std::get<1>(x_shape)});
    auto X_test_shape = std::vector<int32_t>({std::get<0>(x_test_shape), std::get<1>(x_test_shape)});

    // Initialization
    classifier.init(y_train_span, X_shape);

    // Prepare encoded data for training and testing
    auto clause_bank = *classifier.clause_banks.begin();
    auto encoded_X = clause_bank->prepare_X(x_train_span, X_shape);
    auto encoded_X_span = std::span<uint32_t>(encoded_X.data(), encoded_X.size());
    auto encoded_X_test = clause_bank->prepare_X(x_test_span, X_test_shape);
    auto encoded_X_test_span = std::span<uint32_t>(encoded_X_test.data(), encoded_X_test.size());

    // Determine encoded shapes
    auto encoded_X_train_shape = std::vector<int32_t>({std::get<0>(x_shape), static_cast<int>(clause_bank->number_of_ta_chunks)});
    auto encoded_X_test_shape = std::vector<int32_t>({std::get<0>(x_test_shape), static_cast<int>(clause_bank->number_of_ta_chunks)});

    // Training and Testing loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        perform_epoch(classifier, epoch, y_train_span, encoded_X_span, encoded_X_train_shape, encoded_X_test_span, encoded_X_test_shape, y_test_span);
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