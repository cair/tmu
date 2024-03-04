//
// Created by per on 3/3/24.
//

#ifndef TUMLIBPP_TM_DATASET_H
#define TUMLIBPP_TM_DATASET_H

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint> // for uint32_t
#include <iostream>
#include <utility> // for std::pair
#include <numeric> // for std::accumulate
#include <random> // for std::random_device, std::mt19937, std::uniform_int_distribution

class TMDataset {

public:

    static std::pair<std::vector<uint32_t>, std::vector<int>> read_dataset_from_txt(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + file_path);
        }

        std::string line;
        std::vector<uint32_t> dataset_1d;
        int num_rows = 0;
        int num_cols = 0;

        while (std::getline(file, line)) {
            std::istringstream line_stream(line);
            std::string value;
            int current_col_count = 0;

            while (std::getline(line_stream, value, ' ')) {
                if (!value.empty()) { // Check if value is not empty to handle multiple spaces
                    uint32_t pixel = std::stoul(value); // Convert string to unsigned long, then implicitly to uint32_t
                    dataset_1d.push_back(pixel);
                    ++current_col_count;
                }
            }

            if (num_rows == 0) {
                num_cols = current_col_count; // Set the number of columns based on the first row
            }
            ++num_rows;
        }

        file.close();
        return {dataset_1d, {num_rows, num_cols}};
    }

    static void generate_pattern_based_dataset(std::vector<std::vector<uint32_t>>& X, std::vector<uint32_t>& y, int num_samples) {
        X.clear();
        y.clear();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 1);

        for (int i = 0; i < num_samples; ++i) {
            std::vector<uint32_t> x(50);
            // Randomly generate binary features
            for (auto& feature : x) {
                feature = distrib(gen);
            }

            // Determine the class based on a pattern in the features
            // Here, we use the sum of '1's in each 10-feature group to influence the class
            int sum[5] = {0};
            for (int group = 0; group < 5; ++group) {
                sum[group] = std::accumulate(x.begin() + group * 10, x.begin() + (group + 1) * 10, 0);
            }

            // Example pattern-based class determination:
            // Use the parity of the sum of '1's in each group to determine the class
            uint32_t class_label = 0;
            for (int group = 0; group < 5; ++group) {
                if (sum[group] % 2 == 0) { // If the sum of '1's is even, add to the class label
                    class_label += 1 << group;
                }
            }

            // This scheme can produce 32 different classes (from 0 to 31) based on the combination of even/odd sums.
            // To limit it to 10 classes, we can simply take the modulo 10 of the class_label.
            class_label %= 10;

            X.push_back(x);
            y.push_back(class_label);
        }
    }

};


#endif //TUMLIBPP_TM_DATASET_H
