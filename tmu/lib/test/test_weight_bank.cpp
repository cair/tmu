#include "gtest/gtest.h"

extern "C"{
    // Include the header files for the functions you want to test
    #include "WeightBank.h"
}

TEST(WeightBankTest, Increment) {
    int clause_weights[] = {0, 1, -1, 2};
    int number_of_clauses = 4;
    unsigned int clause_output[] = {1, 1, 1, 1};
    float update_p = 1.0f;
    unsigned int clause_active[] = {1, 1, 1, 1};
    unsigned int positive_weights = 1;

    wb_increment(clause_weights, number_of_clauses, clause_output, update_p, clause_active, positive_weights);

    int expected_clause_weights[] = {1, 2, -1, 3};
    for (int i = 0; i < number_of_clauses; i++) {
        ASSERT_EQ(expected_clause_weights[i], clause_weights[i]);
    }
}

TEST(WeightBankTest, Decrement) {
    int clause_weights[] = {0, 1, -1, 2};
    int number_of_clauses = 4;
    unsigned int clause_output[] = {1, 1, 1, 1};
    float update_p = 1.0f;
    unsigned int clause_active[] = {1, 1, 1, 1};
    unsigned int negative_weights = 1;

    wb_decrement(clause_weights, number_of_clauses, clause_output, update_p, clause_active, negative_weights);

    int expected_clause_weights[] = {-1, 0, -2, 1};
    for (int i = 0; i < number_of_clauses; i++) {
        ASSERT_EQ(expected_clause_weights[i], clause_weights[i]);
    }
}
