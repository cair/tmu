/*

Copyright (c) 2023 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

void cbs_prepare_Xi(
    unsigned int *indices,
    int number_of_indices,
    unsigned int *Xi,
    int number_of_features
);

void cbs_restore_Xi(
    unsigned int *indices,
    int number_of_indices,
    unsigned int *Xi,
    int number_of_features
);

void cbs_calculate_clause_outputs_predict_packed_X(
    unsigned int *packed_X,
    int number_of_clauses,
    int number_of_literals,
    unsigned int *clause_output_batch,
    unsigned short *clause_bank_included,
    unsigned short *clause_bank_included_length
);

void cbs_unpack_clause_output(
    int e,
    unsigned int *clause_output,
    unsigned int *clause_output_batch,
    int number_of_clauses
);

void cbs_pack_X(
    int *indptr,
    int *indices,
    int number_of_examples,
    int e,
    unsigned int *packed_X,
    int number_of_literals
);

void cbs_calculate_clause_outputs_update(
    unsigned int *literal_active,
    unsigned int *Xi,
    int number_of_clauses,
    int number_of_literals,
    unsigned int *clause_output,
    unsigned short *clause_bank_included,
    unsigned short *clause_bank_included_length
);

void cbs_calculate_clause_outputs_predict(
    unsigned int *Xi,
    int number_of_clauses,
    int number_of_literals,
    unsigned int *clause_output,
    unsigned short *clause_bank_included,
    unsigned short *clause_bank_included_length
);

void cbs_type_i_feedback(
    float update_p,
    float s,
    int boost_true_positive_feedback,
    int max_included_literals,
    int absorbing,
    int feedback_rate_excluded_literals,
    int literal_insertion_state,
    int *clause_active,
    unsigned int *literal_active,
    unsigned int *Xi,
    int number_of_clauses,
    int number_of_literals,
    int number_of_states,
    unsigned short *clause_bank_included,
    unsigned short *clause_bank_included_length,
    unsigned short *clause_bank_excluded,
    unsigned short *clause_bank_excluded_length,
    unsigned short *clause_bank_unallocated,
    unsigned short *clause_bank_unallocated_length
);

void cbs_type_ii_feedback(
    float update_p,
    int feedback_rate_excluded_literals,
    int *clause_active,
    unsigned int *literal_active,
    unsigned int *Xi,
    int number_of_clauses,
    int number_of_literals,
    int number_of_states,
    unsigned short *clause_bank_included,
    unsigned short *clause_bank_included_length,
    unsigned short *clause_bank_excluded,
    unsigned short *clause_bank_excluded_length
);