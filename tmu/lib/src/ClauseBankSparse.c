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

#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcount __popcnt
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"
#include <stdint.h>

static inline void cbs_initialize_random_streams(unsigned int *feedback_to_ta, int number_of_literals, int number_of_ta_chunks, float s)
{
	// Initialize all bits to zero	
	memset(feedback_to_ta, 0, number_of_ta_chunks*sizeof(unsigned int));

	int n = number_of_literals;
	float p = 1.0 / s;

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % (number_of_literals);
		while (feedback_to_ta[f / 32] & (1U << (f % 32))) {
			f = fast_rand() % (number_of_literals);
	    }
		feedback_to_ta[f / 32] |= (1U << (f % 32));
	}
}

void cbs_prepare_Xi(unsigned int *indices, int number_of_indices, unsigned int *Xi, int number_of_features)
{
    for (int k = 0; k < number_of_indices; ++k) { 
        unsigned int chunk = indices[k] / 32;
        unsigned int pos = indices[k] % 32;
        Xi[chunk] |= (1U << pos);
        chunk = (indices[k] + number_of_features) / 32;
        pos = (indices[k] + number_of_features) % 32;
        Xi[chunk] &= ~(1U << pos);
    }
}

void cbs_restore_Xi(unsigned int *indices, int number_of_indices, unsigned int *Xi, int number_of_features)
{
    for (int k = 0; k < number_of_indices; ++k) { 
        unsigned int chunk = indices[k] / 32;
        unsigned int pos = indices[k] % 32;
        Xi[chunk] &= ~(1U << pos);
        chunk = (indices[k] + number_of_features) / 32;
        pos = (indices[k] + number_of_features) % 32;
        Xi[chunk] |= (1U << pos);
    }
}

void cbs_pack_X(int *indptr, int *indices, int number_of_examples, int e, unsigned int *packed_X, int number_of_literals)
{
    for (int k = 0; k < number_of_literals/2; ++k) {
    	packed_X[k] = 0U;
    }

    for (int k = number_of_literals/2; k < number_of_literals; ++k) {
    	packed_X[k] = ~0U;
    }

    for (int i = 0; i < 32; ++i) {
        if (e+i >= number_of_examples) {
            break;
        }

        for (int k = indptr[e+i]; k < indptr[e+i+1]; ++k) {
            packed_X[indices[k]] |= (1U << i);
            packed_X[indices[k] + number_of_literals/2] &= ~(1U << i);
        }
    }
}

void cbs_unpack_clause_output(int e, unsigned int *clause_output, unsigned int *clause_output_batch, int number_of_clauses)
{
	for (int j = 0; j < number_of_clauses; ++j) {
		clause_output[j] = ((clause_output_batch[j] & (1U << (e % 32U))) > 0);
	}
}

void cbs_calculate_clause_outputs_update(unsigned int *literal_active, unsigned int *Xi, int number_of_clauses, int number_of_literals, unsigned int *clause_output, unsigned short *clause_bank_included, unsigned short *clause_bank_included_length, unsigned short *clause_bank_included_absorbed, unsigned short *clause_bank_included_absorbed_length)
{
    for (int j = 0; j < number_of_clauses; ++j) {
        clause_output[j] = 1;

        for (int k = 0; k < clause_bank_included_absorbed_length[j]; ++k) {
            unsigned int clause_pos = j*number_of_literals + k;
            unsigned int literal_chunk = clause_bank_included_absorbed[clause_pos] / 32U;
            unsigned int literal_pos = clause_bank_included_absorbed[clause_pos] % 32U;
            if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                clause_output[j] = 0;
                break;
            }
        }

        if (clause_output[j]) {
            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
            	unsigned int clause_pos = j*number_of_literals*2 + k*2;
                unsigned int literal_chunk = clause_bank_included[clause_pos] / 32U;
                unsigned int literal_pos = clause_bank_included[clause_pos] % 32U;
                if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                    clause_output[j] = 0;
                    break;
                }
            }
        }
    }
}

void cbs_calculate_clause_outputs_predict_packed_X(unsigned int *packed_X, int number_of_clauses, int number_of_literals, unsigned int *clause_output_batch, unsigned short *clause_bank_included, unsigned short *clause_bank_included_length, unsigned short *clause_bank_included_absorbed,
                    unsigned short *clause_bank_included_absorbed_length)
{
     for (int j = 0; j < number_of_clauses; ++j) {
         if (clause_bank_included_absorbed_length[j] + clause_bank_included_length[j] == 0) {
             clause_output_batch[j] = 0;
        } else {
             clause_output_batch[j] = ~0;
        }

        for (int k = 0; k < clause_bank_included_absorbed_length[j]; ++k) {
            unsigned int clause_pos = j*number_of_literals + k;
            clause_output_batch[j] &= packed_X[clause_bank_included_absorbed[clause_pos]];
        }

        for (int k = 0; k < clause_bank_included_length[j]; ++k) {
        	unsigned int clause_pos = j*number_of_literals*2 + k*2;
        	clause_output_batch[j] &= packed_X[clause_bank_included[clause_pos]];
        }
    }
}

void cbs_calculate_clause_outputs_predict(unsigned int *Xi, int number_of_clauses, int number_of_literals, unsigned int *clause_output, unsigned short *clause_bank_included, unsigned short *clause_bank_included_length, unsigned short *clause_bank_included_absorbed,
                    unsigned short *clause_bank_included_absorbed_length)
{
    for (int j = 0; j < number_of_clauses; ++j) {
        if (clause_bank_included_absorbed_length[j] + clause_bank_included_length[j] == 0) {
            clause_output[j] = 0;
        } else {
            clause_output[j] = 1;
        }


        for (int k = 0; k < clause_bank_included_absorbed_length[j]; ++k) {
            unsigned int clause_pos = j*number_of_literals + k;
            unsigned int literal_chunk = clause_bank_included_absorbed[clause_pos] / 32;
            unsigned int literal_pos = clause_bank_included_absorbed[clause_pos] % 32;
            if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                clause_output[j] = 0;
                break;
            }
        }

        if (clause_output[j]) {
            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
                unsigned int clause_pos = j*number_of_literals*2 + k*2;
                unsigned int literal_chunk = clause_bank_included[clause_pos] / 32;
                unsigned int literal_pos = clause_bank_included[clause_pos] % 32;
                if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                    clause_output[j] = 0;
                    break;
                }
            }
        }
    }
}

void cbs_type_i_feedback(float update_p, float s, int boost_true_positive_feedback, int max_included_literals, int absorbing_include, int absorbing_exclude, int *clause_active,
                    unsigned int *literal_active, unsigned int *feedback_to_ta, unsigned int *Xi, int number_of_clauses, int number_of_literals, int number_of_states, unsigned short *clause_bank_included,
                    unsigned short *clause_bank_included_length, unsigned short *clause_bank_included_absorbed,
                    unsigned short *clause_bank_included_absorbed_length, unsigned short *clause_bank_excluded, unsigned short *clause_bank_excluded_length)
{
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

    for (int j = 0; j < number_of_clauses; ++j) {
        if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		int clause_pos_base = j*number_of_literals*2;

        int clause_output = 1;

        for (int k = 0; k < clause_bank_included_absorbed_length[j]; ++k) {
            unsigned int clause_pos = j*number_of_literals + k;
            unsigned int literal_chunk = clause_bank_included_absorbed[clause_pos] / 32;
            unsigned int literal_pos = clause_bank_included_absorbed[clause_pos] % 32;
            if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                clause_output = 0;
                break;
            }
        }

        if (clause_output) {
            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
            	unsigned int clause_pos = clause_pos_base + k*2;
                unsigned int literal_chunk = clause_bank_included[clause_pos] / 32;
                unsigned int literal_pos = clause_bank_included[clause_pos] % 32;
                if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                    clause_output = 0;
                    break;
                }
            }
        }

       	cbs_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);

        if (clause_output && (clause_bank_included_length[j] <= max_included_literals)) {
			int k = clause_bank_included_length[j];
			while (k--) {
				int clause_included_pos = clause_pos_base + k*2;
	            unsigned int literal_chunk = clause_bank_included[clause_included_pos] / 32;
	            unsigned int literal_pos = clause_bank_included[clause_included_pos] % 32;

            	if ((Xi[literal_chunk] & (1U << literal_pos)) != 0) {
                   	if (clause_bank_included[clause_included_pos + 1] < number_of_states-1 && (boost_true_positive_feedback || ((feedback_to_ta[literal_chunk] & (1U << literal_pos)) == 0))) {
                        clause_bank_included[clause_included_pos + 1] += 1;

                        if (clause_bank_included[clause_included_pos + 1] >= absorbing_include) {
                            int clause_included_absorbed_pos = j*number_of_literals + clause_bank_included_absorbed_length[j];
                            clause_bank_included_absorbed[clause_included_absorbed_pos] = clause_bank_included[clause_included_pos];
                            clause_bank_included_absorbed_length[j] += 1;

                            clause_bank_included_length[j] -= 1;
                            int clause_included_end_pos = clause_pos_base + clause_bank_included_length[j]*2;
                            clause_bank_included[clause_included_pos] = clause_bank_included[clause_included_end_pos];       
                            clause_bank_included[clause_included_pos + 1] = clause_bank_included[clause_included_end_pos + 1];
                        }
                	}
                } else if (feedback_to_ta[literal_chunk] & (1U << literal_pos)) {
                    clause_bank_included[clause_included_pos + 1] -= 1;
                    if (clause_bank_included[clause_included_pos + 1] < number_of_states / 2) {
                    	int clause_excluded_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                        clause_bank_excluded[clause_excluded_pos] = clause_bank_included[clause_included_pos];
                        clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_included[clause_included_pos + 1];
                        clause_bank_excluded_length[j] += 1;

                        clause_bank_included_length[j] -= 1;
                        int clause_included_end_pos = clause_pos_base + clause_bank_included_length[j]*2;
                        clause_bank_included[clause_included_pos] = clause_bank_included[clause_included_end_pos];       
                        clause_bank_included[clause_included_pos + 1] = clause_bank_included[clause_included_end_pos + 1];
                    }
                }
            }

            k = clause_bank_excluded_length[j];
			while (k--) {
				int clause_excluded_pos = clause_pos_base + k*2;
            	unsigned int literal_chunk = clause_bank_excluded[clause_excluded_pos] / 32;
            	unsigned int literal_pos = clause_bank_excluded[clause_excluded_pos] % 32;
		
            	if ((Xi[literal_chunk] & (1U << literal_pos)) != 0) {
	               if (boost_true_positive_feedback || ((feedback_to_ta[literal_chunk] & (1 << literal_pos)) == 0)) {
                        clause_bank_excluded[clause_excluded_pos + 1] += 1;
                        if (clause_bank_excluded[clause_excluded_pos + 1] >= number_of_states / 2) {
                            int clause_included_pos = clause_pos_base + clause_bank_included_length[j]*2;
		                    clause_bank_included[clause_included_pos] = clause_bank_excluded[clause_excluded_pos];
		                    clause_bank_included[clause_included_pos + 1] = clause_bank_excluded[clause_excluded_pos + 1];
		                    clause_bank_included_length[j] += 1;

		                    clause_bank_excluded_length[j] -= 1;
		                    int clause_excluded_end_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
		                    clause_bank_excluded[clause_excluded_pos] = clause_bank_excluded[clause_excluded_end_pos];
		                    clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_excluded[clause_excluded_end_pos + 1];
                        }
                    }
                } else if ((feedback_to_ta[literal_chunk] & (1 << literal_pos)) && (clause_bank_excluded[clause_excluded_pos + 1] > 0)) {
                    clause_bank_excluded[clause_excluded_pos + 1] -= 1;
                    
                    if ((int)clause_bank_excluded[clause_excluded_pos + 1] <= absorbing_exclude) {
                        clause_bank_excluded_length[j] -= 1;
                        int clause_excluded_end_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                        clause_bank_excluded[clause_excluded_pos] = clause_bank_excluded[clause_excluded_end_pos];
                        clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_excluded[clause_excluded_end_pos + 1];
                    }
                }
            }
        } else {
        	int k = clause_bank_included_length[j];
			while (k--) {
				int clause_included_pos = clause_pos_base + k*2;
            	unsigned int literal_chunk = clause_bank_included[clause_included_pos] / 32;
            	unsigned int literal_pos = clause_bank_included[clause_included_pos] % 32;

				if (feedback_to_ta[literal_chunk] & (1U << literal_pos)) {
                    clause_bank_included[clause_included_pos + 1] -= 1;
                    if (clause_bank_included[clause_included_pos + 1] < number_of_states / 2) {
                    	int clause_excluded_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                        clause_bank_excluded[clause_excluded_pos] = clause_bank_included[clause_included_pos];
                        clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_included[clause_included_pos + 1];
                        clause_bank_excluded_length[j] += 1;

                        clause_bank_included_length[j] -= 1;
                        int clause_included_end_pos = clause_pos_base + clause_bank_included_length[j]*2;
                        clause_bank_included[clause_included_pos] = clause_bank_included[clause_included_end_pos];       
                        clause_bank_included[clause_included_pos + 1] = clause_bank_included[clause_included_end_pos + 1];
                    }
                }
            }
 			
 			k = clause_bank_excluded_length[j];
			while (k--) {
				int clause_excluded_pos = clause_pos_base + k*2;
            	unsigned int literal_chunk = clause_bank_excluded[clause_excluded_pos] / 32;
            	unsigned int literal_pos = clause_bank_excluded[clause_excluded_pos] % 32;
		
            	if ((feedback_to_ta[literal_chunk] & (1 << literal_pos)) && (clause_bank_excluded[clause_excluded_pos + 1] > 0)) {
                    clause_bank_excluded[clause_excluded_pos + 1] -= 1;

                    if ((int)clause_bank_excluded[clause_excluded_pos + 1] <= absorbing_exclude) {
                        clause_bank_excluded_length[j] -= 1;
                        int clause_excluded_end_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                        clause_bank_excluded[clause_excluded_pos] = clause_bank_excluded[clause_excluded_end_pos];
                        clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_excluded[clause_excluded_end_pos + 1];
                    }
                }
            }
    	}
    }
}

void cbs_type_ii_feedback(float update_p, int *clause_active, unsigned int *literal_active, unsigned int *Xi, int number_of_clauses, int number_of_literals, int number_of_states, unsigned short *clause_bank_included,
                    unsigned short *clause_bank_included_length, unsigned short *clause_bank_included_absorbed,
                    unsigned short *clause_bank_included_absorbed_length, unsigned short *clause_bank_excluded, unsigned short *clause_bank_excluded_length)
{
    for (int j = 0; j < number_of_clauses; ++j) {
    	if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

        int clause_output = 1;

        for (int k = 0; k < clause_bank_included_absorbed_length[j]; ++k) {
            unsigned int clause_pos = j*number_of_literals + k;
            unsigned int literal_chunk = clause_bank_included_absorbed[clause_pos] / 32;
            unsigned int literal_pos = clause_bank_included_absorbed[clause_pos] % 32;
            if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                clause_output = 0;
                break;
            }
        }

        if (clause_output) {
            for (int k = 0; k < clause_bank_included_length[j]; ++k) {
            	unsigned int clause_pos = j*number_of_literals*2 + k*2;
                unsigned int literal_chunk = clause_bank_included[clause_pos] / 32;
                unsigned int literal_pos = clause_bank_included[clause_pos] % 32;
                if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                    clause_output = 0;
                    break;
                }
            }
        }

        if (clause_output == 0) {
            continue;
        }

        // Type II Feedback
	
		int clause_pos_base = j*number_of_literals*2;
		int k = clause_bank_excluded_length[j];
		while (k--) {
			int clause_excluded_pos = clause_pos_base + k*2;
            unsigned int literal_chunk = clause_bank_excluded[clause_excluded_pos] / 32;
            unsigned int literal_pos = clause_bank_excluded[clause_excluded_pos] % 32;
		
            if ((Xi[literal_chunk] & (1U << literal_pos)) == 0) {
                clause_bank_excluded[clause_excluded_pos + 1] += 1;

                if (clause_bank_excluded[clause_excluded_pos + 1] >= number_of_states/2) {
                	int clause_included_pos = clause_pos_base + clause_bank_included_length[j]*2;
                    clause_bank_included[clause_included_pos] = clause_bank_excluded[clause_excluded_pos];
                    clause_bank_included[clause_included_pos + 1] = clause_bank_excluded[clause_excluded_pos + 1];
                    clause_bank_included_length[j] += 1;

                    clause_bank_excluded_length[j] -= 1;
                    int clause_excluded_end_pos = clause_pos_base + clause_bank_excluded_length[j]*2;
                    clause_bank_excluded[clause_excluded_pos] = clause_bank_excluded[clause_excluded_end_pos];
                    clause_bank_excluded[clause_excluded_pos + 1] = clause_bank_excluded[clause_excluded_end_pos + 1];
                }
			}
  		} 
	}
}
