/*

Copyright (c) 2024 Ole-Christoffer Granmo

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

#include "ClauseBank.h"

static inline void cb_initialize_random_streams(unsigned int *feedback_to_ta, int number_of_literals, int number_of_ta_chunks, float s)
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
		while (feedback_to_ta[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % (number_of_literals);
	    }
		feedback_to_ta[f / 32] |= 1 << (f % 32);
	}
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cb_inc(
	unsigned int *ta_state,
	unsigned int active,
	int number_of_state_bits
)
{
	unsigned int carry, carry_next;

	carry = active;
	for (int b = 0; b < number_of_state_bits; ++b) {
		carry_next = ta_state[b] & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < number_of_state_bits; ++b) {
			ta_state[b] |= carry;
		}
	} 	
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cb_dec(
        unsigned int *ta_state,
        unsigned int active,
        int number_of_state_bits
)
{
	unsigned int carry, carry_next;

	carry = active;
	for (int b = 0; b < number_of_state_bits; ++b) {
	        carry_next = (~ta_state[b]) & carry; // Sets carry bits (overflow) passing on to next bit
	        ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
	        carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < number_of_state_bits; ++b) {
			ta_state[b] &= ~carry;
		}
	}
}

static inline unsigned int cb_clause_all_exclude(
	unsigned int *ta_state,
	int number_of_ta_chunks,
	int number_of_state_bits,
	unsigned int filter
)
{
	unsigned int all_exclude = 1;
	for (int k = 0; k < number_of_ta_chunks-1; k++) {
		unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
		all_exclude = all_exclude && (ta_state[pos] == 0);
	}

	unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
	all_exclude = all_exclude && ((ta_state[pos] & filter) == 0);

	return(all_exclude);
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void cb_calculate_clause_output_feedback(
	unsigned int *ta_state,
	unsigned int *output_one_patches,
	unsigned int *clause_output,
	unsigned int *clause_patch,
	int number_of_ta_chunks,
	int number_of_state_bits,
	unsigned int filter,
	int number_of_patches,
	unsigned int *literal_active,
	unsigned int *Xi
)
{
	int output_one_patches_count = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & (Xi[patch*number_of_ta_chunks + k] | (~literal_active[k]))) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & (Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] | (~literal_active[number_of_ta_chunks - 1])) & filter) ==
			(ta_state[pos] & filter);

		if (output) {
			output_one_patches[output_one_patches_count] = patch;
			output_one_patches_count++;
		}
	}

	if (output_one_patches_count > 0) {
		*clause_output = 1;

		int patch_id = fast_rand() % output_one_patches_count;
 		*clause_patch = output_one_patches[patch_id];
	} else {
		*clause_output = 0;
	}
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline int cb_calculate_clause_output_single_false_literal(unsigned int *ta_state, unsigned int *candidate_offending_literals, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *literal_active, unsigned int *Xi)
{
	int offending_literals_count = 0;
	int offending_literal_id = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int max_one_offending_literal = 1;
		unsigned int already_one_offending_literal = 0;

		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			unsigned int offending_literals = (ta_state[pos] & (Xi[patch*number_of_ta_chunks + k] | (~literal_active[k]))) ^ ta_state[pos];
			if ((offending_literals & (offending_literals - 1)) > 0) {
				max_one_offending_literal = 0;
				break;
			} else if (offending_literals != 0) {
				if (!already_one_offending_literal) {
					already_one_offending_literal = 1;
					offending_literal_id = log2(offending_literals);
				} else {
					max_one_offending_literal = 0;
					break;
				}
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		unsigned int offending_literals = (ta_state[pos] & (Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] | (~literal_active[number_of_ta_chunks - 1])) & filter) ^ (ta_state[pos] & filter);
		if ((offending_literals & (offending_literals - 1)) > 0) {
			max_one_offending_literal = 0;
			break;
		} else if (offending_literals != 0) {
			if (!already_one_offending_literal) {
				already_one_offending_literal = 1;
				offending_literal_id = log2(offending_literals);
			} else {
				max_one_offending_literal = 0;
				break;
			}
		}

		if (max_one_offending_literal && already_one_offending_literal) {
			candidate_offending_literals[offending_literals_count] = offending_literal_id;
			offending_literals_count++;
		}
	}

	if (offending_literals_count > 0) {
		int offending_literal_pos = fast_rand() % offending_literals_count;
 		return(candidate_offending_literals[offending_literal_pos]);
	} else {
		return(-1);
	}
}

static inline unsigned int cb_calculate_clause_output_update(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *literal_active, unsigned int *Xi)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & (Xi[patch*number_of_ta_chunks + k] | (~literal_active[k]))) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & (Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] | (~literal_active[number_of_ta_chunks - 1])) & filter) ==
			(ta_state[pos] & filter);

		if (output) {
			return(1);
		}
	}

	return(0);
}

static inline void cb_calculate_clause_output_patchwise(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *output, unsigned int *Xi)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		output[patch] = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output[patch] = output[patch] && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output[patch]) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output[patch] = output[patch] &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);
	}

	return;
}

static inline unsigned int cb_calculate_clause_output_without_literal_active(
	unsigned int *ta_state,
	int number_of_ta_chunks,
	int number_of_state_bits,
	unsigned int filter,
	unsigned int *Xi
)
{
	unsigned int output = 1;
	for (int k = 0; k < number_of_ta_chunks-1; k++) {
		unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
		output = output && (ta_state[pos] & Xi[k]) == ta_state[pos];

		if (!output) {
			break;
		}
	}

	unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
	output = output &&
		(ta_state[pos] & (Xi[number_of_ta_chunks - 1]) & filter) ==
		(ta_state[pos] & filter);

	return output;
}

static inline unsigned int cb_calculate_clause_output_predict(
	unsigned int *ta_state,
	int number_of_ta_chunks,
	int number_of_state_bits,
	unsigned int filter,
	int number_of_patches,
	unsigned int *Xi
)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		unsigned int all_exclude = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output) {
				break;
			}
			all_exclude = all_exclude && (ta_state[pos] == 0);
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);

		all_exclude = all_exclude && ((ta_state[pos] & filter) == 0);

		if (output && all_exclude == 0) {
			return(1);
		}
	}

	return(0);
}

void cb_calculate_clause_specific_features(
	int clause,
	int number_of_clauses,
        int number_of_literals,
	int number_of_state_bits,
	int number_of_patches,
	unsigned int *clause_value_in_patch,
        unsigned int *clause_true_consecutive,
	unsigned int *Xi
)
{
	unsigned int chunk_nr;
	unsigned int chunk_pos;

	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	// Counts how many times true
	unsigned int number_of_matches = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		if (clause_value_in_patch[clause*number_of_patches + patch]) {
			number_of_matches++;
		}
	}
			
	for (int patch = 0; patch < number_of_patches; ++patch) {
		// Set bits based on how many times true

		for (int l = 0; l < number_of_patches; ++l) {
			if (l < number_of_matches) { 
				chunk_nr = (number_of_clauses*6 + l) / 32;
				chunk_pos = (number_of_clauses*6 + l) % 32;
				Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos);

				chunk_nr = (number_of_clauses*6 + l + number_of_literals/2) / 32;
				chunk_pos = (number_of_clauses*6 + l + number_of_literals/2) % 32;
				Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1 << chunk_pos);
			} else {
				chunk_nr = (number_of_clauses*6 + l) / 32;
				chunk_pos = (number_of_clauses*6 + l) % 32;
				Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1 << chunk_pos);

				chunk_nr = (number_of_clauses*6 + l + number_of_literals/2) / 32;
				chunk_pos = (number_of_clauses*6 + l + number_of_literals/2) % 32;
				Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos);
			}
		}
	}

	unsigned int number_of_consecutive_matches = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		clause_true_consecutive[patch] = number_of_consecutive_matches;
		if (clause_value_in_patch[clause*number_of_patches + patch]) {
			number_of_consecutive_matches++;
		} else {
			number_of_consecutive_matches = 0;
		}
	}

	// number_of_consecutive_matches = 0;
	// for (int patch = number_of_patches-1; patch >= 0; --patch) {
	// 	clause_true_consecutive[patch] += number_of_consecutive_matches;
	// 	if (clause_value_in_patch[clause*number_of_patches + patch]) {
	// 		number_of_consecutive_matches++;
	// 	} else {
	// 		number_of_consecutive_matches = 0;
	// 	}
	// }

	// for (int patch = 0; patch < number_of_patches; ++patch) {
	// 	// Set bits based on how many times true

	// 	for (int l = 0; l < number_of_patches; ++l) {
	// 		if (l < clause_true_consecutive[patch]) { 
	// 			chunk_nr = (number_of_clauses*6 + number_of_patches + l) / 32;
	// 			chunk_pos = (number_of_clauses*6 + number_of_patches + l) % 32;
	// 			Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos);

	// 			chunk_nr = (number_of_clauses*6 + number_of_patches + l + number_of_literals/2) / 32;
	// 			chunk_pos = (number_of_clauses*6 + number_of_patches + l + number_of_literals/2) % 32;
	// 			Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1 << chunk_pos);
	// 		} else {
	// 			chunk_nr = (number_of_clauses*6 + number_of_patches + l) / 32;
	// 			chunk_pos = (number_of_clauses*6 + number_of_patches + l) % 32;
	// 			Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1 << chunk_pos);

	// 			chunk_nr = (number_of_clauses*6 + number_of_patches + l + number_of_literals/2) / 32;
	// 			chunk_pos = (number_of_clauses*6 + number_of_patches + l + number_of_literals/2) % 32;
	// 			Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos);
	// 		}
	// 	}
	// }
}

void cb_type_i_feedback_spatio_temporal(
        unsigned int *ta_state,
        unsigned int *feedback_to_ta,
        unsigned int *output_one_patches,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        float update_p,
        float s,
        unsigned int boost_true_positive_feedback,
        unsigned int reuse_random_feedback,
        unsigned int max_included_literals,
        unsigned int *clause_active,
        unsigned int *literal_active,
        unsigned int *clause_value_in_patch,
        unsigned int *clause_true_consecutive,
        unsigned int *clause_true_consecutive_before,
        unsigned int *clause_false_consecutive_before,
        unsigned int *Xi
)
{
    // Lage mask/filter
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	if (reuse_random_feedback && s > 1.0) {
		cb_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
	}

	for (int j = 0; j < number_of_clauses; ++j) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		// Calculate clause specific features
 	 	cb_calculate_clause_specific_features(
 	 		j,
 	 		number_of_clauses,
 	 		number_of_literals,
 	 		number_of_state_bits,
 	 		number_of_patches,
 	 		clause_value_in_patch,
 	 		clause_true_consecutive,
 	 		Xi
 	 	);

		unsigned int clause_output;
		unsigned int clause_patch;
		cb_calculate_clause_output_feedback(
			&ta_state[clause_pos],
			output_one_patches,
			&clause_output,
			&clause_patch,
			number_of_ta_chunks,
			number_of_state_bits,
			filter,
			number_of_patches,
			literal_active,
			Xi
		);

		if (!reuse_random_feedback && s > 1.0) {
			cb_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
		}

		if (clause_output && cb_number_of_include_actions(ta_state, j, number_of_literals, number_of_state_bits) <= max_included_literals) {
			// Type Ia Feedback
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (boost_true_positive_feedback == 1) {
	 				cb_inc(&ta_state[clause_pos + ta_pos], literal_active[k] & Xi[clause_patch*number_of_ta_chunks + k], number_of_state_bits);
				} else {
					cb_inc(&ta_state[clause_pos + ta_pos], literal_active[k] & Xi[clause_patch*number_of_ta_chunks + k] & (~feedback_to_ta[k]), number_of_state_bits);
				}

				if (s > 1.0) {
		 			cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k] & (~Xi[clause_patch*number_of_ta_chunks + k]) & feedback_to_ta[k], number_of_state_bits);
		 		} else {
		 			cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k] & (~Xi[clause_patch*number_of_ta_chunks + k]), number_of_state_bits);
		 		}
			}
		} else {
			// Type Ib Feedback
				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (s > 1.0) {
					cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k] & feedback_to_ta[k], number_of_state_bits);
				} else {
					cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k], number_of_state_bits);
				}
			}
		}
	}
}

void cb_type_i_feedback(
        unsigned int *ta_state,
        unsigned int *feedback_to_ta,
        unsigned int *output_one_patches,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        float update_p,
        float s,
        unsigned int boost_true_positive_feedback,
        unsigned int reuse_random_feedback,
        unsigned int max_included_literals,
        unsigned int *clause_active,
        unsigned int *literal_active,
        unsigned int *Xi
)
{
    // Lage mask/filter
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	if (reuse_random_feedback && s > 1.0) {
		cb_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
	}

	for (int j = 0; j < number_of_clauses; ++j) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		unsigned int clause_output;
		unsigned int clause_patch;
		cb_calculate_clause_output_feedback(
			&ta_state[clause_pos],
			output_one_patches,
			&clause_output,
			&clause_patch,
			number_of_ta_chunks,
			number_of_state_bits,
			filter,
			number_of_patches,
			literal_active,
			Xi
		);

		if (!reuse_random_feedback && s > 1.0) {
			cb_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
		}

		if (clause_output && cb_number_of_include_actions(ta_state, j, number_of_literals, number_of_state_bits) <= max_included_literals) {
			// Type Ia Feedback
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (boost_true_positive_feedback == 1) {
	 				cb_inc(&ta_state[clause_pos + ta_pos], literal_active[k] & Xi[clause_patch*number_of_ta_chunks + k], number_of_state_bits);
				} else {
					cb_inc(&ta_state[clause_pos + ta_pos], literal_active[k] & Xi[clause_patch*number_of_ta_chunks + k] & (~feedback_to_ta[k]), number_of_state_bits);
				}

				if (s > 1.0) {
		 			cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k] & (~Xi[clause_patch*number_of_ta_chunks + k]) & feedback_to_ta[k], number_of_state_bits);
		 		} else {
		 			cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k] & (~Xi[clause_patch*number_of_ta_chunks + k]), number_of_state_bits);
		 		}
			}
		} else {
			// Type Ib Feedback
				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (s > 1.0) {
					cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k] & feedback_to_ta[k], number_of_state_bits);
				} else {
					cb_dec(&ta_state[clause_pos + ta_pos], literal_active[k], number_of_state_bits);
				}
			}
		}
	}
}

void cb_type_ii_feedback_spatio_temporal(
        unsigned int *ta_state,
        unsigned int *output_one_patches,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        float update_p,
        unsigned int *clause_active,
        unsigned int *literal_active,
        unsigned int *clause_value_in_patch,
        unsigned int *clause_true_consecutive,
        unsigned int *clause_true_consecutive_before,
        unsigned int *clause_false_consecutive_before,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		// Calculate clause specific features
 	 	cb_calculate_clause_specific_features(
 	 		j,
 	 		number_of_clauses,
 	 		number_of_literals,
 	 		number_of_state_bits,
 	 		number_of_patches,
 	 		clause_value_in_patch,
 	 		clause_true_consecutive,
 	 		Xi
 	 	);

		unsigned int clause_output;
		unsigned int clause_patch;
		cb_calculate_clause_output_feedback(&ta_state[clause_pos], output_one_patches, &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, literal_active, Xi);

		if (clause_output) {				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;
				cb_inc(&ta_state[clause_pos + ta_pos], literal_active[k] & (~Xi[clause_patch*number_of_ta_chunks + k]), number_of_state_bits);
			}
		}
	}
}

void cb_type_ii_feedback(
        unsigned int *ta_state,
        unsigned int *output_one_patches,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        float update_p,
        unsigned int *clause_active,
        unsigned int *literal_active,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		unsigned int clause_output;
		unsigned int clause_patch;
		cb_calculate_clause_output_feedback(&ta_state[clause_pos], output_one_patches, &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, literal_active, Xi);

		if (clause_output) {				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;
				cb_inc(&ta_state[clause_pos + ta_pos], literal_active[k] & (~Xi[clause_patch*number_of_ta_chunks + k]), number_of_state_bits);
			}
		}
	}
}

void cb_type_iii_feedback(
        unsigned int *ta_state,
        unsigned int *ind_state,
        unsigned int *clause_and_target,
        unsigned int *output_one_patches,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits_ta,
        int number_of_state_bits_ind,
        int number_of_patches,
        float update_p,
        float d,
        unsigned int *clause_active,
        unsigned int *literal_active,
        unsigned int *Xi,
        unsigned int target
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; ++j) {
		if ((!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos_ta = j*number_of_ta_chunks*number_of_state_bits_ta;
		unsigned int clause_pos_ind = j*number_of_ta_chunks*number_of_state_bits_ind;

		unsigned int clause_output;
		unsigned int clause_patch;
		cb_calculate_clause_output_feedback(
		    &ta_state[clause_pos_ta],
		    output_one_patches,
		    &clause_output,
		    &clause_patch,
		    number_of_ta_chunks,
		    number_of_state_bits_ta,
		    filter,
		    number_of_patches,
		    literal_active,
		    Xi
        	);

		if (clause_output) {
			if (target) {
				if (((float)fast_rand())/((float)FAST_RAND_MAX) <= (1.0 - 1.0/d)) {
					for (int k = 0; k < number_of_ta_chunks; ++k) {

						unsigned int ind_pos = k*number_of_state_bits_ind;
						cb_inc(
						    &ind_state[clause_pos_ind + ind_pos],
						    literal_active[k] & clause_and_target[j * number_of_ta_chunks + k] & Xi[clause_patch * number_of_ta_chunks + k],
						    number_of_state_bits_ind
                        );
					}
				}
			}

			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ind_pos = k*number_of_state_bits_ind;
				// Decrease if clause is true and literal is true
				cb_dec(
                    &ind_state[clause_pos_ind + ind_pos],
                    literal_active[k] & (~clause_and_target[j * number_of_ta_chunks + k]) & Xi[clause_patch*number_of_ta_chunks + k],
                    number_of_state_bits_ind);
			}

			// Invert literals
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int remove;
				if (target) {
				 	remove = clause_and_target[j*number_of_ta_chunks + k];
				} else {
					remove = 0;
				}
				unsigned int add = ~clause_and_target[j*number_of_ta_chunks + k];
				clause_and_target[j*number_of_ta_chunks + k] |= add;
				clause_and_target[j*number_of_ta_chunks + k] &= (~remove);
			}
		}

		// Included
		if (!clause_output) {
			int offending_literal = cb_calculate_clause_output_single_false_literal(&ta_state[clause_pos_ta], output_one_patches, number_of_ta_chunks, number_of_state_bits_ta, filter, number_of_patches, literal_active, Xi);
			if (offending_literal != - 1) {
				unsigned int ta_chunk = offending_literal / 32;
				unsigned int ta_pos = offending_literal % 32;

				if ((clause_and_target[j*number_of_ta_chunks + ta_chunk] & (1 << ta_pos)) == 0) {
					clause_and_target[j*number_of_ta_chunks + ta_chunk] |= (1 << ta_pos);
				} else if (target) {
					clause_and_target[j*number_of_ta_chunks + ta_chunk] &= (~(1 << ta_pos));
				}
			}
		}

		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		for (int k = 0; k < number_of_ta_chunks; ++k) {
			unsigned int ta_pos = k*number_of_state_bits_ta;
			unsigned int ind_pos = k*number_of_state_bits_ind;

			cb_dec(
			    &ta_state[clause_pos_ta + ta_pos],
			    literal_active[k] & (~ind_state[clause_pos_ind + ind_pos + number_of_state_bits_ind - 1]),
			    number_of_state_bits_ta
            		);
		}
	}
}

void cb_identify_temporal_truth_value_transitions(
        int number_of_clauses,
        int number_of_patches,
        unsigned int *clause_value_in_patch,
        unsigned int *clause_true_consecutive_before,
        unsigned int *clause_false_consecutive_before
)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		for (int j = 0; j < number_of_clauses; ++j) {
			if (clause_value_in_patch[j*number_of_patches + patch]) {
				clause_true_consecutive_before[j]++;
				clause_false_consecutive_before[j] = 0;
			} else {
				clause_true_consecutive_before[j] = 0;
				clause_false_consecutive_before[j]++;
			}
		}
	}
}

void cb_calculate_spatio_temporal_features(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_value_in_patch,
        unsigned int *clause_new_value_in_patch,
        unsigned int *clause_true_consecutive,
        unsigned int *clause_true_consecutive_before,
        unsigned int *clause_false_consecutive_before,
        unsigned int *Xi
)
{
	unsigned int chunk_nr;
	unsigned int chunk_pos;

	int number_of_rounds = 3;

	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	// Set all clause literals to True:
	// Calculate spatio-temporal literals, patch by patch
	for (int patch = 0; patch < number_of_patches; ++patch) {
		for (int j = 0; j < number_of_clauses; j++) {
			clause_value_in_patch[j*number_of_patches + patch] = 1;
		}
	}

	for (int round = 0; round < number_of_rounds; ++round) {
		// Calculate spatio-temporal literals, patch by patch
		for (int j = 0; j < number_of_clauses; ++j) {
			// for (int patch = 0; patch < number_of_patches; ++patch) {
			// 	// Just before
			// 	if (patch > 0) {
			// 		if (clause_value_in_patch[j*number_of_patches + (patch-1)]) {
			// 			chunk_nr = j / 32;
			// 			chunk_pos = j % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos); // Sets left clause feature to True

			// 			chunk_nr = (j + number_of_literals / 2) / 32;
			// 			chunk_pos = (j + number_of_literals / 2) % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets left negated clause feature to False
			// 		} else {
			// 			chunk_nr = j / 32;
			// 			chunk_pos = j % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets left clause feature to False

			// 			chunk_nr = (j + number_of_literals / 2) / 32;
			// 			chunk_pos = (j + number_of_literals / 2) % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos); // Sets left negated clause feature to True
			// 		}
	
			// 	}

			// 	// Just after
			// 	if ((patch < number_of_patches-1)) {
			// 		if (clause_value_in_patch[j*number_of_patches + (patch+1)]) {
			// 			chunk_nr = (j + number_of_clauses) / 32;
			// 			chunk_pos = (j + number_of_clauses) % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos); // Sets right clause feature to True

			// 			chunk_nr = (j + number_of_clauses + number_of_literals / 2) / 32;
			// 			chunk_pos = (j + number_of_clauses + number_of_literals / 2) % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets right negated clause feature to False
			// 		} else {
			// 			chunk_nr = (j + number_of_clauses) / 32;
			// 			chunk_pos = (j + number_of_clauses) % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets right clause feature to False

			// 			chunk_nr = (j + number_of_clauses + number_of_literals / 2) / 32;
			// 			chunk_pos = (j + number_of_clauses + number_of_literals / 2) % 32;
			// 			Xi[patch*number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos); // Sets right negated clause feature to True
			// 		}
			// 	}

			// 	// Before
			// 	int clause_value = 0;
			// 	for (int patch_before = 0; patch_before < patch; ++patch_before) {
			// 		if (clause_value_in_patch[j*number_of_patches + patch_before]) {
			// 			clause_value = 1;
			// 			break;
			// 		}
			// 	}

			// 	if (clause_value) {
			// 		chunk_nr = (j + number_of_clauses*2) / 32;
			// 		chunk_pos = (j + number_of_clauses*2) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos); // Sets right clause feature to True

			// 		chunk_nr = (j + number_of_clauses*2 + number_of_literals / 2) / 32;
			// 		chunk_pos = (j + number_of_clauses*2 + number_of_literals / 2) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets right negated clause feature to False
			// 	} else {
			// 		chunk_nr = (j + number_of_clauses*2) / 32;
			// 		chunk_pos = (j + number_of_clauses*2) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets right clause feature to False

			// 		chunk_nr = (j + number_of_clauses*2 + number_of_literals / 2) / 32;
			// 		chunk_pos = (j + number_of_clauses*2 + number_of_literals / 2) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos); // Sets right negated clause feature to True
			// 	}

			// 	// After
			// 	clause_value = 0;
			// 	for (int patch_after = patch+1; patch_after < number_of_patches; ++patch_after) {
			// 		if (clause_value_in_patch[j*number_of_patches + patch_after]) {
			// 			clause_value = 1;
			// 			break;
			// 		}
			// 	}

			// 	if (clause_value) {
			// 		chunk_nr = (j + number_of_clauses*3) / 32;
			// 		chunk_pos = (j + number_of_clauses*3) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] |= (1 << chunk_pos); // Sets right clause feature to True

			// 		chunk_nr = (j + number_of_clauses*3 + number_of_literals / 2) / 32;
			// 		chunk_pos = (j + number_of_clauses*3 + number_of_literals / 2) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets right negated clause feature to False
			// 	} else {
			// 		chunk_nr = (j + number_of_clauses*3) / 32;
			// 		chunk_pos = (j + number_of_clauses*3) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos); // Sets right clause feature to False

			// 		chunk_nr = (j + number_of_clauses*3 + number_of_literals / 2) / 32;
			// 		chunk_pos = (j + number_of_clauses*3 + number_of_literals / 2) % 32;
			// 		Xi[patch*number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos); // Sets right negated clause feature to True
			// 	}
			// }
		}

		if (round > 0) {
			cb_identify_temporal_truth_value_transitions(
			        number_of_clauses,
			        number_of_patches,
			        clause_value_in_patch,
			        clause_true_consecutive_before,
			       	clause_false_consecutive_before
			);
		}

		if (round < number_of_rounds-1) {
			for (int j = 0; j < number_of_clauses; j++) {
				unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits; // Calculates the position of the Tsetlin automata states of the current clause

	 			cb_calculate_clause_specific_features(
	 				j,
	 				number_of_clauses,
	 				number_of_literals,
	 				number_of_state_bits,
	 				number_of_patches,
	 				clause_value_in_patch,
	 	 			clause_true_consecutive,
	 				Xi
	 			);

				for (int patch = 0; patch < number_of_patches; ++patch) {
					clause_new_value_in_patch[j*number_of_patches + patch] = cb_calculate_clause_output_without_literal_active(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, &Xi[patch*number_of_ta_chunks]);
				}
			}

			unsigned int *tmp = clause_value_in_patch;
			clause_value_in_patch = clause_new_value_in_patch;
			clause_new_value_in_patch = tmp;
		} 
	}

	if (!(number_of_rounds % 2)) {
		for (int j = 0; j < number_of_clauses; ++j) {
			for (int patch = 0; patch < number_of_patches; ++patch) {
				clause_value_in_patch[j*number_of_patches + patch] = clause_new_value_in_patch[j*number_of_patches + patch];
			}
		}
	}

	return;
}

void cb_calculate_clause_outputs_predict_spatio_temporal(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *clause_value_in_patch,
        unsigned int *clause_true_consecutive,
        unsigned int *clause_true_consecutive_before,
        unsigned int *clause_false_consecutive_before,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		// Calculate clause specific features
 		cb_calculate_clause_specific_features(
 			j,
 			number_of_clauses,
 			number_of_literals,
 			number_of_state_bits,
 			number_of_patches,
 			clause_value_in_patch,
  	 		clause_true_consecutive,
 			Xi
 		);

		clause_output[j] = cb_calculate_clause_output_predict(
			&ta_state[clause_pos],
			number_of_ta_chunks,
			number_of_state_bits,
			filter,
			number_of_patches,
			Xi
		);
	}
}

void cb_calculate_clause_outputs_predict(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		clause_output[j] = cb_calculate_clause_output_predict(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);
	}
}

void cb_initialize_incremental_clause_calculation(
        unsigned int *ta_state,
        unsigned int *literal_clause_map,
        unsigned int *literal_clause_map_pos,
        unsigned int *false_literals_per_clause,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *previous_Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	// Initialize all literals as false for the previous example per patch
	for (int k = 0; k < number_of_ta_chunks; ++k) {
		previous_Xi[k] = 0;
	}

	// Initialize all false literal counters to 0 per patch
	for (int j = 0; j < number_of_clauses; ++j) {
		false_literals_per_clause[j] = 0;
	}

	// Build the literal clause map, and update the false literal counters
	// Start filling out the map from position 0
	unsigned int pos = 0;
	for (int k = 0; k < number_of_literals; ++k) {
		unsigned int ta_chunk = k / 32;
		unsigned int chunk_pos = k % 32;

		// For each literal, find out which clauses includes it
		for (int j = 0; j < number_of_clauses; ++j) {	
			// Obtain the clause ta chunk containing the literal decision (exclude/include)
			unsigned int clause_ta_chunk = j * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits + number_of_state_bits - 1;
			if (ta_state[clause_ta_chunk] & (1 << chunk_pos)) {
				// Literal k included in clause j
				literal_clause_map[pos] = j;

				++false_literals_per_clause[j];
				++pos;
			}
		}
		literal_clause_map_pos[k] = pos;
	}

	// Make empty clauses false
	for (int j = 0; j < number_of_clauses; ++j) {
		if (false_literals_per_clause[j] == 0) {
			false_literals_per_clause[j] = 1;
		}
	}
}

// This function retrieves the count of literals from the given Tsetlin Automaton state.
// ta_state: an array representing the state of the Tsetlin Automaton.
// number_of_clauses: the total number of clauses in the TA.
// number_of_literals: the total number of literals in the TA.
// number_of_state_bits: the number of bits used to represent each state in the TA.
// result: an array to store the count of each literal.
void cb_get_literals(
    const unsigned int *ta_state,
    unsigned int number_of_clauses,
    unsigned int number_of_literals,
    unsigned int number_of_state_bits,
    unsigned int *result
){
    // Calculate the number of chunks required to represent all literals.
    unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

    // Iterate through all the clauses.
    for (unsigned int j = 0; j < number_of_clauses; j++) {
        // Iterate through all the literals.
        for (unsigned int k = 0; k < number_of_literals; k++) {

            // Determine which chunk the literal is in and its position within the chunk.
            unsigned int ta_chunk = k / 32;
            unsigned int chunk_pos = k % 32;

            // Calculate the position of the literal in the TA state array.
            unsigned int pos = j * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits + number_of_state_bits-1;

            // Check if the literal is present (bit is set) in the TA state array.
            if ((ta_state[pos] & (1 << chunk_pos)) > 0) {
                // Increment the count of the literal in the result array.
                unsigned int result_pos = j * number_of_literals + k;
                result[result_pos] = 1;
            }
        }
    }
}

void cb_calculate_clause_outputs_incremental_batch(
        unsigned int * literal_clause_map,
        unsigned int *literal_clause_map_pos,
        unsigned int *false_literals_per_clause,
        int number_of_clauses,
        int number_of_literals,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *previous_Xi,
        unsigned int *Xi,
        int batch_size
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	unsigned int *current_Xi = Xi;
	for (int b = 0; b < batch_size; ++b) {
		for (int j = 0; j < number_of_clauses; ++j) {
			clause_output[b*number_of_clauses + j] = 0;
		}

		for (int patch = 0; patch < number_of_patches; ++patch) {
			cb_calculate_clause_outputs_incremental(literal_clause_map, literal_clause_map_pos, false_literals_per_clause, number_of_clauses, number_of_literals, previous_Xi, current_Xi);
			for (int j = 0; j < number_of_clauses; ++j) {
				if (false_literals_per_clause[j] == 0) {
					clause_output[b*number_of_clauses + j] = 1;
				}
			}
			current_Xi += number_of_ta_chunks;
		}
	}
}

void cb_calculate_clause_outputs_incremental(
        unsigned int * literal_clause_map,
        unsigned int *literal_clause_map_pos,
        unsigned int *false_literals_per_clause,
        int number_of_clauses,
        int number_of_literals,
        unsigned int *previous_Xi,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	// Look up each in literal clause map
	unsigned int start_pos = 0;
	for (int k = 0; k < number_of_literals; ++k) {
		unsigned int ta_chunk = k / 32;
		unsigned int chunk_pos = k % 32;

		// Check which literals have changed
		if ((Xi[ta_chunk] & (1 << chunk_pos)) && !(previous_Xi[ta_chunk] & (1 << chunk_pos))) {
			// If the literal now is True, decrement the false literal counter of all clauses including the literal
			for (int j = 0; j < literal_clause_map_pos[k] - start_pos; ++j) {
				--false_literals_per_clause[literal_clause_map[start_pos + j]];
			}
		} else if (!(Xi[ta_chunk] & (1 << chunk_pos)) && (previous_Xi[ta_chunk] & (1 << chunk_pos))) {
			// If the literal now is False, increment the false counter of all clauses including literal
			for (int j = 0; j < literal_clause_map_pos[k] - start_pos; ++j) {
				++false_literals_per_clause[literal_clause_map[start_pos + j]];
			}
		}

		start_pos = literal_clause_map_pos[k];
	}

	// Copy current Xi to previous_Xi
	for (int k = 0; k < number_of_ta_chunks; ++k) {
		previous_Xi[k] = Xi[k];
	}
}

void cb_calculate_clause_outputs_update_spatio_temporal(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *literal_active,
        unsigned int *clause_value_in_patch,
        unsigned int *clause_true_consecutive,
        unsigned int *clause_true_consecutive_before,
        unsigned int *clause_false_consecutive_before,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		// Calculate clause specific features
 		cb_calculate_clause_specific_features(
 			j,
 			number_of_clauses,
 			number_of_literals,
 			number_of_state_bits,
 			number_of_patches,
 			clause_value_in_patch,
  	 		clause_true_consecutive,
 			Xi
 		);

		clause_output[j] = cb_calculate_clause_output_update(
			&ta_state[clause_pos],
			number_of_ta_chunks,
			number_of_state_bits,
			filter,
			number_of_patches,
			literal_active,
			Xi
		);
	}
}

void cb_calculate_clause_outputs_update(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *literal_active,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		clause_output[j] = cb_calculate_clause_output_update(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, literal_active, Xi);
	}
}

void cb_calculate_clause_outputs_patchwise(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        int number_of_patches,
        unsigned int *clause_output,
        unsigned int *Xi
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		cb_calculate_clause_output_patchwise(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, &clause_output[j*number_of_patches], Xi);
	}
}

void cb_calculate_literal_frequency(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *clause_active,
        unsigned int *literal_count
)
{
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int k = 0; k < number_of_literals; k++) {
		literal_count[k] = 0;
	}
	
	for (int j = 0; j < number_of_clauses; j++) {
		if ((!clause_active[j])) {
			continue;
		}

		for (int k = 0; k < number_of_literals; k++) {
			unsigned int ta_chunk = k / 32;
			unsigned int chunk_pos = k % 32;
			unsigned int pos = j * number_of_ta_chunks * number_of_state_bits + ta_chunk * number_of_state_bits + number_of_state_bits-1;
			if ((ta_state[pos] & (1 << chunk_pos)) > 0) {
				literal_count[k] += 1;
			}
		}
	}
}

void cb_included_literals(
        unsigned int *ta_state,
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *actions
)
{
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

	for (int k = 0; k < number_of_ta_chunks; k++) {
		actions[k] = 0;
	}
	
	for (int j = 0; j < number_of_clauses; j++) {	
		for (int k = 0; k < number_of_ta_chunks; k++) {
			unsigned int pos = j * number_of_ta_chunks * number_of_state_bits + k * number_of_state_bits + number_of_state_bits-1;
			actions[k] |= ta_state[pos];
		}
	}
}

int cb_number_of_include_actions(
        unsigned int *ta_state,
        int clause,
        int number_of_literals,
        int number_of_state_bits
)
{
	unsigned int filter;
	if (((number_of_literals) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_literals) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;
	
	unsigned int clause_pos = clause*number_of_ta_chunks*number_of_state_bits;

	int number_of_include_actions = 0;
	for (int k = 0; k < number_of_ta_chunks-1; ++k) {
		unsigned int ta_pos = k*number_of_state_bits + number_of_state_bits-1;
		number_of_include_actions += __builtin_popcount(ta_state[clause_pos + ta_pos]);
	}
	unsigned int ta_pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
	number_of_include_actions += __builtin_popcount(ta_state[clause_pos + ta_pos] & filter);

	return(number_of_include_actions);
}
