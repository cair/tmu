/*

Copyright (c) 2021 Ole-Christoffer Granmo

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

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

static inline void cb_initialize_random_streams(unsigned int *feedback_to_ta, int number_of_features, int number_of_ta_chunks, float s)
{
	// Initialize all bits to zero	
	memset(feedback_to_ta, 0, number_of_ta_chunks*sizeof(unsigned int));

	int n = number_of_features;
	float p = 1.0 / s;

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % (number_of_features);
		while (feedback_to_ta[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % (number_of_features);
	    }
		feedback_to_ta[f / 32] |= 1 << (f % 32);
	}
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cb_inc(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
{
	unsigned int carry, carry_next;

	carry = active;
	for (int b = 0; b < number_of_state_bits; ++b) {
		if (carry == 0)
			break;

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
static inline void cb_dec(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
{
	unsigned int carry, carry_next;

	carry = active;
	for (int b = 0; b < number_of_state_bits; ++b) {
		if (carry == 0)
			break;

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

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void cb_calculate_clause_output_feedback(unsigned int *ta_state, unsigned int *output_one_patches, unsigned int *clause_output, unsigned int *clause_patch, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
{		
	int output_one_patches_count = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);

		if (output) {
			output_one_patches[output_one_patches_count] = patch;
			output_one_patches_count++;
		}
	}

	if (output_one_patches_count > 0) {
		*clause_output = 1;
		unsigned int patch_id = fast_rand() % output_one_patches_count;
 		*clause_patch = output_one_patches[patch_id];
	} else {
		*clause_output = 0;
	}
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void cb_calculate_clause_output_feedback_test(unsigned int *ta_state, unsigned int *output_one_patches, unsigned int *clause_output, unsigned int *clause_patch, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi, unsigned int random_integer)
{		
	int output_one_patches_count = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);

		if (output) {
			output_one_patches[output_one_patches_count] = patch;
			output_one_patches_count++;
		}
	}

	if (output_one_patches_count > 0) {
		*clause_output = 1;
		unsigned int patch_id = random_integer % output_one_patches_count;
 		*clause_patch = output_one_patches[patch_id];
	} else {
		*clause_output = 0;
	}
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void cb_calculate_clause_output_feedback_old(unsigned int *ta_state, unsigned int *output_one_patches, unsigned int *clause_output, unsigned int *clause_patch, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
{
	int output_one_patches_count = 0;
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
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

static inline unsigned int cb_calculate_clause_output_update(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
{
	for (int patch = 0; patch < number_of_patches; ++patch) {
		unsigned int output = 1;
		for (int k = 0; k < number_of_ta_chunks-1; k++) {
			unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
			output = output && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos];

			if (!output) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output = output &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
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

static inline unsigned int cb_calculate_clause_output_predict(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, int number_of_patches, unsigned int *Xi)
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

void cb_type_i_feedback(unsigned int *ta_state, unsigned int *feedback_to_ta, unsigned int *output_one_patches, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, float update_p, float s, unsigned int boost_true_positive_feedback, unsigned int *clause_active, unsigned int *Xi)
{
	unsigned int filter;
	if (((number_of_features) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_features-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; ++j) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		unsigned int clause_output;
		unsigned int clause_patch;

		cb_calculate_clause_output_feedback(&ta_state[clause_pos], output_one_patches, &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);

		cb_initialize_random_streams(feedback_to_ta, number_of_features, number_of_ta_chunks, s);

		if (clause_output) {
			// Type Ia Feedback
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				if (boost_true_positive_feedback == 1) {
	 				cb_inc(&ta_state[clause_pos + ta_pos], Xi[clause_patch*number_of_ta_chunks + k], number_of_state_bits);
				} else {
					cb_inc(&ta_state[clause_pos + ta_pos], Xi[clause_patch*number_of_ta_chunks + k] & (~feedback_to_ta[k]), number_of_state_bits);
				}

	 			cb_dec(&ta_state[clause_pos + ta_pos], (~Xi[clause_patch*number_of_ta_chunks + k]) & feedback_to_ta[k], number_of_state_bits);
			}
		} else {
			// Type Ib Feedback
				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;

				cb_dec(&ta_state[clause_pos + ta_pos], feedback_to_ta[k], number_of_state_bits);
			}
		}
	}
}

void cb_type_ii_feedback(unsigned int *ta_state, unsigned int *output_one_patches, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, float update_p, unsigned int *clause_active, unsigned int *Xi)
{
	unsigned int filter;
	if (((number_of_features) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_features-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		if ((((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) || (!clause_active[j])) {
			continue;
		}

		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

		unsigned int clause_output;
		unsigned int clause_patch;
		cb_calculate_clause_output_feedback(&ta_state[clause_pos], output_one_patches, &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);

		if (clause_output) {				
			for (int k = 0; k < number_of_ta_chunks; ++k) {
				unsigned int ta_pos = k*number_of_state_bits;
				cb_inc(&ta_state[clause_pos + ta_pos], (~Xi[clause_patch*number_of_ta_chunks + k]) & (~ta_state[clause_pos + ta_pos + number_of_state_bits - 1]), number_of_state_bits);
			}
		}
	}
}

void cb_calculate_clause_outputs_predict(unsigned int *ta_state, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, unsigned int *clause_output, unsigned int *Xi)
{
	unsigned int filter;
	if (((number_of_features) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		filter = 0xffffffff;
	}
	unsigned int number_of_ta_chunks = (number_of_features-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		clause_output[j] = cb_calculate_clause_output_predict(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);
	}
}

void cb_calculate_clause_outputs_update(unsigned int *ta_state, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, unsigned int *clause_output, unsigned int *Xi)
{
	unsigned int filter;
	if (((number_of_features) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_features-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		clause_output[j] = cb_calculate_clause_output_update(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, Xi);
	}
}

void cb_calculate_clause_outputs_patchwise(unsigned int *ta_state, int number_of_clauses, int number_of_features, int number_of_state_bits, int number_of_patches, unsigned int *clause_output, unsigned int *Xi)
{
	unsigned int filter;
	if (((number_of_features) % 32) != 0) {
		filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		filter = 0xffffffff;
	}

	unsigned int number_of_ta_chunks = (number_of_features-1)/32 + 1;

	for (int j = 0; j < number_of_clauses; j++) {
		unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
		cb_calculate_clause_output_patchwise(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, number_of_patches, &clause_output[j*number_of_patches], Xi);
	}
}

