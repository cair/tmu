# Copyright (c) 2021 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

code_calculate_clause_outputs_predict = """
	#include <curand_kernel.h>

	extern "C"
    {
		__device__ inline unsigned int calculate_clause_output_predict(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, unsigned int *Xi)
		{
			for (int patch = 0; patch < NUMBER_OF_PATCHES; ++patch) {
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

		__global__ void calculate_clause_outputs_predict(unsigned int *ta_state, int number_of_clauses, int number_of_literals, int number_of_state_bits, unsigned int *clause_output, unsigned int *X, int e)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			unsigned int filter;
			if (((number_of_literals) % 32) != 0) {
				filter  = (~(0xffffffff << ((number_of_literals) % 32)));
			} else {
				filter = 0xffffffff;
			}
			unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

			for (int j = index; j < number_of_clauses; j += stride) {
				unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
				clause_output[j] = calculate_clause_output_predict(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, &X[e*(number_of_ta_chunks*NUMBER_OF_PATCHES)]);
			}
		}
	}
"""

code_clause_feedback = """
	#include <curand_kernel.h>

	extern "C"
    {
    	// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void inc(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
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
		__device__ inline void dec(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
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
		__device__ inline void calculate_clause_output_feedback(curandState *localState, unsigned int *ta_state, unsigned int *clause_output, unsigned int *clause_patch, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, unsigned int *Xi)
		{	
			unsigned int output_one_patches[NUMBER_OF_PATCHES];

			int output_one_patches_count = 0;
			for (int patch = 0; patch < NUMBER_OF_PATCHES; ++patch) {
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
				int patch_id = curand(localState) % output_one_patches_count;
		 		*clause_patch = output_one_patches[patch_id];
			} else {
				*clause_output = 0;
			}
		}

		__global__ void type_i_feedback(curandState *state, unsigned int *ta_state, int number_of_clauses, int number_of_literals, int number_of_state_bits, float update_p, float s, unsigned int boost_true_positive_feedback, unsigned int *clause_active, unsigned int *X, int e)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];

			unsigned int filter;
			if (((number_of_literals) % 32) != 0) {
				filter  = (~(0xffffffff << ((number_of_literals) % 32)));
			} else {
				filter = 0xffffffff;
			}
			unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

			unsigned int *Xi = &X[e*(number_of_ta_chunks*NUMBER_OF_PATCHES)];

			for (int j = index; j < number_of_clauses; j += stride) {
				if ((curand_uniform(&localState) > update_p) || (!clause_active[j])) {
					continue;
				}

				unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

				unsigned int clause_output;
				unsigned int clause_patch;
				calculate_clause_output_feedback(&localState, &ta_state[clause_pos], &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, Xi);

				for (int k = 0; k < number_of_ta_chunks; ++k) {
					// Generate random bit values
					unsigned int feedback_to_ta = 0;
					for (int b = 0; b < 32; ++b) {
						if (curand_uniform(&localState) <= 1.0/s) {
							feedback_to_ta |= (1 << b);
						}
					}

					unsigned int ta_pos = k*number_of_state_bits;

					if (clause_output) {
						// Type Ia Feedback

						if (boost_true_positive_feedback == 1) {
			 				inc(&ta_state[clause_pos + ta_pos], Xi[clause_patch*number_of_ta_chunks + k], number_of_state_bits);
						} else {
							inc(&ta_state[clause_pos + ta_pos], Xi[clause_patch*number_of_ta_chunks + k] & (~feedback_to_ta), number_of_state_bits);
						}

		 				dec(&ta_state[clause_pos + ta_pos], (~Xi[clause_patch*number_of_ta_chunks + k]) & feedback_to_ta, number_of_state_bits);
					} else {
						// Type Ib Feedback
						
						dec(&ta_state[clause_pos + ta_pos], feedback_to_ta, number_of_state_bits);
					}
				}
			}

			state[index] = localState;
		}

		__global__ void type_ii_feedback(curandState *state, unsigned int *ta_state, int number_of_clauses, int number_of_literals, int number_of_state_bits, float update_p, unsigned int *clause_active, unsigned int *X, int e)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			curandState localState = state[index];

			unsigned int filter;
			if (((number_of_literals) % 32) != 0) {
				filter  = (~(0xffffffff << ((number_of_literals) % 32)));
			} else {
				filter = 0xffffffff;
			}
			unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

			unsigned int *Xi = &X[e*(number_of_ta_chunks*NUMBER_OF_PATCHES)];

			for (int j = index; j < number_of_clauses; j += stride) {
				if ((curand_uniform(&localState) > update_p) || (!clause_active[j])) {
					continue;
				}

				unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;

				unsigned int clause_output;
				unsigned int clause_patch;
				calculate_clause_output_feedback(&localState, &ta_state[clause_pos], &clause_output, &clause_patch, number_of_ta_chunks, number_of_state_bits, filter, Xi);

				if (clause_output) {				
					for (int k = 0; k < number_of_ta_chunks; ++k) {
						unsigned int ta_pos = k*number_of_state_bits;
						inc(&ta_state[clause_pos + ta_pos], (~Xi[clause_patch*number_of_ta_chunks + k]) & (~ta_state[clause_pos + ta_pos + number_of_state_bits - 1]), number_of_state_bits);
					}
				}
			}

			state[index] = localState;
		}
    }
"""

code_calculate_clause_outputs_update = """
	#include <curand_kernel.h>

	extern "C"
    {
		__device__ inline unsigned int calculate_clause_output_update(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, unsigned int *Xi)
		{
			for (int patch = 0; patch < NUMBER_OF_PATCHES; ++patch) {
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

		__global__ void calculate_clause_outputs_update(unsigned int *ta_state, int number_of_clauses, int number_of_literals, int number_of_state_bits, unsigned int *clause_output, unsigned int *X, int e)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			unsigned int filter;
			if (((number_of_literals) % 32) != 0) {
				filter  = (~(0xffffffff << ((number_of_literals) % 32)));
			} else {
				filter = 0xffffffff;
			}
			unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

			for (int j = index; j < number_of_clauses; j += stride) {
				unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
				clause_output[j] = calculate_clause_output_update(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, &X[e*(number_of_ta_chunks*NUMBER_OF_PATCHES)]);
			}
		}
	}
"""