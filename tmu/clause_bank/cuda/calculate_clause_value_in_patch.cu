/***
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
***/

#include <curand_kernel.h>

extern "C"
{
    __global__ void calculate_clause_value_in_patch(
        int number_of_clauses,
        int number_of_literals,
        int number_of_state_bits,
        unsigned int *global_ta_state,
        int *global_clause_node_output,
        int *global_clause_node_output_next,
        unsigned int *literal_active,
        unsigned int *X
    )
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;


        if (index != 0) {
            return;
        }

        unsigned int clause_node_output;

        unsigned int filter;
        if (((number_of_literals) % 32) != 0) {
            filter  = (~(0xffffffff << ((number_of_literals) % 32)));
        } else {
            filter = 0xffffffff;
        }
        unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

        int number_of_node_chunks = (NUMBER_OF_PATCHES - 1)/32 + 1;
        unsigned int node_filter;
        if ((NUMBER_OF_PATCHES % 32) != 0) {
            node_filter = (~(0xffffffff << (NUMBER_OF_PATCHES % 32)));
        } else {
            node_filter = 0xffffffff;
        }

        for (int clause_node_chunk = 0; clause_node_chunk < (number_of_clauses)*(number_of_node_chunks); clause_node_chunk += 1) {
            int clause = clause_node_chunk / number_of_node_chunks;
            int node_chunk = clause_node_chunk % number_of_node_chunks;

            unsigned int *ta_state = &global_ta_state[clause*number_of_ta_chunks*number_of_state_bits];

            if (clause == 0) {
                prinf("* ");
                for (int k = 0; k < number_of_literals; ++k) {
                    int literal_chunk = k / 32;
                    int literal_pos = k % 32;

                    if (ta_state[literal_chunk*number_of_state_bits + number_of_state_bits - 1] & (1 << literal_pos)) {
                        printf("%d", k);
                    }
                }
                printf("\n");
            }

            clause_node_output = ~0;
            for (int node_pos = 0; (node_pos < 32) && ((node_chunk * 32 + node_pos) < NUMBER_OF_PATCHES); ++node_pos) {
                int node = node_chunk * 32 + node_pos;

                for (int la_chunk = 0; la_chunk < number_of_ta_chunks-1; ++la_chunk) {
                    if ((ta_state[la_chunk*number_of_state_bits + number_of_state_bits - 1] & (X[node*number_of_ta_chunks + la_chunk] | (!literal_active[la_chunk]))) != ta_state[la_chunk*number_of_state_bits + number_of_state_bits - 1]) {
                        clause_node_output &= ~(1 << node_pos);
                    }
                }

                if ((ta_state[(number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits - 1] & (X[node*number_of_ta_chunks + number_of_ta_chunks-1] | (!literal_active[number_of_ta_chunks-1])) & filter) != (ta_state[(number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits - 1] & filter)) {
                    clause_node_output &= ~(1 << node_pos);
                }
            }

            if (node_chunk == number_of_node_chunks - 1) {
                global_clause_node_output[clause*number_of_node_chunks + node_chunk] = clause_node_output & node_filter;
            } else {
                global_clause_node_output[clause*number_of_node_chunks + node_chunk] = clause_node_output;
            }
        }
    }
}