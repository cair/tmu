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

void wb_increment(int *clause_weights, int number_of_clauses, unsigned int *clause_output, float update_p, unsigned int *clause_active, unsigned int positive_weights)
{
	for (int j = 0; j < number_of_clauses; ++j) {
		if (clause_active[j] && clause_output[j] && (positive_weights || (clause_weights[j] != -1)) && (((float)fast_rand())/((float)FAST_RAND_MAX) <= update_p)) {
			clause_weights[j]++;
		}
	}
}

void wb_decrement(int *clause_weights, int number_of_clauses, unsigned int *clause_output, float update_p, unsigned int *clause_active, unsigned int negative_weights)
{
	for (int j = 0; j < number_of_clauses; j++) {
		if (clause_active[j] && clause_output[j] && (negative_weights || (clause_weights[j] != 1)) && (((float)fast_rand())/((float)FAST_RAND_MAX) <= update_p)) {
			clause_weights[j]--;
		}
	}
}
