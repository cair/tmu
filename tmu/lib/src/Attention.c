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

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

void at_get_attention(
        unsigned int *ranking,
        int number_of_literals,
        unsigned int attention_span,
        unsigned int *included_literals,
        unsigned int *Xi
)
{
	for (int k = attention_span; k < number_of_literals; ++k) {
		int chunk = ranking[k] / 32;
		int pos = ranking[k] % 32;

		if (!(included_literals[chunk] & (1 << pos))) {
			Xi[chunk] &= ~(1 << pos);
		}
	}
}

void at_type_i_feedback(
        unsigned int *ranking,
        int number_of_literals,
        float update_p,
        float s,
        unsigned int *Xi
)
{
	if (((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) {
		return;
	}

	for (int k = 0; k < number_of_literals; ++k) {
		int chunk = ranking[k] / 32;
		int pos = ranking[k] % 32;

		if (Xi[chunk] & (1 << pos)) {
			if (k > 0) {
				unsigned int tmp = ranking[k-1];
                ranking[k-1] = ranking[k];
                ranking[k] = tmp;
			}
		} else {
			if (k < number_of_literals - 1 && ((float)fast_rand())/((float)FAST_RAND_MAX) <= 1.0/s) {
				unsigned int tmp = ranking[k+1];
               	ranking[k+1] = ranking[k];
                ranking[k] = tmp;
			} 
		}
	}                    
}

void at_type_ii_feedback(
        unsigned int *ranking,
        int number_of_literals,
        float update_p,
        unsigned int *Xi
)
{
	if (((float)fast_rand())/((float)FAST_RAND_MAX) > update_p) {
		return;
	}

	for (int k = 0; k < number_of_literals; ++k) {
		int chunk = ranking[k] / 32;
		int pos = ranking[k] % 32;

		if ((!(Xi[chunk] & (1 << pos))) && k > 0) {
			unsigned int tmp = ranking[k-1];
            ranking[k-1] = ranking[k];
            ranking[k] = tmp;
		}
  	}
}