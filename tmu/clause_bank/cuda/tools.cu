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

#include <curand_kernel.h>


__device__ bsearch(int index, int *indices, int nunber_of_elements)
{
  return NULL;
}

extern "C"
{
	__global__ void tmu_produce_autoencoder_examples(unsigned int *active_output, int number_of_active_outputs, unsigned int *indptr_row, unsigned int *indices_row, int number_of_rows, unsigned int *indptr_col, unsigned int *indices_col, int number_of_cols, unsigned int *X, unsigned int *Y, int accumulation)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		int row;

		int number_of_features = number_of_cols;

		// Loop over active outputs, producing one example per output
		for (int o = index; o < number_of_active_outputs; o += stride) {
			int output_pos = o*number_of_features;

			for (int k = 0; k < number_of_features; ++k) {
					X[output_pos + k] = 0;
			}

			if ((indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == 0) || (indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == number_of_rows)) {
				// If no positive/negative examples, produce a random example
				for (int a = 0; a < accumulation; ++a) {
					row = rand() % number_of_rows;
					for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
						X[output_pos + indices_row[k]] = 1;
					}
				}
			}

			if (indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == 0) {
				// If no positive examples, produce a negative output value
				Y[o] = 0;
				continue;
			} else if (indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == number_of_rows) {
				// If no negative examples, produce a positive output value
				Y[o] = 1;
				continue;
			} 
			
			// Randomly select either positive or negative example
			Y[o] = rand() % 2;
		
			if (Y[o]) {
				for (int a = 0; a < accumulation; ++a) {
					// Pick example randomly among positive examples
					int random_index = indptr_col[active_output[o]] + (rand() % (indptr_col[active_output[o]+1] - indptr_col[active_output[o]]));
					row = indices_col[random_index];
					
					for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
						X[output_pos + indices_row[k]] = 1;
					}
				}
			} else {
				int a = 0;
				while (a < accumulation) {
					row = rand() % number_of_rows;

					if (bsearch(&row, &indices_col[indptr_col[active_output[o]]], indptr_col[active_output[o]+1] - indptr_col[active_output[o]], sizeof(unsigned int), compareints) == NULL) {
						for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
							X[output_pos + indices_row[k]] = 1;
						}
						a++;
					}
				}
			}
		}
	}
}

