/*

Copyright (c) 2019 Ole-Christoffer Granmo

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
#include <string.h>
#include <stdlib.h>

unsigned int compareints(const void * a, const void * b)
{
  return(*(unsigned int*)a - *(unsigned int*)b);
}

void tmu_produce_autoencoder_examples(unsigned int *active_output, int number_of_active_outputs, unsigned int *indptr_row, unsigned int *indices_row, int number_of_rows, unsigned int *indptr_col, unsigned int *indices_col, int number_of_cols, unsigned int *encoded_X, unsigned int *Y, int accumulation, int append_negated)
{
	int row;

	int number_of_ta_chunks;
	if (append_negated) {
		number_of_ta_chunks= (((2*number_of_cols-1)/32 + 1));
	} else {
		number_of_ta_chunks= (((number_of_cols-1)/32 + 1));
	}

	// Loop over active outputs, producing one example per output
	for (int o = 0; o < number_of_active_outputs; ++o) {
		int output_pos = o*number_of_ta_chunks;

		// Initialize all example literals to zero
		for (int k = 0; k < number_of_ta_chunks; ++k) {
			encoded_X[output_pos + k] = 0;
		}

		if ((indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == 0) || (indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == number_of_rows)) {
			printf("No examples: %d\n", active_output[o]);
			// If no positive/negative examples, produce a random example
			for (int a = 0; a < accumulation; ++a) {
				row = rand() % number_of_rows;
				for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
					int chunk_nr = indices_row[k] / 32;
					int chunk_pos = indices_row[k] % 32;
					encoded_X[output_pos + chunk_nr] |= (1 << chunk_pos);
				}
			}
		}

		if (indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == 0) {
			// If no positive examples, produce a negative one
			Y[o] = 0;
			continue;
		} else if (indptr_col[active_output[o]+1] - indptr_col[active_output[o]] == number_of_rows) {
			// If no negative examples, produce a positive one
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
					int chunk_nr = indices_row[k] / 32;
					int chunk_pos = indices_row[k] % 32;
					encoded_X[output_pos + chunk_nr] |= (1 << chunk_pos);
				}
			}
		} else {
			int a = 0;
			while (a < accumulation) {
				row = rand() % number_of_rows;

				if (bsearch(&row, &indices_col[active_output[o]], indptr_col[active_output[o]+1] - indptr_col[active_output[o]], sizeof(unsigned int), compareints) == NULL) {
					for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
						int chunk_nr = indices_row[k] / 32;
						int chunk_pos = indices_row[k] % 32;
						encoded_X[output_pos + chunk_nr] |= (1 << chunk_pos);
					}
					a++;
				}
			}
		}
	}
}

void tmu_encode(unsigned int *X, unsigned int *encoded_X, int number_of_examples, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
{
	int global_number_of_features = dim_x * dim_y * dim_z;
	int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
	int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

	int number_of_ta_chunks;
	if (append_negated) {
		number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
	} else {
		number_of_ta_chunks= (((number_of_features-1)/32 + 1));
	}

	unsigned int *Xi;
	unsigned int *encoded_Xi;

	unsigned int input_pos = 0;
	unsigned int input_step_size = global_number_of_features;

	// Fill encoded_X with zeros

	memset(encoded_X, 0, number_of_examples * number_of_patches * number_of_ta_chunks * sizeof(unsigned int));

	unsigned int encoded_pos = 0;
	for (int i = 0; i < number_of_examples; ++i) {
		//printf("%d\n", i);

		int patch_nr = 0;
		// Produce the patches of the current image
		for (int y = 0; y < dim_y - patch_dim_y + 1; ++y) {
			for (int x = 0; x < dim_x - patch_dim_x + 1; ++x) {
				Xi = &X[input_pos];
				encoded_Xi = &encoded_X[encoded_pos];

				// Encode class into feature vector 
				for (int class_feature = 0; class_feature < class_features; ++class_feature) {

					int chunk_nr = (class_feature + number_of_features) / 32;
					int chunk_pos = (class_feature + number_of_features) % 32;
					encoded_Xi[chunk_nr] |= (1 << chunk_pos);
				}

				// Encode y coordinate of patch into feature vector 
				for (int y_threshold = 0; y_threshold < dim_y - patch_dim_y; ++y_threshold) {
					int patch_pos = class_features + y_threshold;

					if (y > y_threshold) {
						int chunk_nr = patch_pos / 32;
						int chunk_pos = patch_pos % 32;
						encoded_Xi[chunk_nr] |= (1 << chunk_pos);
					} else if (append_negated) {
						int chunk_nr = (patch_pos + number_of_features) / 32;
						int chunk_pos = (patch_pos + number_of_features) % 32;
						encoded_Xi[chunk_nr] |= (1 << chunk_pos);
					}
				}

				// Encode x coordinate of patch into feature vector
				for (int x_threshold = 0; x_threshold < dim_x - patch_dim_x; ++x_threshold) {
					int patch_pos = class_features + (dim_y - patch_dim_y) + x_threshold;

					if (x > x_threshold) {
						int chunk_nr = patch_pos / 32;
						int chunk_pos = patch_pos % 32;

						encoded_Xi[chunk_nr] |= (1 << chunk_pos);
					} else if (append_negated) {
						int chunk_nr = (patch_pos + number_of_features) / 32;
						int chunk_pos = (patch_pos + number_of_features) % 32;
						encoded_Xi[chunk_nr] |= (1 << chunk_pos);
					}
				} 

				// Encode patch content into feature vector
				for (int p_y = 0; p_y < patch_dim_y; ++p_y) {
					for (int p_x = 0; p_x < patch_dim_x; ++p_x) {
						for (int z = 0; z < dim_z; ++z) {
							int image_pos = (y + p_y)*dim_x*dim_z + (x + p_x)*dim_z + z;
							int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

							if (Xi[image_pos] == 1) {
								int chunk_nr = patch_pos / 32;
								int chunk_pos = patch_pos % 32;
								encoded_Xi[chunk_nr] |= (1 << chunk_pos);
							} else if (append_negated) {
								int chunk_nr = (patch_pos + number_of_features) / 32;
								int chunk_pos = (patch_pos + number_of_features) % 32;
								encoded_Xi[chunk_nr] |= (1 << chunk_pos);
							}
						}
					}
				}
				encoded_pos += number_of_ta_chunks;
				patch_nr++;
			}
		}
		input_pos += input_step_size;
	}
}