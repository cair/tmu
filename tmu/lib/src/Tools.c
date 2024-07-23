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

#include <string.h>
#include <stdlib.h>

int compareints(const void *a, const void *b) {
    const unsigned int *ia = (const unsigned int *)a;
    const unsigned int *ib = (const unsigned int *)b;
    if (*ia < *ib) return -1;
    if (*ia > *ib) return 1;
    return 0;
}


void tmu_produce_autoencoder_example(
        const unsigned int *active_output,
        int number_of_active_outputs,
        const unsigned int *indptr_row,
        const unsigned int *indices_row,
        int number_of_rows,
        unsigned int *indptr_col,
        unsigned int *indices_col,
        int number_of_cols,
        unsigned int *X,
        int target,
        int target_value,
        int accumulation
)
{
	int row;

	int number_of_features = number_of_cols;
	int number_of_literals = 2*number_of_features;

	unsigned int number_of_literal_chunks = (number_of_literals-1)/32 + 1;

	// Initialize example vector X
	memset(X, 0, number_of_literal_chunks * sizeof(unsigned int));
	for (int k = number_of_features; k < number_of_literals; ++k) {
		int chunk_nr = k / 32;
		int chunk_pos = k % 32;
		X[chunk_nr] |= (1U << chunk_pos);
	}
	
	if ((indptr_col[active_output[target]+1] - indptr_col[active_output[target]] == 0) || (indptr_col[active_output[target]+1] - indptr_col[active_output[target]] == number_of_rows)) {
		// If no positive/negative examples, produce a random example
		for (int a = 0; a < accumulation; ++a) {
			row = rand() % number_of_rows;
			for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
				int chunk_nr = indices_row[k] / 32;
				int chunk_pos = indices_row[k] % 32;
				X[chunk_nr] |= (1U << chunk_pos);

				chunk_nr = (indices_row[k] + number_of_features) / 32;
				chunk_pos = (indices_row[k] + number_of_features) % 32;
				X[chunk_nr] &= ~(1U << chunk_pos);
			}
		}
		return;
	}

	if (target_value) {
		for (int a = 0; a < accumulation; ++a) {
			// Pick example randomly among positive examples
			int random_index = indptr_col[active_output[target]] + (rand() % (indptr_col[active_output[target]+1] - indptr_col[active_output[target]]));
			row = indices_col[random_index];
			
			for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
				int chunk_nr = indices_row[k] / 32;
				int chunk_pos = indices_row[k] % 32;
				X[chunk_nr] |= (1U << chunk_pos);

				chunk_nr = (indices_row[k] + number_of_features) / 32;
				chunk_pos = (indices_row[k] + number_of_features) % 32;
				X[chunk_nr] &= ~(1U << chunk_pos);
			}
		}
	} else {
		int a = 0;
		while (a < accumulation) {
			row = rand() % number_of_rows;

			if (bsearch(&row, &indices_col[indptr_col[active_output[target]]], indptr_col[active_output[target]+1] - indptr_col[active_output[target]], sizeof(unsigned int), compareints) == NULL) {
				for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
					int chunk_nr = indices_row[k] / 32;
					int chunk_pos = indices_row[k] % 32;
					X[chunk_nr] |= (1U << chunk_pos);

					chunk_nr = (indices_row[k] + number_of_features) / 32;
					chunk_pos = (indices_row[k] + number_of_features) % 32;
					X[chunk_nr] &= ~(1U << chunk_pos);
				}
				a++;
			}
		}
	}
}

void tmu_encode(
        unsigned int *X,
        unsigned int *encoded_X,
        int number_of_examples,
        int dim_x,
        int dim_y,
        int dim_z,
        int patch_dim_x,
        int patch_dim_y,
        int append_negated,
        int spatio_temporal_features
)
{
	int global_number_of_features = dim_x * dim_y * dim_z;
	int number_of_features = spatio_temporal_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
	int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

	int number_of_literal_chunks;
	if (append_negated) {
		number_of_literal_chunks= (((2*number_of_features-1)/32 + 1));
	} else {
		number_of_literal_chunks= (((number_of_features-1)/32 + 1));
	}

	unsigned int *Xi;
	unsigned int *encoded_Xi;

	unsigned int input_pos = 0;
	unsigned int input_step_size = global_number_of_features;

	// Fill encoded_X with zeros

	memset(encoded_X, 0, number_of_examples * number_of_patches * number_of_literal_chunks * sizeof(unsigned int));

	unsigned int encoded_pos = 0;
	for (int i = 0; i < number_of_examples; ++i) {
		//printf("%d\n", i);

		int patch_nr = 0;
		// Produce the patches of the current image
		for (int y = 0; y < dim_y - patch_dim_y + 1; ++y) {
			for (int x = 0; x < dim_x - patch_dim_x + 1; ++x) {
				Xi = &X[input_pos];
				encoded_Xi = &encoded_X[encoded_pos];

				// Encode spatio temporal features into feature vector 
				for (int spatio_temporal_feature = 0; spatio_temporal_feature < spatio_temporal_features; ++spatio_temporal_feature) {

					int chunk_nr = (spatio_temporal_feature + number_of_features) / 32;
					int chunk_pos = (spatio_temporal_feature + number_of_features) % 32;
					encoded_Xi[chunk_nr] |= (1 << chunk_pos);
				}

				// // Encode y coordinate of patch into feature vector 
				// for (int y_threshold = 0; y_threshold < dim_y - patch_dim_y; ++y_threshold) {
				// 	int patch_pos = spatio_temporal_features + y_threshold;

				// 	if (y > y_threshold) {
				// 		int chunk_nr = patch_pos / 32;
				// 		int chunk_pos = patch_pos % 32;
				// 		encoded_Xi[chunk_nr] |= (1 << chunk_pos);
				// 	} else if (append_negated) {
				// 		int chunk_nr = (patch_pos + number_of_features) / 32;
				// 		int chunk_pos = (patch_pos + number_of_features) % 32;
				// 		encoded_Xi[chunk_nr] |= (1 << chunk_pos);
				// 	}
				// }

				// // Encode x coordinate of patch into feature vector
				// for (int x_threshold = 0; x_threshold < dim_x - patch_dim_x; ++x_threshold) {
				// 	int patch_pos = spatio_temporal_features + (dim_y - patch_dim_y) + x_threshold;

				// 	if (x > x_threshold) {
				// 		int chunk_nr = patch_pos / 32;
				// 		int chunk_pos = patch_pos % 32;

				// 		encoded_Xi[chunk_nr] |= (1 << chunk_pos);
				// 	} else if (append_negated) {
				// 		int chunk_nr = (patch_pos + number_of_features) / 32;
				// 		int chunk_pos = (patch_pos + number_of_features) % 32;
				// 		encoded_Xi[chunk_nr] |= (1 << chunk_pos);
				// 	}
				// } 

				// Encode patch content into feature vector
				for (int p_y = 0; p_y < patch_dim_y; ++p_y) {
					for (int p_x = 0; p_x < patch_dim_x; ++p_x) {
						for (int z = 0; z < dim_z; ++z) {
							int image_pos = (y + p_y)*dim_x*dim_z + (x + p_x)*dim_z + z;
							int patch_pos = spatio_temporal_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

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
				encoded_pos += number_of_literal_chunks;
				patch_nr++;
			}
		}
		input_pos += input_step_size;
	}
}