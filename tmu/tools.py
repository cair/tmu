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

import numpy as np

from ._tools import ffi, lib

def encode(X, number_of_examples, number_of_ta_chunks, dim_x, dim_y, dim_z, patch_dim_x, patch_dim_y, class_features):
	Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
	encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * number_of_ta_chunks), dtype=np.uint32))
	lib.tmu_encode(ffi.cast("unsigned int *", Xm.ctypes.data), ffi.cast("unsigned int *", encoded_X.ctypes.data), number_of_examples, dim_x, dim_y, dim_z, patch_dim_x, patch_dim_y, 1, class_features)
	return np.ascontiguousarray(encoded_X.reshape((int(number_of_examples), number_of_ta_chunks)))
