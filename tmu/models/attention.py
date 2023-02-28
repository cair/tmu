# Copyright (c) 2023 Ole-Christoffer Granmo

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

from tmu.tmulib import ffi, lib
import tmu.tools


class Attention:
    def __init__(self, X, patch_dim):
        self.patch_dim = patch_dim

        if len(X.shape) == 2:
            self.dim = (X.shape[1], 1, 1)
        elif len(X.shape) == 3:
            self.dim = (X.shape[1], X.shape[2], 1)
        elif len(X.shape) == 4:
            self.dim = (X.shape[1], X.shape[2], X.shape[3])

        if self.patch_dim is None:
            self.patch_dim = (self.dim[0] * self.dim[1] * self.dim[2], 1)

        self.number_of_features = int(
            self.patch_dim[0] * self.patch_dim[1] * self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (
                    self.dim[1] - self.patch_dim[1]))
        self.number_of_literals = self.number_of_features * 2

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))

        self.ranking = np.arange(self.number_of_literals, dtype=np.uint32)
        self.ra_p = ffi.cast("unsigned int *", self.ranking.ctypes.data)
        np.random.shuffle(self.ranking)

    def type_i_feedback(self, update_p, s, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        lib.at_type_i_feedback(self.ra_p, self.number_of_literals, update_p, s, xi_p)

    def type_ii_feedback(self, update_p, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        lib.at_type_ii_feedback(self.ra_p, self.number_of_literals, update_p, xi_p)

    def get_attention(self, included_literals, encoded_X):
        xi_p = ffi.cast("unsigned int *", encoded_X.ctypes.data)
        il_p = ffi.cast("unsigned int *", included_literals.ctypes.data)

        lib.at_get_attention(self.ra_p, self.number_of_literals, 10, il_p, xi_p)

    def prepare_X(self, X):
        return tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                self.patch_dim, 0)
