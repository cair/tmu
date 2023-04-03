# Copyright (c) 2023 Ole-Christoffer Granmo
import abc
import time

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
from scipy.sparse import csr_matrix, csc_matrix

from tmu.tmulib import ffi, lib


class CFFISerializable(abc.ABC):
    @abc.abstractmethod
    def _cffi_init(self):
        raise NotImplementedError("You need to implement _cffi_init")

    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items() if v.__class__.__name__ != "_CDataBase"}
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._cffi_init()


class BenchmarkTimer:
    start_time: float
    end_time: float

    def __init__(self, logger=None, text=None):
        self.text = text if text else ""
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

        if self.logger:
            self.logger.debug(f"[{self.end_time - self.start_time:2f}ms] - {self.text}")

    def elapsed(self):
        if self.end_time is None:
            raise RuntimeError("elapsed() must be called after context is ended.")
        return self.end_time - self.start_time


class Matrix:

    def __init__(self, data: np.ndarray, cast_to="unsigned int *"):
        if not isinstance(data, np.ndarray):
            raise RuntimeError("data argument should be of type np.ndarray")
        elif not data.flags['C_CONTIGUOUS']:
            raise RuntimeError(
                "data is not C_CONTIGUOUS. Use order='c' when creating the matrix or np.ascontiguousarray")

        self._cast_to = cast_to
        self._data = data
        self._ptr = ffi.cast(cast_to, self.data.ctypes.data)

    @property
    def ptr(self):
        return self._ptr

    @property
    def data(self):
        return self._data

    def slice_ptr(self, val):
        return ffi.cast(self._cast_to, self._data[val].ctypes.data)

def produce_autoencoder_examples(X_csr, X_csc, active_output, accumulation):
    X = np.ascontiguousarray(np.empty(int(X_csc.shape[1] * active_output.shape[0]), dtype=np.uint32))
    Y = np.ascontiguousarray(np.empty(int(active_output.shape[0]), dtype=np.uint32))

    lib.tmu_produce_autoencoder_examples(ffi.cast("unsigned int *", active_output.ctypes.data), active_output.shape[0],
                                         ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indptr).ctypes.data),
                                         ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indices).ctypes.data),
                                         int(X_csr.shape[0]),
                                         ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indptr).ctypes.data),
                                         ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indices).ctypes.data),
                                         int(X_csc.shape[1]), ffi.cast("unsigned int *", X.ctypes.data),
                                         ffi.cast("unsigned int *", Y.ctypes.data), int(accumulation));
    return X.reshape((len(active_output), -1)), Y

def encode(X, number_of_examples, number_of_patches, number_of_ta_chunks, dim, patch_dim, class_features,
           append_negated=True):
    Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
    encoded_X = np.ascontiguousarray(
        np.empty(int(number_of_examples * number_of_patches * number_of_ta_chunks), dtype=np.uint32))
    lib.tmu_encode(ffi.cast("unsigned int *", Xm.ctypes.data), ffi.cast("unsigned int *", encoded_X.ctypes.data),
                   number_of_examples, dim[0], dim[1], dim[2], patch_dim[0], patch_dim[1], int(append_negated),
                   class_features)
    return np.ascontiguousarray(encoded_X.reshape((int(number_of_examples), number_of_patches * number_of_ta_chunks)))
