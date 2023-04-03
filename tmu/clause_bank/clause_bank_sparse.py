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

from tmu.tmulib import ffi, lib
import tmu.tools

import numpy as np
from scipy.sparse import csr_matrix

class ClauseBankSparse:
    def __init__(self, X, number_of_clauses, number_of_states, patch_dim,
                 batching=True, incremental=True, absorbing=-1, literal_sampling=1.0,
                 feedback_rate_excluded_literals=1, literal_insertion_state = -1):
        self.number_of_clauses = int(number_of_clauses)
        self.number_of_states = int(number_of_states)
        self.patch_dim = patch_dim
        self.batching = batching
        self.incremental = incremental
        self.absorbing = int(absorbing)
        self.literal_sampling = float(literal_sampling)
        self.feedback_rate_excluded_literals = feedback_rate_excluded_literals
        self.literal_insertion_state = literal_insertion_state

        if len(X.shape) == 2:
            self.dim = (X.shape[1], 1, 1)
        elif len(X.shape) == 3:
            self.dim = (X.shape[1], X.shape[2], 1)
        elif len(X.shape) == 4:
            self.dim = (X.shape[1], X.shape[2], X.shape[3])

        if self.patch_dim == None:
            self.patch_dim = (self.dim[0] * self.dim[1] * self.dim[2], 1)

        self.number_of_features = int(
            self.patch_dim[0] * self.patch_dim[1] * self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (
                        self.dim[1] - self.patch_dim[1]))
        self.number_of_literals = self.number_of_features * 2

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.number_of_ta_chunks = int((self.number_of_literals - 1) / 32 + 1)

        self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)
        self.clause_output_batch = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
        self.cob_p = ffi.cast("unsigned int *", self.clause_output_batch.ctypes.data)

        self.clause_output_patchwise = np.ascontiguousarray(
            np.empty((int(self.number_of_clauses * self.number_of_patches)), dtype=np.uint32))

        self.output_one_patches = np.ascontiguousarray(np.empty(self.number_of_patches, dtype=np.uint32))

        self.literal_clause_count = np.ascontiguousarray(np.empty((int(self.number_of_literals)), dtype=np.uint32))

        self.packed_X = np.ascontiguousarray(np.empty(self.number_of_literals, dtype=np.uint32))
        self.px_p = ffi.cast("unsigned int *", self.packed_X.ctypes.data)

        self.Xi = np.ascontiguousarray(np.zeros(self.number_of_ta_chunks, dtype=np.uint32))
        self.Xi_p = ffi.cast("unsigned int *", self.Xi.ctypes.data)
        for k in range(self.number_of_features, self.number_of_literals):
            chunk = k // 32
            pos = k % 32
            self.Xi[chunk] |= (1 << pos)
        
        self.initialize_clauses()

    def initialize_clauses(self):
        self.clause_bank_included = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals, 2), dtype=np.uint16)) # Contains index and state of included literals, none at start
        self.cbi_p = ffi.cast("unsigned short *", self.clause_bank_included.ctypes.data)
        self.clause_bank_included_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint16)) 
        self.cbil_p = ffi.cast("unsigned short *", self.clause_bank_included_length.ctypes.data)

        self.clause_bank_excluded = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals, 2), dtype=np.uint16)) # Contains index and state of excluded literals
        self.cbe_p = ffi.cast("unsigned short *", self.clause_bank_excluded.ctypes.data)
        self.clause_bank_excluded_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint16)) # All literals excluded at start
        self.cbel_p = ffi.cast("unsigned short *", self.clause_bank_excluded_length.ctypes.data)
        self.clause_bank_excluded_length[:] = int(self.number_of_literals * self.literal_sampling)

        for j in range(self.number_of_clauses):
            literal_indexes = np.arange(self.number_of_literals, dtype=np.uint16)
            np.random.shuffle(literal_indexes)
            self.clause_bank_excluded[j,:,0] = literal_indexes # Initialize clause literals randomly
            self.clause_bank_excluded[j,:,1] = self.number_of_states // 2 - 1 # Initialize excluded literals in least forgotten state

        self.clause_bank_unallocated = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals), dtype=np.uint16)) # Contains index and unallocated literals
        self.cbu_p = ffi.cast("unsigned short *", self.clause_bank_unallocated.ctypes.data)
        self.clause_bank_unallocated_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint16)) # All literals excluded at start
        self.cbul_p = ffi.cast("unsigned short *", self.clause_bank_unallocated_length.ctypes.data)
        self.clause_bank_unallocated_length[:] = self.number_of_literals - int(self.number_of_literals * self.literal_sampling)

        for j in range(self.number_of_clauses):
            self.clause_bank_unallocated[j,:] = np.flip(self.clause_bank_excluded[j,:,0])

    def calculate_clause_outputs_predict(self, encoded_X, e):
        if not self.batching:
            lib.cbs_prepare_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)
            lib.cbs_calculate_clause_outputs_predict(self.Xi_p, self.number_of_clauses, self.number_of_literals, self.co_p, self.cbi_p, self.cbil_p)
            lib.cbs_restore_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)
            return self.clause_output

        if e % 32 == 0:
            lib.cbs_pack_X(ffi.cast("int *", encoded_X[0].indptr.ctypes.data), ffi.cast("int *", encoded_X[0].indices.ctypes.data), encoded_X[0].indptr.shape[0]-1, e, self.px_p, self.number_of_literals)
            lib.cbs_calculate_clause_outputs_predict_packed_X(self.px_p, self.number_of_clauses, self.number_of_literals, self.cob_p, self.cbi_p, self.cbil_p)
        lib.cbs_unpack_clause_output(e, self.co_p, self.cob_p, self.number_of_clauses)
        return self.clause_output

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        lib.cbs_prepare_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)
        lib.cbs_calculate_clause_outputs_update(ffi.cast("unsigned int *", literal_active.ctypes.data), self.Xi_p, self.number_of_clauses, self.number_of_literals, self.co_p, self.cbi_p, self.cbil_p)
        lib.cbs_restore_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)
        return self.clause_output

    def type_i_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e):
        lib.cbs_prepare_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)
        lib.cbs_type_i_feedback(update_p, s, int(boost_true_positive_feedback), int(max_included_literals), self.absorbing, self.feedback_rate_excluded_literals, self.literal_insertion_state, ffi.cast("int *", clause_active.ctypes.data), ffi.cast("unsigned int *", literal_active.ctypes.data), self.Xi_p, self.number_of_clauses, self.number_of_literals, self.number_of_states, self.cbi_p,
                        self.cbil_p, self.cbe_p, self.cbel_p, self.cbu_p, self.cbul_p)
        lib.cbs_restore_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        lib.cbs_prepare_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)
        lib.cbs_type_ii_feedback(update_p, self.feedback_rate_excluded_literals, ffi.cast("int *", clause_active.ctypes.data), ffi.cast("unsigned int *", literal_active.ctypes.data), self.Xi_p, self.number_of_clauses, self.number_of_literals, self.number_of_states, self.cbi_p,
                        self.cbil_p, self.cbe_p, self.cbel_p)
        lib.cbs_restore_Xi(encoded_X[1][e], encoded_X[0].indptr[e+1] - encoded_X[0].indptr[e], self.Xi_p, self.number_of_features)

    def number_of_include_actions(self, clause):
        return self.clause_bank_included_length[clause]

    def number_of_exclude_actions(self, clause):
        return self.clause_bank_excluded_length[clause]

    def number_of_unallocated_literals(self, clause):
        return self.clause_bank_unallocated_length[clause]

    def get_ta_action(self, clause, ta):
        if ta in self.clause_bank_included[clause, :self.clause_bank_included_length[clause], 0]:
            return 1
        else:
            return 0

    def get_ta_state(self, clause, ta):
        action = self.get_ta_action(clause, ta)
        if action == 0:
            literals = self.clause_bank_excluded[clause, :self.clause_bank_excluded_length[clause], 0]
            return self.clause_bank_excluded[clause, np.nonzero(literals == ta)[0][0], 1]
        else:
            literals = self.clause_bank_included[clause, :self.clause_bank_included_length[clause], 0]
            return self.clause_bank_included[clause, np.nonzero(literals == ta)[0][0], 1]

    def prepare_X(self, X):
        X_csr = csr_matrix(X, dtype=np.uint32)
        X_p = []
        for e in range(X.shape[0]):
            X_p.append(ffi.cast("unsigned int *",X_csr.indices[X_csr.indptr[e]:X_csr.indptr[e+1]].ctypes.data))
        return (X_csr, X_p)
