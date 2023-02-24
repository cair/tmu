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

import tmu.tools

from numba import jit, prange

@jit(nopython=True)
def pack_X_numba(encoded_X, e, packed_X, number_of_literals):
    packed_X[:] = 0
    for i in range(32):
        if e+i >= encoded_X.shape[0]:
            break

        for k in range(number_of_literals):
            chunk = k // 32
            pos = k % 32

            if encoded_X[e+i,chunk] & (1 << pos) > 0:
                packed_X[k] |= (1 << i)
    return packed_X

jit(nopython=True)
def unpack_X_numba(encoded_X, e, packed_X, number_of_literals):

    encoded_X[e:e+i] = 0
    for i in range(32):
        if e+i >= encoded_X.shape[0]:
            break

        for k in range(number_of_literals):
            chunk = k // 32
            pos = k % 32

            if packed_X[k] & (1 << i) > 0:
                encoded_X[e+i,chunk] |= (1 << pos)
    return packed_X

@jit(nopython=True)
def unpack_clause_output_numba(e, clause_output, clause_output_batch, number_of_clauses):
    for j in range(number_of_clauses):
        clause_output[j] = (clause_output_batch[j] & (1 << (e % 32))) > 0
    return clause_output

@jit(nopython=True)
def calculate_clause_outputs_update_numba(literal_active, encoded_X, e, number_of_clauses, clause_output, clause_bank_included_dynamic, clause_bank_included_dynamic_length,
                    clause_bank_included_static, clause_bank_included_static_length):
    for j in range(number_of_clauses):
        clause_output[j] = 1

        for k in range(clause_bank_included_static_length[j]):
                literal_chunk = clause_bank_included_static[j, k] // 32
                literal_pos = clause_bank_included_static[j, k] % 32
                if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                    clause_output[j] = 0
                    break

        if clause_output[j] == 0:
            continue

        for k in range(clause_bank_included_dynamic_length[j]):
            literal_chunk = clause_bank_included_dynamic[j, k, 0] // 32
            literal_pos = clause_bank_included_dynamic[j, k, 0] % 32
            if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                clause_output[j] = 0
                break

    return clause_output

@jit(nopython=True)
def calculate_clause_outputs_predict_packed_X_numba(packed_X, number_of_clauses, clause_output_batch, clause_bank_included_dynamic, clause_bank_included_dynamic_length,
            clause_bank_included_static, clause_bank_included_static_length):
    for j in range(number_of_clauses):
        if clause_bank_included_dynamic_length[j] == 0 and clause_bank_included_static_length[j] == 0:
            clause_output_batch[j] = 0
        else:
            clause_output_batch[j] = ~0

            for k in range(clause_bank_included_static_length[j]):
                clause_output_batch[j] &= packed_X[clause_bank_included_static[j, k]]

            for k in range(clause_bank_included_dynamic_length[j]):
                clause_output_batch[j] &= packed_X[clause_bank_included_dynamic[j, k, 0]]

    return clause_output_batch

@jit(nopython=True)
def calculate_clause_outputs_predict_numba(encoded_X, e, number_of_clauses, clause_output, clause_bank_included_dynamic, clause_bank_included_dynamic_length,
                    clause_bank_included_static, clause_bank_included_static_length):
    for j in range(number_of_clauses):
        if clause_bank_included_dynamic_length[j] == 0 and clause_bank_included_static_length[j] == 0:
            clause_output[j] = 0
        else:
            clause_output[j] = 1

            for k in range(clause_bank_included_static_length[j]):
                literal_chunk = clause_bank_included_static[j, k] // 32
                literal_pos = clause_bank_included_static[j, k] % 32
                if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                    clause_output[j] = 0
                    break

            if clause_output[j] == 0:
                continue

            for k in range(clause_bank_included_dynamic_length[j]):
                literal_chunk = clause_bank_included_dynamic[j, k, 0] // 32
                literal_pos = clause_bank_included_dynamic[j, k, 0] % 32
                if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                    clause_output[j] = 0
                    break

    return clause_output

@jit(nopython=True,parallel=True)
def type_i_feedback_numba(update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                    literal_active, encoded_X, e, number_of_clauses, number_of_states, clause_bank_included_dynamic,
                    clause_bank_included_dynamic_length, clause_bank_excluded_dynamic, clause_bank_excluded_dynamic_length, clause_bank_included_static,
                    clause_bank_included_static_length):
    for j in prange(number_of_clauses):
        if np.random.random() > update_p or not clause_active[j]:
            continue

        clause_output = 1

        for k in range(clause_bank_included_static_length[j]):
            literal_chunk = clause_bank_included_static[j, k] // 32
            literal_pos = clause_bank_included_static[j, k] % 32
            if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                clause_output = 0
                break

        if clause_output:
            for k in range(clause_bank_included_dynamic_length[j]):
                literal_chunk = clause_bank_included_dynamic[j, k, 0] // 32
                literal_pos = clause_bank_included_dynamic[j, k, 0] % 32
                if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                    clause_output = 0
                    break

        if clause_output and clause_bank_included_dynamic_length[j] <= max_included_literals:
            # Type Ia Feedback

            for k in range(clause_bank_included_dynamic_length[j]-1, -1, -1):
                literal_chunk = clause_bank_included_dynamic[j, k, 0] // 32
                literal_pos = clause_bank_included_dynamic[j, k, 0] % 32

                if encoded_X[e, literal_chunk] & (1 << literal_pos) > 0:
                    if clause_bank_included_dynamic[j, k, 1] < number_of_states-1:# and np.random.random() <= (s-1.0)/s:
                        clause_bank_included_dynamic[j, k, 1] += 1
                    else:
                        clause_bank_included_static[j, clause_bank_included_static_length[j]] = clause_bank_included_dynamic[j, k, 0]
                        clause_bank_included_static_length[j] += 1

                        clause_bank_included_dynamic_length[j] -= 1
                        clause_bank_included_dynamic[j, k, 0] = clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 0]       
                        clause_bank_included_dynamic[j, k, 1] = clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 1]

                elif np.random.random() <= 1.0/s:
                    clause_bank_included_dynamic[j, k, 1] -= 1
                    if clause_bank_included_dynamic[j, k, 1] < number_of_states//2:
                        clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 0] = clause_bank_included_dynamic[j, k, 0]
                        clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 1] = clause_bank_included_dynamic[j, k, 1]
                        clause_bank_excluded_dynamic_length[j] += 1

                        clause_bank_included_dynamic_length[j] -= 1
                        clause_bank_included_dynamic[j, k, 0] = clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 0]       
                        clause_bank_included_dynamic[j, k, 1] = clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 1]

            for k in range(clause_bank_excluded_dynamic_length[j]-1, -1, -1):
                literal_chunk = clause_bank_excluded_dynamic[j, k, 0] // 32
                literal_pos = clause_bank_excluded_dynamic[j, k, 0] % 32

                if encoded_X[e, literal_chunk] & (1 << literal_pos) > 0:
                    if True or np.random.random() <= (s-1.0)/s:
                        clause_bank_excluded_dynamic[j, k, 1] += 1
                        if clause_bank_excluded_dynamic[j, k, 1] >= number_of_states//2:
                            clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 0] = clause_bank_excluded_dynamic[j, k, 0]
                            clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 1] = clause_bank_excluded_dynamic[j, k, 1]
                            clause_bank_included_dynamic_length[j] += 1

                            clause_bank_excluded_dynamic_length[j] -= 1
                            clause_bank_excluded_dynamic[j, k, 0] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 0]       
                            clause_bank_excluded_dynamic[j, k, 1] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 1]
                elif np.random.random() <= 1.0/s:
                    if clause_bank_excluded_dynamic[j, k, 1] > 0:
                        clause_bank_excluded_dynamic[j, k, 1] -= 1
                    else:
                        clause_bank_excluded_dynamic_length[j] -= 1
                        clause_bank_excluded_dynamic[j, k, 0] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 0]       
                        clause_bank_excluded_dynamic[j, k, 1] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 1]
        else:
            # Type Ib Feedback

            for k in range(clause_bank_included_dynamic_length[j]-1, -1, -1):
                if np.random.random() <= 1.0/s:
                    clause_bank_included_dynamic[j, k, 1] -= 1

                    if clause_bank_included_dynamic[j, k, 1] < number_of_states//2:
                        clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 0] = clause_bank_included_dynamic[j, k, 0]
                        clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 1] = clause_bank_included_dynamic[j, k, 1]
                        clause_bank_excluded_dynamic_length[j] += 1

                        clause_bank_included_dynamic_length[j] -= 1
                        clause_bank_included_dynamic[j, k, 0] = clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 0]       
                        clause_bank_included_dynamic[j, k, 1] = clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 1]
                        clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 0] = 0
                        clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 1] = 0
            
            for k in range(clause_bank_excluded_dynamic_length[j]-1, -1, -1):
                if np.random.random() <= 1.0/s:
                    if clause_bank_excluded_dynamic[j, k, 1] > 0:
                        clause_bank_excluded_dynamic[j, k, 1] -= 1
                    else:
                        clause_bank_excluded_dynamic_length[j] -= 1
                        clause_bank_excluded_dynamic[j, k, 0] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 0]       
                        clause_bank_excluded_dynamic[j, k, 1] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 1]

@jit(nopython=True,parallel=True)
def type_ii_feedback_numba(update_p, clause_active, literal_active, encoded_X, e, number_of_clauses, number_of_states, clause_bank_included_dynamic,
                    clause_bank_included_dynamic_length, clause_bank_excluded_dynamic, clause_bank_excluded_dynamic_length, clause_bank_included_static,
                    clause_bank_included_static_length):
    for j in prange(number_of_clauses):
        if np.random.random() > update_p or not clause_active[j]:
            continue

        clause_output = 1

        for k in range(clause_bank_included_static_length[j]):
            literal_chunk = clause_bank_included_static[j, k] // 32
            literal_pos = clause_bank_included_static[j, k] % 32
            if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                clause_output = 0
                break

        if clause_output:
            for k in range(clause_bank_included_dynamic_length[j]):
                literal_chunk = clause_bank_included_dynamic[j, k, 0] // 32
                literal_pos = clause_bank_included_dynamic[j, k, 0] % 32
                if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                    clause_output = 0
                    break

        if not clause_output:
            continue

        # Type II Feedback

        for k in range(clause_bank_excluded_dynamic_length[j]-1, -1, -1):
            literal_chunk = clause_bank_excluded_dynamic[j, k, 0] // 32
            literal_pos = clause_bank_excluded_dynamic[j, k, 0] % 32

            if encoded_X[e, literal_chunk] & (1 << literal_pos) == 0:
                clause_bank_excluded_dynamic[j, k, 1] += 1
                if clause_bank_excluded_dynamic[j, k, 1] >= number_of_states//2:
                    clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 0] = clause_bank_excluded_dynamic[j, k, 0]
                    clause_bank_included_dynamic[j, clause_bank_included_dynamic_length[j], 1] = clause_bank_excluded_dynamic[j, k, 1]
                    clause_bank_included_dynamic_length[j] += 1

                    clause_bank_excluded_dynamic_length[j] -= 1
                    clause_bank_excluded_dynamic[j, k, 0] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 0]       
                    clause_bank_excluded_dynamic[j, k, 1] = clause_bank_excluded_dynamic[j, clause_bank_excluded_dynamic_length[j], 1]

class ClauseBankSparse:
    def __init__(self, X, number_of_clauses, number_of_states, patch_dim,
                 batching=True, incremental=True, skip=0.1):
        self.number_of_clauses = int(number_of_clauses)
        self.number_of_states = int(number_of_states)
        self.patch_dim = patch_dim
        self.batching = batching
        self.incremental = incremental

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
        self.clause_output_batch = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))

        self.clause_output_patchwise = np.ascontiguousarray(
            np.empty((int(self.number_of_clauses * self.number_of_patches)), dtype=np.uint32))

        self.output_one_patches = np.ascontiguousarray(np.empty(self.number_of_patches, dtype=np.uint32))

        self.literal_clause_count = np.ascontiguousarray(np.empty((int(self.number_of_literals)), dtype=np.uint32))

        self.packed_X = np.ascontiguousarray(np.empty(self.number_of_literals, dtype=np.uint32))

        self.initialize_clauses()

    def initialize_clauses(self):
        self.clause_bank_included_static = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals), dtype=np.uint32)) # Contains index of included literals, none at start
        self.clause_bank_included_static_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint32))
        
        self.clause_bank_included_dynamic = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals, 2), dtype=np.uint32)) # Contains index and state of included literals, none at start
        self.clause_bank_included_dynamic_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint32)) 

        self.clause_bank_excluded_dynamic = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals, 2), dtype=np.uint32)) # Contains index and state of excluded literals
        self.clause_bank_excluded_dynamic_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint32)) # All literals excluded at start
        self.clause_bank_excluded_dynamic_length[:] = self.number_of_literals
        self.clause_bank_excluded_dynamic[:,:,0] = np.arange(self.number_of_literals, dtype=np.uint32) # Initialize clause literals with increasing index
        self.clause_bank_excluded_dynamic[:,:,1] = self.number_of_states // 2 - 1 # Initialize excluded literals in least forgotten state
 
    def calculate_clause_outputs_predict(self, encoded_X, e):
        if not self.batching:
            return calculate_clause_outputs_predict_numba(encoded_X, e, self.number_of_clauses, self.clause_output, self.clause_bank_included_dynamic, self.clause_bank_included_dynamic_length,
            self.clause_bank_included_static, self.clause_bank_included_static_length)
            
        if e % 32 == 0:
            self.packed_X = pack_X_numba(encoded_X, e, self.packed_X, self.number_of_literals)

            self.clause_output_batch = calculate_clause_outputs_predict_packed_X_numba(self.packed_X, self.number_of_clauses, self.clause_output_batch, self.clause_bank_included_dynamic, self.clause_bank_included_dynamic_length,
                self.clause_bank_included_static, self.clause_bank_included_static_length)
        return unpack_clause_output_numba(e, self.clause_output, self.clause_output_batch, self.number_of_clauses)


    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        return calculate_clause_outputs_update_numba(literal_active, encoded_X, e, self.number_of_clauses, self.clause_output, self.clause_bank_included_dynamic, self.clause_bank_included_dynamic_length,
            self.clause_bank_included_static, self.clause_bank_included_static_length)

    def type_i_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e):
        type_i_feedback_numba(update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e, self.number_of_clauses, self.number_of_states, self.clause_bank_included_dynamic,
                        self.clause_bank_included_dynamic_length, self.clause_bank_excluded_dynamic, self.clause_bank_excluded_dynamic_length,
                        self.clause_bank_included_static, self.clause_bank_included_static_length)

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        type_ii_feedback_numba(update_p, clause_active, literal_active, encoded_X, e, self.number_of_clauses, self.number_of_states, self.clause_bank_included_dynamic,
                    self.clause_bank_included_dynamic_length, self.clause_bank_excluded_dynamic, self.clause_bank_excluded_dynamic_length,
                    self.clause_bank_included_static, self.clause_bank_included_static_length)

    def prepare_X(self, X):
        return tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                self.patch_dim, 0)

    def prepare_autoencoder_examples(self, X_csr, X_csc, active_output, accumulation):
        return tmu.tools.produce_autoencoder_examples(X_csr, X_csc, active_output, accumulation,
                                                      self.number_of_ta_chunks)
