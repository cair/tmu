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
from scipy.sparse import csr_matrix
from numba import jit

@jit(nopython=True)
def prepare_Xi_numba(indices, Xi, number_of_features):
    for k in indices:
        chunk = k // 32
        pos = k % 32
        Xi[chunk] |= (1 << pos)
        chunk = (k + number_of_features) // 32
        pos = (k + number_of_features) % 32
        Xi[chunk] &= ~(1 << pos)

@jit(nopython=True)
def restore_Xi_numba(indices, Xi, number_of_features):
    for k in indices:
        chunk = k // 32
        pos = k % 32
        Xi[chunk] &= ~(1 << pos)
        chunk = (k + number_of_features) // 32
        pos = (k + number_of_features) % 32
        Xi[chunk] |= (1 << pos)

@jit(nopython=True)
def pack_X_numba(indptr, indices, e, packed_X, number_of_literals):
    packed_X[:number_of_literals//2] = 0
    packed_X[number_of_literals//2:] = ~0
    for i in range(32):
        if e+i >= indptr.shape[0]-1:
            break

        for k in indices[indptr[e+i]:indptr[e+i+1]]:
            packed_X[k] |= (1 << i)
            packed_X[k + number_of_literals//2] &= ~(1 << i)

@jit(nopython=True)
def unpack_clause_output_numba(e, clause_output, clause_output_batch, number_of_clauses):
    for j in range(number_of_clauses):
        clause_output[j] = (clause_output_batch[j] & (1 << (e % 32))) > 0

@jit(nopython=True)
def calculate_clause_outputs_update_numba(literal_active, Xi, number_of_clauses, clause_output, clause_bank_included, clause_bank_included_length):
    for j in range(number_of_clauses):
        clause_output[j] = 1
        for k in range(clause_bank_included_length[j]):
            literal_chunk = clause_bank_included[j, k, 0] // 32
            literal_pos = clause_bank_included[j, k, 0] % 32
            if Xi[literal_chunk] & (1 << literal_pos) == 0:
                clause_output[j] = 0
                break

@jit(nopython=True)
def calculate_clause_outputs_predict_packed_X_numba(packed_X, number_of_clauses, clause_output_batch, clause_bank_included, clause_bank_included_length):
    for j in range(number_of_clauses):
        if clause_bank_included_length[j] == 0:
            clause_output_batch[j] = 0
        else:
            clause_output_batch[j] = ~0

            for k in range(clause_bank_included_length[j]):
                clause_output_batch[j] &= packed_X[clause_bank_included[j, k, 0]]

@jit(nopython=True)
def calculate_clause_outputs_predict_numba(Xi, number_of_clauses, clause_output, clause_bank_included, clause_bank_included_length):
    for j in range(number_of_clauses):
        if clause_bank_included_length[j] == 0:
            clause_output[j] = 0
        else:
            clause_output[j] = 1

            for k in range(clause_bank_included_length[j]):
                literal_chunk = clause_bank_included[j, k, 0] // 32
                literal_pos = clause_bank_included[j, k, 0] % 32
                if Xi[literal_chunk] & (1 << literal_pos) == 0:
                    clause_output[j] = 0
                    break

@jit(nopython=True)
def type_i_feedback_numba(update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                    literal_active, feedback_to_ta, Xi, number_of_clauses, number_of_literals, number_of_states, clause_bank_included,
                    clause_bank_included_length, clause_bank_excluded, clause_bank_excluded_length):
    for j in range(number_of_clauses):
        if (not clause_active[j]) or np.random.random() > update_p:
            continue

        p = 1.0/s
        # number_of_decrements = np.random.normal(1.0*number_of_literals*p, np.sqrt(number_of_literals * p * (1.0 - p)))
        # if number_of_decrements > number_of_literals:
        #     number_of_decrements = number_of_literals
        # elif number_of_decrements < 0:
        #     number_of_decrements = 0

        number_of_decrements = np.random.binomial(number_of_literals, p)

        feedback_to_ta[:] = 0
        for k in range(number_of_decrements):
            literal = np.random.randint(number_of_literals)
            literal_chunk = literal // 32
            literal_pos = literal % 32
            while feedback_to_ta[literal_chunk] & (1 << literal_pos):
                literal = np.random.randint(number_of_literals)
                literal_chunk = literal // 32
                literal_pos = literal % 32
            feedback_to_ta[literal_chunk] |= (1 << literal_pos)

        clause_output = 1
        for k in range(clause_bank_included_length[j]):
            literal_chunk = clause_bank_included[j, k, 0] // 32
            literal_pos = clause_bank_included[j, k, 0] % 32
            if Xi[literal_chunk] & (1 << literal_pos) == 0:
                clause_output = 0
                break

        if clause_output and (clause_bank_included_length[j] <= max_included_literals):
            for k in range(clause_bank_included_length[j]-1, -1, -1):
                literal_chunk = clause_bank_included[j, k, 0] // 32
                literal_pos = clause_bank_included[j, k, 0] % 32

                if Xi[literal_chunk] & (1 << literal_pos) > 0:
                    if clause_bank_included[j, k, 1] < number_of_states-1 and (boost_true_positive_feedback or (feedback_to_ta[literal_chunk] & (1 << literal_pos) == 0)):
                        clause_bank_included[j, k, 1] += 1
                elif feedback_to_ta[literal_chunk] & (1 << literal_pos):
                    clause_bank_included[j, k, 1] -= 1
                    if clause_bank_included[j, k, 1] < number_of_states//2:
                        clause_bank_excluded[j, clause_bank_excluded_length[j], 0] = clause_bank_included[j, k, 0]
                        clause_bank_excluded[j, clause_bank_excluded_length[j], 1] = clause_bank_included[j, k, 1]
                        clause_bank_excluded_length[j] += 1

                        clause_bank_included_length[j] -= 1
                        clause_bank_included[j, k, 0] = clause_bank_included[j, clause_bank_included_length[j], 0]       
                        clause_bank_included[j, k, 1] = clause_bank_included[j, clause_bank_included_length[j], 1]

            for k in range(clause_bank_excluded_length[j]-1, -1, -1):
                literal_chunk = clause_bank_excluded[j, k, 0] // 32
                literal_pos = clause_bank_excluded[j, k, 0] % 32

                if Xi[literal_chunk] & (1 << literal_pos) > 0:
                    if boost_true_positive_feedback or (feedback_to_ta[literal_chunk] & (1 << literal_pos) == 0):
                        clause_bank_excluded[j, k, 1] += 1
                        if clause_bank_excluded[j, k, 1] >= number_of_states//2:
                            clause_bank_included[j, clause_bank_included_length[j], 0] = clause_bank_excluded[j, k, 0]
                            clause_bank_included[j, clause_bank_included_length[j], 1] = clause_bank_excluded[j, k, 1]
                            clause_bank_included_length[j] += 1

                            clause_bank_excluded_length[j] -= 1
                            clause_bank_excluded[j, k, 0] = clause_bank_excluded[j, clause_bank_excluded_length[j], 0]       
                            clause_bank_excluded[j, k, 1] = clause_bank_excluded[j, clause_bank_excluded_length[j], 1]
                elif (feedback_to_ta[literal_chunk] & (1 << literal_pos)) and (clause_bank_excluded[j, k, 1] > 0):
                    clause_bank_excluded[j, k, 1] -= 1
        else:
            for k in range(clause_bank_included_length[j]-1, -1, -1):
                literal_chunk = clause_bank_included[j, k, 0] // 32
                literal_pos = clause_bank_included[j, k, 0] % 32
                if feedback_to_ta[literal_chunk] & (1 << literal_pos):
                    clause_bank_included[j, k, 1] -= 1

                    if clause_bank_included[j, k, 1] < number_of_states//2:
                        clause_bank_excluded[j, clause_bank_excluded_length[j], 0] = clause_bank_included[j, k, 0]
                        clause_bank_excluded[j, clause_bank_excluded_length[j], 1] = clause_bank_included[j, k, 1]
                        clause_bank_excluded_length[j] += 1

                        clause_bank_included_length[j] -= 1
                        clause_bank_included[j, k, 0] = clause_bank_included[j, clause_bank_included_length[j], 0]       
                        clause_bank_included[j, k, 1] = clause_bank_included[j, clause_bank_included_length[j], 1]
            
            for k in range(clause_bank_excluded_length[j]-1, -1, -1):
                literal_chunk = clause_bank_excluded[j, k, 0] // 32
                literal_pos = clause_bank_excluded[j, k, 0] % 32
                if (feedback_to_ta[literal_chunk] & (1 << literal_pos)) and (clause_bank_excluded[j, k, 1] > 0):
                    clause_bank_excluded[j, k, 1] -= 1

@jit(nopython=True)
def type_ii_feedback_numba(update_p, clause_active, literal_active, Xi, number_of_clauses, number_of_states, clause_bank_included,
                    clause_bank_included_length, clause_bank_excluded, clause_bank_excluded_length):
    for j in range(number_of_clauses):
        if (not clause_active[j]) or np.random.random() > update_p:
            continue

        clause_output = 1
        for k in range(clause_bank_included_length[j]):
            literal_chunk = clause_bank_included[j, k, 0] // 32
            literal_pos = clause_bank_included[j, k, 0] % 32
            if Xi[literal_chunk] & (1 << literal_pos) == 0:
                clause_output = 0
                break

        if not clause_output:
            continue

        # Type II Feedback

        for k in range(clause_bank_excluded_length[j]-1, -1, -1):
            literal_chunk = clause_bank_excluded[j, k, 0] // 32
            literal_pos = clause_bank_excluded[j, k, 0] % 32

            if Xi[literal_chunk] & (1 << literal_pos) == 0:
                clause_bank_excluded[j, k, 1] += 1
                if clause_bank_excluded[j, k, 1] >= number_of_states//2:
                    clause_bank_included[j, clause_bank_included_length[j], 0] = clause_bank_excluded[j, k, 0]
                    clause_bank_included[j, clause_bank_included_length[j], 1] = clause_bank_excluded[j, k, 1]
                    clause_bank_included_length[j] += 1

                    clause_bank_excluded_length[j] -= 1
                    clause_bank_excluded[j, k, 0] = clause_bank_excluded[j, clause_bank_excluded_length[j], 0]       
                    clause_bank_excluded[j, k, 1] = clause_bank_excluded[j, clause_bank_excluded_length[j], 1]

class ClauseBankSparse:
    def __init__(self, X, number_of_clauses, number_of_states, patch_dim,
                 batching=True, incremental=True):
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

        self.feedback_to_ta = np.ascontiguousarray(np.empty((int(self.number_of_ta_chunks)), dtype=np.uint32))

        self.packed_X = np.ascontiguousarray(np.empty(self.number_of_literals, dtype=np.uint32))

        self.Xi = np.ascontiguousarray(np.zeros(self.number_of_ta_chunks, dtype=np.uint32))
        for k in range(self.number_of_features, self.number_of_literals):
            chunk = k // 32
            pos = k % 32
            self.Xi[chunk] |= (1 << pos)
        
        self.initialize_clauses()

    def initialize_clauses(self):
        self.clause_bank_included = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals, 2), dtype=np.uint32)) # Contains index and state of included literals, none at start
        self.clause_bank_included_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint32)) 

        self.clause_bank_excluded = np.ascontiguousarray(np.zeros((self.number_of_clauses, self.number_of_literals, 2), dtype=np.uint32)) # Contains index and state of excluded literals
        self.clause_bank_excluded_length = np.ascontiguousarray(np.zeros(self.number_of_clauses, dtype=np.uint32)) # All literals excluded at start
        self.clause_bank_excluded_length[:] = self.number_of_literals
        self.clause_bank_excluded[:,:,0] = np.arange(self.number_of_literals, dtype=np.uint32) # Initialize clause literals with increasing index
        self.clause_bank_excluded[:,:,1] = self.number_of_states // 2 - 1 # Initialize excluded literals in least forgotten state
 
    def calculate_clause_outputs_predict(self, encoded_X, e):
        if not self.batching:
            prepare_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)
            calculate_clause_outputs_predict_numba(self.Xi, self.number_of_clauses, self.clause_output, self.clause_bank_included, self.clause_bank_included_length)
            restore_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)
            return self.clause_output

        if e % 32 == 0:
            pack_X_numba(encoded_X.indptr, encoded_X.indices, e, self.packed_X, self.number_of_literals)

            calculate_clause_outputs_predict_packed_X_numba(self.packed_X, self.number_of_clauses, self.clause_output_batch, self.clause_bank_included, self.clause_bank_included_length)
        unpack_clause_output_numba(e, self.clause_output, self.clause_output_batch, self.number_of_clauses)
        return self.clause_output

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        prepare_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)
        calculate_clause_outputs_update_numba(literal_active, self.Xi, self.number_of_clauses, self.clause_output, self.clause_bank_included, self.clause_bank_included_length)
        restore_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)
        return self.clause_output

    def type_i_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e):
        prepare_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)
        type_i_feedback_numba(update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, self.feedback_to_ta, self.Xi, self.number_of_clauses, self.number_of_literals, self.number_of_states, self.clause_bank_included,
                        self.clause_bank_included_length, self.clause_bank_excluded, self.clause_bank_excluded_length)
        restore_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        prepare_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)
        type_ii_feedback_numba(update_p, clause_active, literal_active, self.Xi, self.number_of_clauses, self.number_of_states, self.clause_bank_included,
                    self.clause_bank_included_length, self.clause_bank_excluded, self.clause_bank_excluded_length)
        restore_Xi_numba(encoded_X.indices[encoded_X.indptr[e]:encoded_X.indptr[e+1]], self.Xi, self.number_of_features)

    def number_of_include_actions(self, clause):
        return self.clause_bank_included_length[clause]

    def prepare_X(self, X):
        return csr_matrix(X, dtype=np.uint32)
