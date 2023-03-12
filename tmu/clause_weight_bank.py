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

import numpy as np

import tmu.tools


class ClauseWeightBank:
    def __init__(self, X, number_of_outputs, number_of_clauses, number_of_state_bits_ta, number_of_state_bits_ind, patch_dim,
                 batch_size=100, incremental=True):
        self.number_of_outputs = int(number_of_outputs)
        self.number_of_clauses = int(number_of_clauses)
        self.number_of_state_bits_ta = int(number_of_state_bits_ta)
        self.number_of_state_bits_ind = int(number_of_state_bits_ind)
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.incremental = incremental

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
        self.number_of_ta_chunks = int((self.number_of_literals - 1) / 32 + 1)

        self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)

        self.clause_output_batch = np.ascontiguousarray(
            np.empty((int(self.number_of_clauses * batch_size)), dtype=np.uint32))
        self.cob_p = ffi.cast("unsigned int *", self.clause_output_batch.ctypes.data)

        self.clause_and_target = np.ascontiguousarray(
            np.zeros((int(self.number_of_clauses * self.number_of_ta_chunks)), dtype=np.uint32))
        self.ct_p = ffi.cast("unsigned int *", self.clause_and_target.ctypes.data)

        self.clause_output_patchwise = np.ascontiguousarray(
            np.empty((int(self.number_of_clauses * self.number_of_patches)), dtype=np.uint32))
        self.cop_p = ffi.cast("unsigned int *", self.clause_output_patchwise.ctypes.data)

        self.feedback_to_ta = np.ascontiguousarray(np.empty((self.number_of_ta_chunks), dtype=np.uint32))
        self.ft_p = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)

        self.output_one_patches = np.ascontiguousarray(np.empty(self.number_of_patches, dtype=np.uint32))
        self.o1p_p = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)

        self.literal_clause_count = np.ascontiguousarray(np.empty((int(self.number_of_literals)), dtype=np.uint32))
        self.lcc_p = ffi.cast("unsigned int *", self.literal_clause_count.ctypes.data)

        self.literal_clause_map = np.ascontiguousarray(
                np.empty((int(self.number_of_literals * self.number_of_clauses)), dtype=np.uint32))
        self.lcm_p = ffi.cast("unsigned int *", self.literal_clause_map.ctypes.data)

        self.literal_clause_map_pos = np.ascontiguousarray(
            np.empty((int(self.number_of_literals)), dtype=np.uint32))
        self.lcmp_p = ffi.cast("unsigned int *", self.literal_clause_map_pos.ctypes.data)

        self.false_literals_per_clause = np.ascontiguousarray(
            np.empty((int(self.number_of_clauses * self.number_of_patches)), dtype=np.uint32))
        self.flpc_p = ffi.cast("unsigned int *", self.false_literals_per_clause.ctypes.data)

        self.previous_xi = np.ascontiguousarray(
            np.empty((int(self.number_of_ta_chunks) * int(self.number_of_patches)), dtype=np.uint32))
        self.previous_xi_p = ffi.cast("unsigned int *", self.previous_xi.ctypes.data)

        self.initialize_clauses()

    def initialize_clauses(self):
        self.clause_bank = np.empty((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
                                    dtype=np.uint32)
        self.clause_bank[:, :, 0:self.number_of_state_bits_ta - 1] = np.uint32(~0)
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta)))
        self.cb_p = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)

        self.clause_bank_ind = np.empty(
            (self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ind), dtype=np.uint32)
        self.clause_bank_ind[:, :, :] = np.uint32(~0)
        self.clause_bank_ind = np.ascontiguousarray(self.clause_bank_ind.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ind)))
        self.cbi_p = ffi.cast("unsigned int *", self.clause_bank_ind.ctypes.data)

        self.weight_bank = np.ascontiguousarray(np.random.choice([-1, 1], size=(self.number_of_outputs, self.number_of_clauses)).astype(np.int32))
        self.wb_p = ffi.cast("int *", self.weight_bank.ctypes.data)

        self.incremental_clause_evaluation_initialized = False

    def calculate_clause_outputs(self, literal_active, encoded_X, e, update):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)

        if not self.incremental:
            if update:
                lib.cwb_calculate_clause_outputs_update(self.cb_p, self.number_of_clauses, self.number_of_literals,
                                               self.number_of_state_bits_ta, self.number_of_patches, self.co_p, la_p,
                                               xi_p)
            else:
                lib.cwb_calculate_clause_outputs_predict(self.cb_p, self.number_of_clauses, self.number_of_literals,
                                                    self.number_of_state_bits_ta, self.number_of_patches, self.co_p,
                                                    xi_p)
            return self.clause_output

        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if self.incremental_clause_evaluation_initialized == False or (update and (e % self.batch_size == 0)):
            lib.cwb_initialize_incremental_clause_calculation(self.cb_p, self.lcm_p, self.lcmp_p, self.flpc_p,
                                                             self.number_of_clauses, self.number_of_literals,
                                                             self.number_of_state_bits_ta, self.previous_xi_p, int(update))

            self.incremental_clause_evaluation_initialized = True

        if e % self.batch_size == 0:
            lib.cwb_calculate_clause_outputs_incremental_batch(self.lcm_p, self.lcmp_p, self.flpc_p,
                                                              self.number_of_clauses, self.number_of_literals,
                                                              self.number_of_patches, self.cob_p, self.previous_xi_p,
                                                              xi_p, np.minimum(self.batch_size, encoded_X.shape[0] - e))

        return self.clause_output_batch.reshape((self.batch_size, self.number_of_clauses))[e % self.batch_size, :]

    def calculate_clause_outputs_predict(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if not self.incremental:
            lib.cb_calculate_clause_outputs_predict(self.cb_p, self.number_of_clauses, self.number_of_literals,
                                                    self.number_of_state_bits_ta, self.number_of_patches, self.co_p,
                                                    xi_p)
            return self.clause_output

        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if self.incremental_clause_evaluation_initialized == False:
            self.literal_clause_map = np.ascontiguousarray(
                np.empty((int(self.number_of_literals * self.number_of_clauses)), dtype=np.uint32))
            self.lcm_p = ffi.cast("unsigned int *", self.literal_clause_map.ctypes.data)

            self.literal_clause_map_pos = np.ascontiguousarray(
                np.empty((int(self.number_of_literals)), dtype=np.uint32))
            self.lcmp_p = ffi.cast("unsigned int *", self.literal_clause_map_pos.ctypes.data)

            self.false_literals_per_clause = np.ascontiguousarray(
                np.empty((int(self.number_of_clauses * self.number_of_patches)), dtype=np.uint32))
            self.flpc_p = ffi.cast("unsigned int *", self.false_literals_per_clause.ctypes.data)

            self.previous_xi = np.ascontiguousarray(
                np.empty((int(self.number_of_ta_chunks) * int(self.number_of_patches)), dtype=np.uint32))
            self.previous_xi_p = ffi.cast("unsigned int *", self.previous_xi.ctypes.data)

            lib.cb_initialize_incremental_clause_calculation(self.cb_p, self.lcm_p, self.lcmp_p, self.flpc_p,
                                                             self.number_of_clauses, self.number_of_literals,
                                                             self.number_of_state_bits_ta, self.previous_xi_p)

            self.incremental_clause_evaluation_initialized = True

        if e % self.batch_size == 0:
            lib.cb_calculate_clause_outputs_incremental_batch(self.lcm_p, self.lcmp_p, self.flpc_p,
                                                              self.number_of_clauses, self.number_of_literals,
                                                              self.number_of_patches, self.cob_p, self.previous_xi_p,
                                                              xi_p, np.minimum(self.batch_size, encoded_X.shape[0] - e))

        return self.clause_output_batch.reshape((self.batch_size, self.number_of_clauses))[e % self.batch_size, :]

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)
        lib.cb_calculate_clause_outputs_update(self.cb_p, self.number_of_clauses, self.number_of_literals,
                                               self.number_of_state_bits_ta, self.number_of_patches, self.co_p, la_p,
                                               xi_p)
        return self.clause_output

    def calculate_clause_outputs_patchwise(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        lib.cwb_calculate_clause_outputs_patchwise(self.cb_p, self.number_of_clauses, self.number_of_literals,
                                                  self.number_of_state_bits_ta, self.number_of_patches, self.cop_p,
                                                  xi_p)
        return self.clause_output_patchwise

    def type_i_and_ii_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e, y, output_literal_index, autoencoder=0):
        xi_p = ffi.cast("unsigned int *", encoded_X[e,:].ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)
        up_p = ffi.cast("float *", update_p.ctypes.data)
        y_p = ffi.cast("unsigned int *", y.ctypes.data)
        oli_p = ffi.cast("unsigned int *", output_literal_index.ctypes.data)

        lib.cwb_type_i_and_ii_feedback(self.cb_p, self.wb_p, self.ft_p, self.o1p_p, self.number_of_outputs, self.number_of_clauses, self.number_of_literals,
                               self.number_of_state_bits_ta, self.number_of_patches, up_p, s,
                               boost_true_positive_feedback, max_included_literals, ca_p, la_p, xi_p, y_p, oli_p, autoencoder)

    def type_iii_feedback(self, update_p, d, clause_active, literal_active, encoded_X, e, target):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)
        lib.cwb_type_iii_feedback(self.cb_p, self.cbi_p, self.ct_p, self.o1p_p, self.number_of_clauses,
                                 self.number_of_literals, self.number_of_state_bits_ta, self.number_of_state_bits_ind,
                                 self.number_of_patches, update_p, d, ca_p, la_p, xi_p, target)

    def calculate_literal_clause_frequency(self, clause_active):
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cwb_calculate_literal_frequency(self.cb_p, self.number_of_clauses, self.number_of_literals,
                                           self.number_of_state_bits_ta, ca_p, self.lcc_p)
        return self.literal_clause_count

    def number_of_include_actions(self, clause):
        return lib.cwb_number_of_include_actions(self.cb_p, clause, self.number_of_literals,
                                                self.number_of_state_bits_ta)

    def get_weights(self):
        return self.weight_bank

    def get_ta_action(self, clause, ta):
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(
            clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + ta_chunk * self.number_of_state_bits_ta + self.number_of_state_bits_ta - 1)
        return (self.clause_bank[pos] & (1 << chunk_pos)) > 0

    def get_ta_state(self, clause, ta):
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(
            clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + ta_chunk * self.number_of_state_bits_ta)
        state = 0
        for b in range(self.number_of_state_bits_ta):
            if self.clause_bank[pos + b] & (1 << chunk_pos) > 0:
                state |= (1 << b)
        return state

    def set_ta_state(self, clause, ta, state):
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(
            clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + ta_chunk * self.number_of_state_bits_ta)
        for b in range(self.number_of_state_bits_ta):
            if state & (1 << b) > 0:
                self.clause_bank[pos + b] |= (1 << chunk_pos)
            else:
                self.clause_bank[pos + b] &= ~(1 << chunk_pos)

    def prepare_X(self, X):
        return tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                self.patch_dim, 0)

    def prepare_autoencoder_examples(self, X_csr, X_csc, active_output, accumulation):
        return tmu.tools.produce_autoencoder_examples(X_csr, X_csc, active_output, accumulation,
                                                      self.number_of_ta_chunks)