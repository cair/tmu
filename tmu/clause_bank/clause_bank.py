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
from tmu.clause_bank.base_clause_bank import BaseClauseBank

import numpy as np


class ClauseBank(BaseClauseBank):
    clause_bank: np.ndarray
    incremental_clause_evaluation_initialized: bool
    co_p = None  # _cffi_backend._CDataBase
    cob_p = None  # _cffi_backend._CDataBase
    ct_p = None  # _cffi_backend._CDataBase
    cop_p = None  # _cffi_backend._CDataBase
    ft_p = None  # _cffi_backend._CDataBase
    o1p_p = None  # _cffi_backend._CDataBase
    lcc_p = None  # _cffi_backend._CDataBase
    ac_p = None  # _cffi_backend._CDataBase

    def __init__(
            self,
            number_of_state_bits_ind: int,
            batch_size: int = 100,
            incremental: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.number_of_state_bits_ind = int(number_of_state_bits_ind)
        self.batch_size = batch_size
        self.incremental = incremental

        self.clause_output = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")
        self.clause_output_batch = np.empty(self.number_of_clauses * batch_size, dtype=np.uint32, order="c")
        self.clause_and_target = np.zeros(self.number_of_clauses * self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.clause_output_patchwise = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.uint32,
                                                order="c")
        self.feedback_to_ta = np.empty(self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.output_one_patches = np.empty(self.number_of_patches, dtype=np.uint32, order="c")
        self.literal_clause_count = np.empty(self.number_of_literals, dtype=np.uint32, order="c")

        self.initialize_clauses()

        # Finally, map numpy arrays to CFFI compatible pointers.
        self._cffi_init()

    def _cffi_init(self):
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)  # clause_output
        self.cob_p = ffi.cast("unsigned int *", self.clause_output_batch.ctypes.data)  # clause_output_batch
        self.ct_p = ffi.cast("unsigned int *", self.clause_and_target.ctypes.data)  # clause_and_target
        self.cop_p = ffi.cast("unsigned int *", self.clause_output_patchwise.ctypes.data)  # clause_output_patchwise
        self.ft_p = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)  # feedback_to_ta
        self.o1p_p = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)  # output_one_patches
        self.lcc_p = ffi.cast("unsigned int *", self.literal_clause_count.ctypes.data)  # literal_clause_count

        # Clause Initialization
        self.cb_p = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)
        self.cbi_p = ffi.cast("unsigned int *", self.clause_bank_ind.ctypes.data)

        # Action Initialization
        self.ac_p = ffi.cast("unsigned int *", self.actions.ctypes.data)

    def initialize_clauses(self):
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="c"
        )

        self.clause_bank[:, :, 0: self.number_of_state_bits_ta - 1] = np.uint32(~0)
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta)))

        self.actions = np.ascontiguousarray(np.zeros(self.number_of_ta_chunks, dtype=np.uint32))

        self.clause_bank_ind = np.empty(
            (self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ind), dtype=np.uint32)
        self.clause_bank_ind[:, :, :] = np.uint32(~0)
        self.clause_bank_ind = np.ascontiguousarray(self.clause_bank_ind.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ind)))

        self.incremental_clause_evaluation_initialized = False

    def calculate_clause_outputs_predict(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if not self.incremental:
            lib.cb_calculate_clause_outputs_predict(
                self.cb_p,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.co_p,
                xi_p
            )
            return self.clause_output

        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if not self.incremental_clause_evaluation_initialized:
            self.literal_clause_map = np.empty(
                (int(self.number_of_literals * self.number_of_clauses)),
                dtype=np.uint32,
                order="c"
            )
            self.lcm_p = ffi.cast("unsigned int *", self.literal_clause_map.ctypes.data)

            self.literal_clause_map_pos = np.empty(
                (int(self.number_of_literals)),
                dtype=np.uint32,
                order="c"
            )
            self.lcmp_p = ffi.cast("unsigned int *", self.literal_clause_map_pos.ctypes.data)

            self.false_literals_per_clause = np.empty(
                int(self.number_of_clauses * self.number_of_patches),
                dtype=np.uint32,
                order="c"
            )
            self.flpc_p = ffi.cast("unsigned int *", self.false_literals_per_clause.ctypes.data)

            self.previous_xi = np.empty(
                int(self.number_of_ta_chunks) * int(self.number_of_patches),
                dtype=np.uint32,
                order="c"
            )
            self.previous_xi_p = ffi.cast("unsigned int *", self.previous_xi.ctypes.data)

            lib.cb_initialize_incremental_clause_calculation(
                self.cb_p,
                self.lcm_p,
                self.lcmp_p,
                self.flpc_p,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.previous_xi_p
            )

            self.incremental_clause_evaluation_initialized = True

        if e % self.batch_size == 0:
            lib.cb_calculate_clause_outputs_incremental_batch(
                self.lcm_p,
                self.lcmp_p,
                self.flpc_p,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_patches,
                self.cob_p,
                self.previous_xi_p,
                xi_p,
                np.minimum(self.batch_size, encoded_X.shape[0] - e)
            )

        return self.clause_output_batch.reshape((self.batch_size, self.number_of_clauses))[e % self.batch_size, :]

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)

        lib.cb_calculate_clause_outputs_update(
            self.cb_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            self.co_p,
            la_p,
            xi_p
        )

        return self.clause_output

    def calculate_clause_outputs_patchwise(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        lib.cb_calculate_clause_outputs_patchwise(
            self.cb_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            self.cop_p,
            xi_p
        )

        return self.clause_output_patchwise

    def type_i_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)
        lib.cb_type_i_feedback(
            self.cb_p,
            self.ft_p,
            self.o1p_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            update_p,
            s,
            boost_true_positive_feedback,
            max_included_literals,
            ca_p,
            la_p,
            xi_p
        )

        self.incremental_clause_evaluation_initialized = False

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)

        lib.cb_type_ii_feedback(
            self.cb_p,
            self.o1p_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            update_p,
            ca_p,
            la_p,
            xi_p
        )

        self.incremental_clause_evaluation_initialized = False

    def type_iii_feedback(self, update_p, d, clause_active, literal_active, encoded_X, e, target):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)

        lib.cb_type_iii_feedback(
            self.cb_p,
            self.cbi_p,
            self.ct_p,
            self.o1p_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_state_bits_ind,
            self.number_of_patches,
            update_p,
            d,
            ca_p,
            la_p,
            xi_p,
            target
        )

        self.incremental_clause_evaluation_initialized = False

    def calculate_literal_clause_frequency(self, clause_active):
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cb_calculate_literal_frequency(
            self.cb_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ca_p,
            self.lcc_p
        )
        return self.literal_clause_count

    def included_literals(self):
        lib.cb_included_literals(
            self.cb_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.ac_p
        )
        return self.actions

    def number_of_include_actions(self, clause):
        return lib.cb_number_of_include_actions(
            self.cb_p,
            clause,
            self.number_of_literals,
            self.number_of_state_bits_ta
        )

    def prepare_X(self, X):
        return tmu.tools.encode(
            X,
            X.shape[0],
            self.number_of_patches,
            self.number_of_ta_chunks,
            self.dim,
            self.patch_dim,
            0
        )
