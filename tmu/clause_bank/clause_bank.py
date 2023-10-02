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
    ptr_clause_and_target = None  # _cffi_backend._CDataBase
    cop_p = None  # _cffi_backend._CDataBase
    ptr_feedback_to_ta = None  # _cffi_backend._CDataBase
    ptr_output_one_patches = None  # _cffi_backend._CDataBase
    ptr_literal_clause_count = None  # _cffi_backend._CDataBase
    ptr_actions = None  # _cffi_backend._CDataBase

    def __init__(
            self,
            seed: int,
            d: float,
            number_of_state_bits_ind: int,
            number_of_state_bits_ta: int,
            batch_size: int,
            incremental: bool,
            **kwargs
    ):
        super().__init__(seed=seed, **kwargs)

        self.d = d
        assert isinstance(number_of_state_bits_ta, int)
        self.number_of_state_bits_ta = number_of_state_bits_ta
        self.number_of_state_bits_ind = int(number_of_state_bits_ind)
        self.batch_size = batch_size
        self.incremental = incremental

        self.clause_output = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")
        self.clause_output_batch = np.empty(self.number_of_clauses * batch_size, dtype=np.uint32, order="c")
        self.clause_and_target = np.zeros(self.number_of_clauses * self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.clause_output_patchwise = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.uint32, order="c")
        self.feedback_to_ta = np.empty(self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.output_one_patches = np.empty(self.number_of_patches, dtype=np.uint32, order="c")
        self.literal_clause_count = np.empty(self.number_of_literals, dtype=np.uint32, order="c")


        self.type_ia_feedback_counter = np.zeros(self.number_of_clauses, dtype=np.uint32, order="c")

        # Incremental Clause Evaluation
        self.literal_clause_map = np.empty(
            (int(self.number_of_literals * self.number_of_clauses)),
            dtype=np.uint32,
            order="c"
        )
        self.literal_clause_map_pos = np.empty(
            (int(self.number_of_literals)),
            dtype=np.uint32,
            order="c"
        )
        self.false_literals_per_clause = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )
        self.previous_xi = np.empty(
            int(self.number_of_ta_chunks) * int(self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )

        self.initialize_clauses()

        # Finally, map numpy arrays to CFFI compatible pointers.
        self._cffi_init()

        # Set pcg32 seed
        if self.seed is not None:
            assert isinstance(self.seed, int), "Seed must be a integer"

            lib.pcg32_seed(self.seed)
            lib.xorshift128p_seed(self.seed)

    def _cffi_init(self):
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)  # clause_output
        self.cob_p = ffi.cast("unsigned int *", self.clause_output_batch.ctypes.data)  # clause_output_batch
        self.ptr_clause_and_target = ffi.cast("unsigned int *", self.clause_and_target.ctypes.data)  # clause_and_target
        self.cop_p = ffi.cast("unsigned int *", self.clause_output_patchwise.ctypes.data)  # clause_output_patchwise
        self.ptr_feedback_to_ta = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)  # feedback_to_ta
        self.ptr_output_one_patches = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)  # output_one_patches
        self.ptr_literal_clause_count = ffi.cast("unsigned int *", self.literal_clause_count.ctypes.data)  # literal_clause_count
        self.tiafc_p = ffi.cast("unsigned int *", self.type_ia_feedback_counter.ctypes.data)  # literal_clause_count

        # Clause Initialization
        self.ptr_ta_state = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)
        self.ptr_ta_state_ind = ffi.cast("unsigned int *", self.clause_bank_ind.ctypes.data)


        # Action Initialization
        self.ptr_actions = ffi.cast("unsigned int *", self.actions.ctypes.data)

        # Incremental Clause Evaluation Initialization
        self.lcm_p = ffi.cast("unsigned int *", self.literal_clause_map.ctypes.data)
        self.lcmp_p = ffi.cast("unsigned int *", self.literal_clause_map_pos.ctypes.data)
        self.flpc_p = ffi.cast("unsigned int *", self.false_literals_per_clause.ctypes.data)
        self.previous_xi_p = ffi.cast("unsigned int *", self.previous_xi.ctypes.data)

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
                self.ptr_ta_state,
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

            lib.cb_initialize_incremental_clause_calculation(
                self.ptr_ta_state,
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
            self.ptr_ta_state,
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
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            self.cop_p,
            xi_p
        )

        return self.clause_output_patchwise

    def type_i_feedback(
        self,
        update_p,
        clause_active,
        literal_active,
        encoded_X,
        e
    ):
        ptr_xi = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_literal_active = ffi.cast("unsigned int *", literal_active.ctypes.data)
        lib.cb_type_i_feedback(
            self.ptr_ta_state,
            self.ptr_feedback_to_ta,
            self.ptr_output_one_patches,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            update_p,
            self.s,
            self.boost_true_positive_feedback,
            self.reuse_random_feedback,
            self.max_included_literals,
            ptr_clause_active,
            ptr_literal_active,
            ptr_xi
        )

        self.incremental_clause_evaluation_initialized = False

    def type_ii_feedback(
        self,
        update_p,
        clause_active,
        literal_active,
        encoded_X,
        e
    ):
        ptr_xi = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_literal_active = ffi.cast("unsigned int *", literal_active.ctypes.data)

        lib.cb_type_ii_feedback(
            self.ptr_ta_state,
            self.ptr_output_one_patches,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            update_p,
            ptr_clause_active,
            ptr_literal_active,
            ptr_xi
        )

        self.incremental_clause_evaluation_initialized = False


    def type_iii_feedback(
            self,
            update_p,
            clause_active,
            literal_active,
            encoded_X,
            e,
            target
    ):
        ptr_xi = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_literal_active = ffi.cast("unsigned int *", literal_active.ctypes.data)

        lib.cb_type_iii_feedback(
            self.ptr_ta_state,
            self.ptr_ta_state_ind,
            self.ptr_clause_and_target,
            self.ptr_output_one_patches,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_state_bits_ind,
            self.number_of_patches,
            update_p,
            self.d,
            ptr_clause_active,
            ptr_literal_active,
            ptr_xi,
            target
        )

        self.incremental_clause_evaluation_initialized = False

    def calculate_literal_clause_frequency(
            self,
            clause_active
    ):
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cb_calculate_literal_frequency(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ptr_clause_active,
            self.ptr_literal_clause_count
        )
        return self.literal_clause_count

    def included_literals(self):
        lib.cb_included_literals(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.ptr_actions
        )
        return self.actions

    def get_literals(self, independent=False):

        result = np.zeros((self.number_of_clauses, self.number_of_literals), dtype=np.uint32, order="c")
        result_p = ffi.cast("unsigned int *", result.ctypes.data)
        lib.cb_get_literals(
            self.ptr_ta_state_ind if independent else self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            result_p
        )
        return result

    def calculate_independent_literal_clause_frequency(self, clause_active):
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cb_calculate_literal_frequency(
            self.ptr_ta_state_ind,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ca_p,
            self.ptr_literal_clause_count
        )
        return self.literal_clause_count

    def number_of_include_actions(
            self,
            clause
    ):
        return lib.cb_number_of_include_actions(
            self.ptr_ta_state,
            clause,
            self.number_of_literals,
            self.number_of_state_bits_ta
        )

    def prepare_X(
            self,
            X
    ):
        return tmu.tools.encode(
            X,
            X.shape[0],
            self.number_of_patches,
            self.number_of_ta_chunks,
            self.dim,
            self.patch_dim,
            0
        )

    def prepare_X_autoencoder(
            self,
            X_csr,
            X_csc,
            active_output
    ):
        X = np.ascontiguousarray(np.empty(int(self.number_of_ta_chunks), dtype=np.uint32))
        return X_csr, X_csc, active_output, X

    def produce_autoencoder_example(
            self,
            encoded_X,
            target,
            target_true_p,
            accumulation
    ):
        (X_csr, X_csc, active_output, X) = encoded_X

        target_value = self.rng.random() <= target_true_p

        lib.tmu_produce_autoencoder_example(ffi.cast("unsigned int *", active_output.ctypes.data), active_output.shape[0],
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indptr).ctypes.data),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indices).ctypes.data),
                                             int(X_csr.shape[0]),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indptr).ctypes.data),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indices).ctypes.data),
                                             int(X_csc.shape[1]),
                                             ffi.cast("unsigned int *", X.ctypes.data),
                                             int(target),
                                             int(target_value),
                                             int(accumulation))

        return X.reshape((1, -1)), target_value
