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
import logging
import typing
import numpy as np
from scipy.sparse import csr_matrix

from tmu.weight_bank import WeightBank
from tmu.clause_bank.clause_bank import ClauseBank
from tmu.clause_bank.clause_bank_cuda import ClauseBankCUDA
from tmu.clause_bank.clause_bank_sparse import ClauseBankSparse
from tmu.util.sparse_clause_container import SparseClauseContainer

_LOGGER = logging.getLogger(__name__)

def _validate_input_dtype(d: np.ndarray):
    if d.dtype is not np.uint32:
        raise RuntimeError(f"The data input is of type {d.dtype}, but should be {np.uint32}")


class MultiWeightBankMixin:
    weight_banks: SparseClauseContainer

    def __init__(self, seed: int):
        self.weight_banks = SparseClauseContainer(random_seed=seed)


class MultiClauseBankMixin:
    clause_banks: SparseClauseContainer

    def __init__(self, seed: int):
        self.clause_banks = SparseClauseContainer(random_seed=seed)


class SingleWeightBankMixin:
    weight_bank: WeightBank

    def __init__(self):
        pass


class SingleClauseBankMixin:
    clause_bank: typing.Union[ClauseBank, ClauseBankSparse, ClauseBankCUDA]

    def __init__(self):
        pass


class TMBaseModel:
    number_of_classes: int

    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            confidence_driven_updating=False,
            type_i_ii_ratio=1.0,
            type_i_feedback=True,
            type_ii_feedback=True,
            type_iii_feedback=False,
            focused_negative_sampling=False,
            output_balancing=False,
            upsampling=1,
            d=200.0,
            platform='CPU',
            patch_dim=None,
            feature_negation=True,
            boost_true_positive_feedback=1,
            reuse_random_feedback=0,
            max_included_literals=None,
            number_of_state_bits_ta=8,
            number_of_state_bits_ind=8,
            weighted_clauses=False,
            clause_drop_p=0.0,
            literal_drop_p=0.0,
            literal_sampling=1.0,
            feedback_rate_excluded_literals=1,
            literal_insertion_state=0,
            batch_size=100,
            incremental=True,
            type_ia_ii_feedback_ratio=0,
            absorbing=-1,
            absorbing_include=None,
            absorbing_exclude=None,
            squared_weight_update_p=False,
            seed=None
    ):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.number_of_clauses = number_of_clauses
        self.number_of_state_bits_ta = number_of_state_bits_ta
        self.number_of_state_bits_ind = number_of_state_bits_ind
        self.T = int(T)
        self.s = s

        self.confidence_driven_updating = confidence_driven_updating

        if type_i_ii_ratio >= 1.0:
            self.type_i_p = 1.0
            self.type_ii_p = 1.0 / type_i_ii_ratio
        else:
            self.type_i_p = type_i_ii_ratio
            self.type_ii_p = 1.0

        self.type_i_feedback = type_i_feedback
        self.type_ii_feedback = type_ii_feedback
        self.type_iii_feedback = type_iii_feedback
        self.focused_negative_sampling = focused_negative_sampling
        self.output_balancing = output_balancing
        self.upsampling = upsampling
        self.d = d
        self.platform = platform
        self.patch_dim = patch_dim
        self.feature_negation = feature_negation
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.max_included_literals = max_included_literals
        self.weighted_clauses = weighted_clauses
        self.clause_drop_p = clause_drop_p
        self.literal_drop_p = literal_drop_p
        self.batch_size = batch_size
        self.incremental = incremental
        self.type_ia_ii_feedback_ratio = type_ia_ii_feedback_ratio
        self.absorbing = absorbing
        self.absorbing_include = absorbing_include
        self.absorbing_exclude = absorbing_exclude
        self.reuse_random_feedback = reuse_random_feedback
        self.initialized = False
        self.literal_sampling = literal_sampling
        self.feedback_rate_excluded_literals = feedback_rate_excluded_literals
        self.literal_insertion_state = literal_insertion_state
        self.squared_weight_update_p = squared_weight_update_p

        # TODO - Change to checksum
        self.X_train = np.zeros(0, dtype=np.uint32)
        self.X_test = np.zeros(0, dtype=np.uint32)

    def clause_co_occurrence(self, X, percentage=False):
        clause_outputs = csr_matrix(self.transform(X))
        if percentage:
            return clause_outputs.transpose().dot(clause_outputs).multiply(1.0 / clause_outputs.sum(axis=0))
        else:
            return clause_outputs.transpose().dot(clause_outputs)

    def transform(self, X):
        encoded_X = self.clause_bank.prepare_X(X)
        transformed_X = np.empty((X.shape[0], self.number_of_clauses), dtype=np.uint32)
        for e in range(X.shape[0]):
            transformed_X[e, :] = self.clause_bank.calculate_clause_outputs_predict(encoded_X, e)
        return transformed_X

    def transform_patchwise(self, X):
        encoded_X = self.clause_bank.prepare_X(X)

        transformed_X = np.empty(
            (X.shape[0], self.number_of_clauses * self.clause_bank.number_of_patches), dtype=np.uint32)
        for e in range(X.shape[0]):
            transformed_X[e, :] = self.clause_bank.calculate_clause_outputs_patchwise(encoded_X, e)
        return transformed_X.reshape(
            (X.shape[0], self.number_of_clauses, self.clause_bank.number_of_patches))

    def literal_clause_frequency(self):
        clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
        return self.clause_bank.calculate_literal_clause_frequency(clause_active)

    def get_ta_action(self, clause, ta, **kwargs):
        return self.clause_bank.get_ta_action(clause, ta)

    def get_ta_state(self, clause, ta, **kwargs):
        return self.clause_bank.get_ta_state(clause, ta)

    def set_ta_state(self, clause, ta, state, **kwargs):
        return self.clause_bank.set_ta_state(clause, ta, state)

    def fit(self, X, Y, *args, **kwargs):
        raise NotImplementedError("fit(self, X, Y, *args, **kwargs) is not implemented for your model")

    def predict(self, X, shuffle=True) -> np.ndarray:
        raise NotImplementedError("predict(self, X: np.ndarray")

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_before(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_num_classes(self, X: np.ndarray, Y: np.ndarray) -> int:
        pass

    def init(self, X: np.ndarray, Y: np.ndarray):
        if self.initialized:
            return

        self.init_before(X, Y)
        self.number_of_classes = self.init_num_classes(X, Y)
        self.init_clause_bank(X, Y)
        self.init_weight_bank(X, Y)
        self.init_after(X, Y)
        self.initialized = True

    def _build_cpu_bank(self, X: np.ndarray):
        from tmu.clause_bank.clause_bank import ClauseBank
        clause_bank_type = ClauseBank
        clause_bank_args = dict(
            X_shape=X.shape,
            d=self.d,
            s=self.s,
            boost_true_positive_feedback=self.boost_true_positive_feedback,
            reuse_random_feedback=self.reuse_random_feedback,
            max_included_literals=self.max_included_literals,
            number_of_clauses=self.number_of_clauses,
            number_of_state_bits_ta=self.number_of_state_bits_ta,
            number_of_state_bits_ind=self.number_of_state_bits_ind,
            patch_dim=self.patch_dim,
            batch_size=self.batch_size,
            incremental=self.incremental,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio,
            seed=self.seed,
        )
        return clause_bank_type, clause_bank_args

    def _build_gpu_bank(self, X: np.ndarray):
        from tmu.clause_bank.clause_bank_cuda import ClauseBankCUDA, cuda_installed

        if not cuda_installed:
            _LOGGER.warning("CUDA not installed, using CPU clause bank")
            return self._build_cpu_bank(X=X)

        clause_bank_type = ClauseBankCUDA
        clause_bank_args = dict(
            X_shape=X.shape,
            s=self.s,
            boost_true_positive_feedback=self.boost_true_positive_feedback,
            reuse_random_feedback=self.reuse_random_feedback,
            number_of_clauses=self.number_of_clauses,
            number_of_state_bits_ta=self.number_of_state_bits_ta,
            patch_dim=self.patch_dim,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio,
            max_included_literals=self.max_included_literals,
            seed=self.seed,
        )
        return clause_bank_type, clause_bank_args

    def _build_cpu_sparse_bank(self, X: np.ndarray):
        from tmu.clause_bank.clause_bank_sparse import ClauseBankSparse
        clause_bank_type = ClauseBankSparse
        clause_bank_args = dict(
            X_shape=X.shape,
            d=self.d,
            s=self.s,
            boost_true_positive_feedback=self.boost_true_positive_feedback,
            reuse_random_feedback=self.reuse_random_feedback,
            max_included_literals=self.max_included_literals,
            number_of_clauses=self.number_of_clauses,
            number_of_states=2 ** self.number_of_state_bits_ta,
            patch_dim=self.patch_dim,
            absorbing=self.absorbing,
            absorbing_exclude=self.absorbing_exclude,
            absorbing_include=self.absorbing_include,
            literal_sampling=self.literal_sampling,
            feedback_rate_excluded_literals=self.feedback_rate_excluded_literals,
            literal_insertion_state=self.literal_insertion_state,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio,
            seed=self.seed,
        )
        return clause_bank_type, clause_bank_args

    def build_clause_bank(self, X: np.ndarray):
        if self.platform == "CPU":
            clause_bank_type, clause_bank_args = self._build_cpu_bank(X=X)

        elif self.platform in ["GPU", "CUDA"]:
            clause_bank_type, clause_bank_args = self._build_gpu_bank(X=X)
        elif self.platform == "CPU_sparse":
            clause_bank_type, clause_bank_args = self._build_cpu_sparse_bank(X=X)
        else:
            raise NotImplementedError(f"Could not find platform of type {self.platform}.")

        return clause_bank_type, clause_bank_args
