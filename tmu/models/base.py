# Copyright (c) 2023 Ole-Christoffer Granmo
import collections
import typing
from collections.abc import Mapping, Iterable

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
import numpy as np
from scipy.sparse import csr_matrix
from tmu.weight_bank import WeightBank


def _validate_input_dtype(d: np.ndarray):
    if d.dtype is not np.uint32:
        raise RuntimeError(f"The data input is of type {d.dtype}, but should be {np.uint32}")


class TMBasis:

    weight_banks: typing.List[WeightBank]
    clause_banks: typing.List[typing.Union["ClauseBank", "ClauseBankCUDA"]]

    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            confidence_driven_updating=False,
            type_i_ii_ratio=1.0,
            type_iii_feedback=False,
            focused_negative_sampling=False,
            output_balancing=False,
            d=200.0,
            platform='CPU',
            patch_dim=None,
            feature_negation=True,
            boost_true_positive_feedback=1,
            max_included_literals=None,
            number_of_state_bits_ta=8,
            number_of_state_bits_ind=8,
            weighted_clauses=False,
            clause_drop_p=0.0,
            literal_drop_p=0.0,
            batch_size=100,
            incremental=True,
            absorbing=-1,
            literal_sampling=1.0,
            feedback_rate_excluded_literals=1,
            literal_insertion_state = 0
    ):
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

        self.type_iii_feedback = type_iii_feedback
        self.focused_negative_sampling = focused_negative_sampling
        self.output_balancing = output_balancing
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
        self.absorbing = absorbing
        self.literal_sampling = literal_sampling
        self.feedback_rate_excluded_literals = feedback_rate_excluded_literals
        self.literal_insertion_state = literal_insertion_state
        self.initialized = False

        # TODO - Change to checksum
        self.X_train = np.zeros(0, dtype=np.uint32)
        self.X_test = np.zeros(0, dtype=np.uint32)

        self.weight_banks = []
        self.clause_banks = []

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

    def init(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError("init(self, X: np.ndarray, Y: np.ndarray)")
