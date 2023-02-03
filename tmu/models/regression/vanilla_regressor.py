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
from tmu.clause_bank import ClauseBank
from tmu.models.base import TMBasis
from tmu.weight_bank import WeightBank
import numpy as np


class TMRegressor(TMBasis):
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, feature_negation=True,
                 boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8,
                 weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim,
                         feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback,
                         max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta,
                         weighted_clauses=weighted_clauses, clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)

    def initialize(self, X, Y):
        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

        if self.platform == 'CPU':
            self.clause_bank = ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta,
                                          self.number_of_state_bits_ind, self.patch_dim)
        elif self.platform == 'CUDA':
            from clause_bank.clause_bank_cuda import ClauseBankCUDA
            self.clause_bank = ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
        else:
            print("Unknown Platform")
            sys.exit(-1)

        self.weight_bank = WeightBank(np.ones(self.number_of_clauses).astype(np.int32))

        if self.max_included_literals == None:
            self.max_included_literals = self.clause_bank.number_of_literals

    def fit(self, X, Y, shuffle=True):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_bank.prepare_X(X)
            self.X_train = X.copy()
        if (self.max_y - self.min_y) == 0:
            encoded_Y = np.ascontiguousarray(np.zeros(Y.shape[0], dtype=np.int32))
        else:
            encoded_Y = np.ascontiguousarray(((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32))

        # Drops clauses randomly based on clause drop probability
        clause_active = (np.random.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        # Literals are dropped based on literal drop probability
        literal_active = np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = np.random.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
        for k in range(self.clause_bank.number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] |= (1 << chunk_pos)

        if not self.feature_negation:
            for k in range(self.clause_bank.number_of_literals // 2, self.clause_bank.number_of_literals):
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] &= (~(1 << chunk_pos))

        literal_active = literal_active.astype(np.uint32)

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(shuffled_index)

        for e in shuffled_index:
            clause_outputs = self.clause_bank.calculate_clause_outputs_update(literal_active, self.encoded_X_train, e)

            pred_y = np.dot(clause_active * self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
            pred_y = np.clip(pred_y, 0, self.T)
            prediction_error = pred_y - encoded_Y[e];

            update_p = (1.0 * prediction_error / self.T) ** 2

            if pred_y < encoded_Y[e]:
                self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                                 self.max_included_literals, clause_active, literal_active,
                                                 self.encoded_X_train, e)
                if self.weighted_clauses:
                    self.weight_bank.increment(clause_outputs, update_p, clause_active, False)
            elif pred_y > encoded_Y[e]:
                self.clause_bank.type_ii_feedback(update_p, clause_active, literal_active, self.encoded_X_train, e)
                if self.weighted_clauses:
                    self.weight_bank.decrement(clause_outputs, update_p, clause_active, False)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0]))
        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            pred_y = np.dot(self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
            Y[e] = 1.0 * pred_y * (self.max_y - self.min_y) / (self.T) + self.min_y
        return Y

    def get_weight(self, clause):
        return self.weight_bank.get_weights()[clause]

    def set_weight(self, clause, weight):
        self.weight_banks.get_weights()[clause] = weight
