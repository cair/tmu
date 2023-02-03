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
import sys


class TMMultiChannelClassifier(TMBasis):
    def __init__(self, number_of_clauses, global_T, T, s, type_i_ii_ratio=1.0, platform='CPU', patch_dim=None,
                 feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None,
                 number_of_state_bits_ta=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, type_i_ii_ratio=type_i_ii_ratio, platform=platform,
                         patch_dim=patch_dim, feature_negation=feature_negation,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta,
                         weighted_clauses=weighted_clauses, clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)
        self.global_T = global_T

    def initialize(self, X, Y):
        self.number_of_classes = int(np.max(Y) + 1)

        if self.platform == 'CPU':
            self.clause_bank = ClauseBank(X[0], self.number_of_clauses, self.number_of_state_bits_ta,
                                          self.number_of_state_bits_ind, self.patch_dim)
        elif self.platform == 'CUDA':
            from clause_bank.clause_bank_cuda import ClauseBankCUDA
            self.clause_bank = ClauseBankCUDA(X[0], self.number_of_clauses, self.number_of_state_bits_ta,
                                              self.patch_dim)
        else:
            raise RuntimeError(f"Unknown platform of type: {self.platform}")

        self.weight_banks = []
        for i in range(self.number_of_classes):
            self.weight_banks.append(
                WeightBank(np.random.choice([-1, 1], size=self.number_of_clauses).astype(np.int32)))

        self.X_train = {}
        self.X_test = {}
        for c in range(X.shape[0]):
            self.X_train[c] = np.zeros(0, dtype=np.uint32)
            self.X_test[c] = np.zeros(0, dtype=np.uint32)

        self.encoded_X_train = {}
        self.encoded_X_test = {}

        if self.max_included_literals == None:
            self.max_included_literals = self.clause_bank.number_of_literals

    def fit(self, X, Y, shuffle=True):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        for c in range(X.shape[0]):
            if not np.array_equal(self.X_train[c], X[c]):
                self.encoded_X_train[c] = self.clause_bank.prepare_X(X[c])
                self.X_train[c] = X[c].copy()

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

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

        local_class_sum = np.empty(X.shape[0], dtype=np.int32)

        shuffled_index = np.arange(X.shape[1])
        if shuffle:
            np.random.shuffle(shuffled_index)

        for e in shuffled_index:
            target = Ym[e]

            clause_outputs = []
            for c in range(X.shape[0]):
                clause_outputs.append(
                    self.clause_bank.calculate_clause_outputs_update(literal_active, self.encoded_X_train[c], e).copy())

            global_class_sum = 0
            for c in range(X.shape[0]):
                local_class_sum[c] = np.dot(clause_active * self.weight_banks[target].get_weights(),
                                            clause_outputs[c]).astype(np.int32)
                local_class_sum[c] = np.clip(local_class_sum[c], -self.T, self.T)
                global_class_sum += local_class_sum[c]
            global_class_sum = np.clip(global_class_sum, -self.global_T[target][0], self.global_T[target][1])
            global_update_p = 1.0 * (self.global_T[target][1] - global_class_sum) / (
                        self.global_T[target][0] + self.global_T[target][1])

            for c in range(X.shape[0]):
                local_update_p = 1.0 * (self.T - local_class_sum[c]) / (2 * self.T)
                update_p = np.minimum(local_update_p, global_update_p)
                self.clause_bank.type_i_feedback(update_p * self.type_i_p, self.s[target],
                                                 self.boost_true_positive_feedback,
                                                 clause_active * (self.weight_banks[target].get_weights() >= 0),
                                                 literal_active, self.encoded_X_train[c], e)
                self.clause_bank.type_ii_feedback(update_p * self.type_ii_p,
                                                  clause_active * (self.weight_banks[target].get_weights() < 0),
                                                  literal_active, self.encoded_X_train[c], e)
                self.weight_banks[target].increment(clause_outputs[c], update_p, clause_active, True)

            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)

            global_class_sum = 0.0
            for c in range(X.shape[0]):
                local_class_sum[c] = np.dot(clause_active * self.weight_banks[not_target].get_weights(),
                                            clause_outputs[c]).astype(np.int32)
                local_class_sum[c] = np.clip(local_class_sum[c], -self.T, self.T)
                global_class_sum += local_class_sum[c]
            global_class_sum = np.clip(global_class_sum, -self.global_T[not_target][0], self.global_T[not_target][1])
            global_update_p = 1.0 * (self.global_T[not_target][0] + global_class_sum) / (
                        self.global_T[not_target][0] + self.global_T[not_target][1])

            for c in range(X.shape[0]):
                local_update_p = 1.0 * (self.T + local_class_sum[c]) / (2 * self.T)
                update_p = np.minimum(local_update_p, global_update_p)
                self.clause_bank.type_i_feedback(update_p * self.type_i_p, self.s[not_target],
                                                 self.boost_true_positive_feedback, self.max_included_literals,
                                                 clause_active * (self.weight_banks[not_target].get_weights() < 0),
                                                 literal_active, self.encoded_X_train[c], e)
                self.clause_bank.type_ii_feedback(update_p * self.type_ii_p,
                                                  clause_active * (self.weight_banks[not_target].get_weights() >= 0),
                                                  literal_active, self.encoded_X_train[c], e)
                self.weight_banks[not_target].decrement(clause_outputs[c], update_p, clause_active, True)
        return

    def predict(self, X):
        for c in range(X.shape[0]):
            if not np.array_equal(self.X_test[c], X[c]):
                self.encoded_X_test[c] = self.clause_bank.prepare_X(X[c])
                self.X_test[c] = X[c].copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[1], dtype=np.uint32))

        for e in range(X.shape[1]):
            max_class_sum = -sys.maxsize
            max_class = 0

            clause_outputs = []
            for c in range(X.shape[0]):
                clause_outputs.append(
                    self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test[c], e).copy())

            for i in range(self.number_of_classes):
                global_class_sum = 1
                for c in range(X.shape[0]):
                    local_class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs[c]).astype(np.int32)
                    local_class_sum = np.clip(local_class_sum, -self.T, self.T)
                    global_class_sum *= local_class_sum >= 0
                global_class_sum = np.clip(global_class_sum, -self.global_T[i][0], self.global_T[i][1])

                if global_class_sum > max_class_sum:
                    max_class_sum = global_class_sum
                    max_class = i
            Y[e] = max_class

        return Y

    def clause_precision(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()
        if positive_polarity == 0:
            positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
        else:
            positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)

        return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                        1.0 * true_positive_clause_outputs / (
                                    true_positive_clause_outputs + false_positive_clause_outputs))

    def clause_recall(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()

        if positive_polarity == 0:
            positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
        else:
            positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(axis=0)

        return true_positive_clause_outputs / Y[Y == the_class].shape[0]

    def get_weight(self, the_class, clause):
        return self.weight_banks[the_class].get_weights()[clause]

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight
