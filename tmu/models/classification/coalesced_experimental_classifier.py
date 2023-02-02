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

from tmu.clause_weight_bank import ClauseWeightBank
from tmu.models.base import TMBasis
import numpy as np

class TMCoalescedClassifier(TMBasis):
    def __init__(self, number_of_clauses, T, s, type_i_ii_ratio=1.0, type_iii_feedback=False,
                 focused_negative_sampling=False, output_balancing=False, d=200.0, platform='CPU', patch_dim=None,
                 feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None,
                 number_of_state_bits_ta=8, number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p=0.0,
                 literal_drop_p=0.0, incremental=False, batch_size=1):
        super().__init__(number_of_clauses, T, s, type_i_ii_ratio=type_i_ii_ratio, type_iii_feedback=type_iii_feedback,
                         focused_negative_sampling=focused_negative_sampling, output_balancing=output_balancing, d=d,
                         platform=platform, patch_dim=patch_dim, feature_negation=feature_negation,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta,
                         number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p, incremental=incremental, batch_size=batch_size)

    def initialize(self, X, Y):
        self.number_of_classes = int(np.max(Y) + 1)

        if self.platform == 'CPU':
            self.clause_bank = ClauseWeightBank(X, self.number_of_classes, self.number_of_clauses, self.number_of_state_bits_ta,
                                          self.number_of_state_bits_ind, self.patch_dim, incremental=self.incremental, batch_size=self.batch_size)
        else:
            raise RuntimeError(f"Unknown platform of type: {self.platform}")

        if self.max_included_literals == None:
            self.max_included_literals = self.clause_bank.number_of_literals

        self.update_p = np.empty(self.number_of_classes, dtype=np.float32)

        self.y = np.empty(self.number_of_classes, dtype=np.uint32)

    def fit(self, X, Y, shuffle=True):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_bank.prepare_X(X)
            self.X_train = X.copy()

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

        # Drops clauses randomly based on clause drop probability
        self.clause_active = (np.random.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        # Literals are dropped based on literal drop probability
        self.literal_active = np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = np.random.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
        for k in range(self.clause_bank.number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                self.literal_active[ta_chunk] |= (1 << chunk_pos)

        if not self.feature_negation:
            for k in range(self.clause_bank.number_of_literals // 2, self.clause_bank.number_of_literals):
                ta_chunk = k // 32
                chunk_pos = k % 32
                self.literal_active[ta_chunk] &= (~(1 << chunk_pos))

        self.literal_active = self.literal_active.astype(np.uint32)

        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs(self.literal_active, self.encoded_X_train, e, 1)
            for i in range(self.number_of_classes):
                self.update_p[i] = np.dot(self.clause_active * self.clause_bank.get_weights()[i],
                                               clause_outputs).astype(np.int32)
                self.update_p[i] = np.clip(self.update_p[i], -self.T, self.T)

                if i == Ym[e]:
                    self.update_p[i] = 1.0*(self.T - self.update_p[i]) / (2 * self.T)
                else:
                    self.update_p[i] = (1.0/(self.number_of_classes-1)) * (self.T + self.update_p[i]) / (2 * self.T)

            self.y[:] = 0
            self.y[Ym[e]] = 1
            self.clause_bank.type_i_and_ii_feedback(self.update_p, self.s, self.boost_true_positive_feedback,
                                         self.max_included_literals, self.clause_active,
                                         self.literal_active, self.encoded_X_train, e, self.y, self.y)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        literal_active = np.ones(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) * (~0)

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

        for e in range(X.shape[0]):
            max_class_sum = -self.T
            max_class = 0
            clause_outputs = self.clause_bank.calculate_clause_outputs(literal_active, self.encoded_X_test, e, 0)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.clause_bank.get_weights()[i], clause_outputs).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y

    def predict_individual(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros((X.shape[0], self.number_of_classes), dtype=np.uint32))

        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs(self.literal_active, self.encoded_X_test, e, 0)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.clause_banks.get_weights()[i], clause_outputs).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                Y[e, i] = (class_sum >= 0)
        return Y

    def clause_precision(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.clause_bank.get_weights()[the_class].get_weights()
        if positive_polarity == 0:
            positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
            false_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(axis=0)
        else:
            positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(axis=0)
            false_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)

        return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                        1.0 * true_positive_clause_outputs / (
                                    true_positive_clause_outputs + false_positive_clause_outputs))

    def clause_recall(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.clause_bank.get_weights()[the_class]

        if positive_polarity == 0:
            positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0) / \
                                           Y[Y == the_class].shape[0]
        else:
            positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(axis=0) / \
                                           Y[Y != the_class].shape[0]

        return true_positive_clause_outputs

    def get_weight(self, the_class, clause):
        return self.clause_bank.get_weights()[the_class, clause]

    def set_weight(self, the_class, clause, weight):
        self.clause_bank.get_weights()[the_class, clause] = weight

    def number_of_include_actions(self, clause):
        return self.clause_bank.number_of_include_actions(clause)
