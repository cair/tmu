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


class TMOneVsOneClassifier(TMBasis):
    def __init__(self, number_of_clauses, T, s, type_i_ii_ratio=1.0, platform='CPU', patch_dim=None,
                 feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None,
                 number_of_state_bits_ta=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, type_i_ii_ratio=type_i_ii_ratio, platform=platform,
                         patch_dim=patch_dim, feature_negation=feature_negation,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta,
                         weighted_clauses=weighted_clauses, clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)

    def initialize(self, X, Y):
        self.number_of_classes = int(np.max(Y) + 1)
        self.number_of_outputs = self.number_of_classes * (self.number_of_classes - 1)

        if self.platform == 'CPU':
            self.clause_bank = ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta,
                                          self.number_of_state_bits_ind, self.patch_dim)
        elif self.platform == 'CUDA':
            from clause_bank.clause_bank_cuda import ClauseBankCUDA
            self.clause_bank = ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
        else:
            raise RuntimeError(f"Unknown platform of type: {self.platform}")

        self.weight_banks = []
        for i in range(self.number_of_outputs):
            self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))

        if self.max_included_literals == None:
            self.max_included_literals = self.clause_bank.number_of_literals

    def fit(self, X, Y, shuffle=True):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_bank.prepare_X(X)
            self.X_train = X.copy()

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

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(shuffled_index)

        for e in shuffled_index:
            clause_outputs = self.clause_bank.calculate_clause_outputs_update(literal_active, self.encoded_X_train, e)

            target = Ym[e]
            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)

            output = target * (self.number_of_classes - 1) + not_target - (not_target > target)

            class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)
            update_p = (self.T - class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(update_p * self.type_i_p, self.s, self.boost_true_positive_feedback,
                                             self.max_included_literals,
                                             clause_active * (self.weight_banks[output].get_weights() >= 0),
                                             literal_active, self.encoded_X_train, e)
            self.clause_bank.type_ii_feedback(update_p * self.type_ii_p,
                                              clause_active * (self.weight_banks[output].get_weights() < 0),
                                              literal_active, self.encoded_X_train, e)
            self.weight_banks[output].increment(clause_outputs, update_p, clause_active, True)

            output = not_target * (self.number_of_classes - 1) + target - (target > not_target)

            class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)
            update_p = (self.T + class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(update_p * self.type_i_p, self.s, self.boost_true_positive_feedback,
                                             self.max_included_literals,
                                             clause_active * (self.weight_banks[output].get_weights() < 0),
                                             literal_active, self.encoded_X_train, e)
            self.clause_bank.type_ii_feedback(update_p * self.type_ii_p,
                                              clause_active * (self.weight_banks[output].get_weights() >= 0),
                                              literal_active, self.encoded_X_train, e)
            self.weight_banks[output].decrement(clause_outputs, update_p, clause_active, True)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)

            max_class_sum = -self.T * self.number_of_classes
            max_class = 0
            for i in range(self.number_of_classes):
                class_sum = 0
                for output in range(i * (self.number_of_classes - 1), (i + 1) * (self.number_of_classes - 1)):
                    output_sum = np.dot(self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
                    output_sum = np.clip(output_sum, -self.T, self.T)
                    class_sum += output_sum

                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y

    def clause_precision(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        precision = np.zeros((self.number_of_classes - 1, self.number_of_clauses))
        for i in range(self.number_of_classes - 1):
            other_class = i + (i >= the_class)
            output = the_class * (self.number_of_classes - 1) + i
            weights = self.weight_banks[output].get_weights()
            if positive_polarity:
                positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
                false_positive_clause_outputs = positive_clause_outputs[Y == other_class].sum(axis=0)
                precision[i] = np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                                        true_positive_clause_outputs / (
                                                    true_positive_clause_outputs + false_positive_clause_outputs))
            else:
                positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == other_class].sum(axis=0)
                false_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
                precision[i] = np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                                        true_positive_clause_outputs / (
                                                    true_positive_clause_outputs + false_positive_clause_outputs))

        return precision

    def clause_recall(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        recall = np.zeros((self.number_of_classes - 1, self.number_of_clauses))
        for i in range(self.number_of_classes - 1):
            other_class = i + (i >= the_class)
            output = the_class * (self.number_of_classes - 1) + i
            weights = self.weight_banks[output].get_weights()
            if positive_polarity:
                positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
                recall[i] = true_positive_clause_outputs / Y[Y == the_class].shape[0]
            else:
                positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == other_class].sum(axis=0)
                recall[i] = true_positive_clause_outputs / Y[Y == other_class].shape[0]
        return recall

    def get_weight(self, output, clause):
        return self.weight_banks[output].get_weights()[clause]

    def set_weight(self, output, weight):
        self.weight_banks[output].get_weights()[output] = weight
