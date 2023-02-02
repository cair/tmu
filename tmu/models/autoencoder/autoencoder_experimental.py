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
from tmu.weight_bank import WeightBank
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class TMAutoEncoder(TMBasis):
    def __init__(self, number_of_clauses, T, s, output_active, accumulation=1, type_i_ii_ratio=1.0,
                 type_iii_feedback=False, focused_negative_sampling=False, output_balancing=False, d=200.0,
                 platform='CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1,
                 max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8,
                 weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0, incremental=True):
        self.output_active = output_active
        self.accumulation = accumulation
        super().__init__(number_of_clauses, T, s, type_i_ii_ratio=type_i_ii_ratio, type_iii_feedback=type_iii_feedback,
                         focused_negative_sampling=focused_negative_sampling, output_balancing=output_balancing, d=d,
                         platform=platform, patch_dim=patch_dim, feature_negation=feature_negation,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta,
                         number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p, incremental=incremental)

    def initialize(self, X):
        self.number_of_classes = self.output_active.shape[0]
        if self.platform == 'CPU':
            self.clause_bank = ClauseWeightBank(X, self.number_of_classes, self.number_of_clauses, self.number_of_state_bits_ta,
                                          self.number_of_state_bits_ind, self.patch_dim, batch_size=self.number_of_classes, incremental=self.incremental)
        else:
            raise RuntimeError(f"Unknown platform of type: {self.platform}")

        self.weight_banks = []
        for i in range(self.number_of_classes):
            self.weight_banks.append(
                WeightBank(np.random.choice([-1, 1], size=self.number_of_clauses).astype(np.int32)))

        if self.max_included_literals == None:
            self.max_included_literals = self.clause_bank.number_of_literals

        self.update_p = np.empty(self.number_of_classes, dtype=np.float32)
        self.class_sum = np.empty(self.number_of_classes, dtype=np.int32)

    def activate_clauses(self):
        # Drops clauses randomly based on clause drop probability
        clause_active = (np.random.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        return clause_active 

    def activate_literals(self):
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

        return literal_active

    def fit(self, X, number_of_examples=2000, shuffle=True):
        if self.initialized == False:
            self.initialize(X)
            self.initialized = True

        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()

        class_index = np.arange(self.number_of_classes, dtype=np.uint32)
        for e in range(number_of_examples):
            average_absolute_weights = np.zeros(self.number_of_clauses, dtype=np.float32)
            for i in class_index:
                average_absolute_weights += np.absolute(self.clause_bank.get_weights()[i])
            average_absolute_weights /= self.number_of_classes
            update_clause = np.random.random(self.number_of_clauses) <= (
                    self.T - np.clip(average_absolute_weights, 0, self.T)) / self.T

            Xu, Yu = self.clause_bank.prepare_autoencoder_examples(X_csr, X_csc, self.output_active, self.accumulation)
            for i in range(self.number_of_classes):
                all_literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
                clause_outputs = self.clause_bank.calculate_clause_outputs(all_literal_active, Xu, i, 1)

                self.update_p[i] = np.dot(clause_active * self.clause_bank.get_weights()[i],
                                               clause_outputs).astype(np.int32)
                self.update_p[i] = np.clip(self.update_p[i], -self.T, self.T)

                if Yu[i]:
                    self.update_p[i] = 1.0 * (self.T - self.update_p[i]) / (2 * self.T)
                else:
                    self.update_p[i] = 1.0 * (self.T + self.update_p[i]) / (2 * self.T)

            self.clause_bank.type_i_and_ii_feedback(self.update_p, self.s, self.boost_true_positive_feedback,
                                         self.max_included_literals, clause_active,
                                         literal_active, Xu, 0, Yu, autoencoder=1, output_literal_index=self.output_active)
        return

    def predict(self, X):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        Y = np.ascontiguousarray(np.zeros((self.number_of_classes, X.shape[0]), dtype=np.uint32))

        all_literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)

        for e in range(X.shape[0]):
            encoded_X = self.clause_bank.prepare_X(X_csr[e, :].toarray())

            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(all_literal_active, encoded_X, 0, 0)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.clause_bank.get_weights()[i], clause_outputs).astype(np.int32)
                Y[i, e] = (class_sum >= 0)
        return Y

    def literal_importance(self, the_class, negated_features=False, negative_polarity=False):
        literal_frequency = np.zeros(self.clause_bank.number_of_literals, dtype=np.uint32)
        if negated_features:
            if negative_polarity:
                literal_frequency[
                self.clause_bank.number_of_literals // 2:] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() < 0)[self.clause_bank.number_of_literals // 2:]
            else:
                literal_frequency[
                self.clause_bank.number_of_literals // 2:] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() >= 0)[self.clause_bank.number_of_literals // 2:]
        else:
            if negative_polarity:
                literal_frequency[
                :self.clause_bank.number_of_literals // 2] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() < 0)[:self.clause_bank.number_of_literals // 2]
            else:
                literal_frequency[
                :self.clause_bank.number_of_literals // 2] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() >= 0)[:self.clause_bank.number_of_literals // 2]

        return literal_frequency

    def clause_precision(self, the_class, positive_polarity, X, number_of_examples=2000):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        true_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)
        false_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)

        weights = self.clause_bank.get_weights()[the_class]

        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()

        for e in range(number_of_examples):
            Xu, Yu = self.clause_bank.prepare_autoencoder_examples(X_csr, X_csc,
                                                                   np.array([self.output_active[the_class]],
                                                                            dtype=np.uint32), self.accumulation)
            (target, encoded_X) = Yu[0], Xu[0].reshape((1, -1))

            #clause_outputs = self.clause_bank.calculate_clause_outputs(literal_active, encoded_X, 0, 0)
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)

            if positive_polarity:
                if target == 1:
                    true_positive += (weights >= 0) * clause_outputs
                else:
                    false_positive += (weights >= 0) * clause_outputs
            else:
                if target == 0:
                    true_positive += (weights < 0) * clause_outputs
                else:
                    false_positive += (weights < 0) * clause_outputs

        return 1.0 * true_positive / (true_positive + false_positive)

    def clause_recall(self, the_class, positive_polarity, X, number_of_examples=2000):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        true_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)
        false_negative = np.zeros(self.number_of_clauses, dtype=np.uint32)

        weights = self.clause_bank.get_weights()[the_class]

        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()

        for e in range(number_of_examples):
            Xu, Yu = self.clause_bank.prepare_autoencoder_examples(X_csr, X_csc,
                                                                   np.array([self.output_active[the_class]],
                                                                            dtype=np.uint32), self.accumulation)
            (target, encoded_X) = Yu[0], Xu[0].reshape((1, -1))

            #clause_outputs = self.clause_bank.calculate_clause_outputs(literal_active, encoded_X, 0, 0)
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)

            if positive_polarity:
                if target == 1:
                    true_positive += (weights >= 0) * clause_outputs
                    false_negative += (weights >= 0) * (1 - clause_outputs)
            else:
                if target == 0:
                    true_positive += (weights < 0) * clause_outputs
                    false_negative += (weights < 0) * (1 - clause_outputs)

        return true_positive / (true_positive + false_negative)

    def get_weight(self, the_class, clause):
        return self.clause_bank.get_weights()[the_class, clause]

    def get_weights(self, the_class):
        return self.clause_bank.get_weights()[the_class]

    def set_weight(self, the_class, clause, weight):
       self.clause_bank.get_weights()[the_class, clause] = weight
