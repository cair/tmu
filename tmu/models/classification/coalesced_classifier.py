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

from tmu.models.base import MultiWeightBankMixin, SingleClauseBankMixin
from tmu.weight_bank import WeightBank
from tmu.models.classification.base_classification import TMBaseClassifier
import numpy as np


class TMCoalescedClassifier(TMBaseClassifier, SingleClauseBankMixin, MultiWeightBankMixin):
    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            type_i_ii_ratio=1.0,
            type_iii_feedback=False,
            focused_negative_sampling=False,
            output_balancing=False,
            d=200.0,
            platform='CPU',
            patch_dim=None,
            feature_negation=True,
            boost_true_positive_feedback=1,
            max_positive_clauses=None,
            max_included_literals=None,
            number_of_state_bits_ta=8,
            number_of_state_bits_ind=8,
            weighted_clauses=False,
            clause_drop_p=0.0,
            literal_drop_p=0.0
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            type_i_ii_ratio=type_i_ii_ratio,
            type_iii_feedback=type_iii_feedback,
            focused_negative_sampling=focused_negative_sampling, output_balancing=output_balancing,
            d=d,
            platform=platform,
            patch_dim=patch_dim,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost_true_positive_feedback,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            weighted_clauses=weighted_clauses,
            clause_drop_p=clause_drop_p,
            literal_drop_p=literal_drop_p
        )
        SingleClauseBankMixin.__init__(self)
        MultiWeightBankMixin.__init__(self)

        self.max_positive_clauses = max_positive_clauses

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_bank = clause_bank_type(**clause_bank_args)

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.number_of_classes = int(np.max(Y) + 1)
        self.weight_banks.set_clause_init(WeightBank, dict(
            weights=WeightBank(np.random.choice([-1, 1], size=self.number_of_clauses).astype(np.int32))
        ))
        self.weight_banks.populate(list(range(self.number_of_classes)))

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        if self.max_included_literals is None:
            self.max_included_literals = self.clause_bank.number_of_literals

        if self.max_positive_clauses is None:
            self.max_positive_clauses = self.number_of_clauses

    def update(self, target, e):
        clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.literal_active, self.encoded_X_train, e)

        class_sum = np.dot(self.clause_active * self.weight_banks[target].get_weights(), clause_outputs).astype(
            np.int32)
        class_sum = np.clip(class_sum, -self.T, self.T)
        update_p = (self.T - class_sum) / (2 * self.T)

        type_iii_feedback_selection = np.random.choice(2)

        self.clause_bank.type_i_feedback(update_p * self.type_i_p, self.s, self.boost_true_positive_feedback,
                                         self.max_included_literals,
                                         self.clause_active * (self.weight_banks[target].get_weights() >= 0),
                                         self.literal_active, self.encoded_X_train, e)
        self.clause_bank.type_ii_feedback(update_p * self.type_ii_p,
                                          self.clause_active * (self.weight_banks[target].get_weights() < 0),
                                          self.literal_active, self.encoded_X_train, e)

        if (self.weight_banks[target].get_weights() >= 0).sum() < self.max_positive_clauses:
            self.weight_banks[target].increment(clause_outputs, update_p, self.clause_active, True)

        if self.type_iii_feedback and type_iii_feedback_selection == 0:
            self.clause_bank.type_iii_feedback(update_p, self.d,
                                               self.clause_active * (self.weight_banks[target].get_weights() >= 0),
                                               self.literal_active, self.encoded_X_train, e, 1)
            self.clause_bank.type_iii_feedback(update_p, self.d,
                                               self.clause_active * (self.weight_banks[target].get_weights() < 0),
                                               self.literal_active, self.encoded_X_train, e, 0)

        for i in range(self.number_of_classes):
            if i == target:
                self.update_ps[i] = 0.0
            else:
                self.update_ps[i] = np.dot(self.clause_active * self.weight_banks[i].get_weights(),
                                           clause_outputs).astype(np.int32)
                self.update_ps[i] = np.clip(self.update_ps[i], -self.T, self.T)
                self.update_ps[i] = 1.0 * (self.T + self.update_ps[i]) / (2 * self.T)

        if self.update_ps.sum() == 0:
            return

        if self.focused_negative_sampling:
            not_target = np.random.choice(self.number_of_classes, p=self.update_ps / self.update_ps.sum())
            update_p = self.update_ps[not_target]
        else:
            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)
            update_p = self.update_ps[not_target]

        self.clause_bank.type_i_feedback(update_p * self.type_i_p, self.s, self.boost_true_positive_feedback,
                                         self.max_included_literals,
                                         self.clause_active * (self.weight_banks[not_target].get_weights() < 0),
                                         self.literal_active, self.encoded_X_train, e)
        self.clause_bank.type_ii_feedback(update_p * self.type_ii_p,
                                          self.clause_active * (self.weight_banks[not_target].get_weights() >= 0),
                                          self.literal_active, self.encoded_X_train, e)
        if self.type_iii_feedback and type_iii_feedback_selection == 1:
            self.clause_bank.type_iii_feedback(update_p, self.d,
                                               self.clause_active * (self.weight_banks[not_target].get_weights() < 0),
                                               self.literal_active, self.encoded_X_train, e, 1)
            self.clause_bank.type_iii_feedback(update_p, self.d,
                                               self.clause_active * (self.weight_banks[not_target].get_weights() >= 0),
                                               self.literal_active, self.encoded_X_train, e, 0)

        self.weight_banks[not_target].decrement(clause_outputs, update_p, self.clause_active, True)

    def fit(self, X, Y, shuffle=True, **kwargs):
        self.init(X, Y)

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

        self.update_ps = np.empty(self.number_of_classes)

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(shuffled_index)

        class_observed = np.zeros(self.number_of_classes, dtype=np.uint32)
        example_indexes = np.zeros(self.number_of_classes, dtype=np.uint32)
        example_counter = 0
        for e in shuffled_index:
            if self.output_balancing:
                if class_observed[Ym[e]] == 0:
                    example_indexes[Ym[e]] = e
                    class_observed[Ym[e]] = 1
                    example_counter += 1
            else:
                example_indexes[example_counter] = e
                example_counter += 1

            if example_counter == self.number_of_classes:
                example_counter = 0

                for i in range(self.number_of_classes):
                    class_observed[i] = 0
                    batch_example = example_indexes[i]
                    self.update(Ym[batch_example], batch_example)
        return

    def predict(self, X, **kwargs):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

        for e in range(X.shape[0]):
            max_class_sum = -self.T
            max_class = 0
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
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
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                Y[e, i] = (class_sum >= 0)
        return Y

    def clause_precision(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()
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
        weights = self.weight_banks[the_class].get_weights()

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
        return self.weight_banks[the_class].get_weights()[clause]

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight

    def number_of_include_actions(self, clause):
        return self.clause_bank.number_of_include_actions(clause)
