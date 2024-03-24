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

from tmu.models.base import SingleClauseBankMixin, MultiWeightBankMixin, TMBaseModel

from tmu.weight_bank import WeightBank
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix


class TMRelational(TMBaseModel, SingleClauseBankMixin, MultiWeightBankMixin):
    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            output_active_facts,
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
            seed=None
    ):
        self.output_active_facts = output_active_facts

        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            type_i_ii_ratio=type_i_ii_ratio,
            type_iii_feedback=type_iii_feedback,
            focused_negative_sampling=focused_negative_sampling,
            output_balancing=output_balancing,
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
            literal_drop_p=literal_drop_p,
            seed=seed
        )

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_bank = clause_bank_type(**clause_bank_args)

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.number_of_classes = X.shape[1]
        self.weight_banks.set_clause_init(WeightBank, dict(
            weights=self.rng.choice([-1, 1], size=self.number_of_clauses).astype(np.int32)
        ))
        self.weight_banks.populate(list(range(self.number_of_classes)))

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        self.output_active = np.empty(len(self.output_active_facts), dtype=np.uint32)
        for i in range(len(self.output_active_facts)):
            self.output_active[i] = self.fact_id[self.output_active_facts[i]]

    def update(self, target_output, target_value, encoded_X, clause_active, literal_active):
        all_literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
        clause_outputs = self.clause_bank.calculate_clause_outputs_update(all_literal_active, encoded_X, 0)

        class_sum = np.dot(clause_active * self.weight_banks[target_output].get_weights(), clause_outputs).astype(
            np.int32)
        class_sum = np.clip(class_sum, -self.T, self.T)

        type_iii_feedback_selection = self.rng.choice(2)

        if target_value == 1:
            update_p = (self.T - class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(
                update_p=update_p * self.type_i_p,
                clause_active=clause_active * (self.weight_banks[target_output].get_weights() >= 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )


            self.clause_bank.type_ii_feedback(
                update_p=update_p * self.type_ii_p,
                clause_active=clause_active * (self.weight_banks[target_output].get_weights() < 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.weight_banks[target_output].increment(
                clause_output=clause_outputs,
                update_p=update_p,
                clause_active=clause_active,
                positive_weights=True
            )


            if self.type_iii_feedback and type_iii_feedback_selection == 0:
                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (self.weight_banks[target_output].get_weights() >= 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=1
                )

                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (self.weight_banks[target_output].get_weights() < 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=0
                )
        else:
            update_p = (self.T + class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(
                update_p=update_p * self.type_i_p,
                clause_active=clause_active * (self.weight_banks[target_output].get_weights() < 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.clause_bank.type_ii_feedback(
                update_p=update_p * self.type_ii_p,
                clause_active=clause_active * (self.weight_banks[target_output].get_weights() >= 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.weight_banks[target_output].decrement(
                clause_output=clause_outputs,
                update_p=update_p,
                clause_active=clause_active,
                negative_weights=True
            )

            if self.type_iii_feedback and type_iii_feedback_selection == 1:
                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (self.weight_banks[target_output].get_weights() < 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=1
                )

                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (self.weight_banks[target_output].get_weights() >= 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=0
                )

    def activate_clauses(self):
        # Drops clauses randomly based on clause drop probability
        clause_active = (self.rng.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        return clause_active

    def activate_literals(self):
        # Literals are dropped based on literal drop probability
        literal_active = np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = self.rng.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
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

    def fit(self, X, number_of_examples=2000, shuffle=True, **kwargs):
        (X, X_active) = self.propositionalize(X)
        self.init(X=X, Y=None)

        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()

        output_index = np.arange(self.output_active.shape[0])
        self.rng.shuffle(output_index)
        for e in range(number_of_examples):
            Xu, Yu = self.clause_bank.prepare_autoencoder_examples(X_csr, X_csc, self.output_active, 1)
            for i in output_index:
                (target, encoded_X) = Yu[i], Xu[i].reshape((1, -1))

                ta_chunk = self.output_active[i] // 32
                chunk_pos = self.output_active[i] % 32
                copy_literal_active_ta_chunk = literal_active[ta_chunk]

                if self.feature_negation:
                    ta_chunk_negated = (self.output_active[i] + self.clause_bank.number_of_features) // 32
                    chunk_pos_negated = (self.output_active[i] + self.clause_bank.number_of_features) % 32
                    copy_literal_active_ta_chunk_negated = literal_active[ta_chunk_negated]
                    literal_active[ta_chunk_negated] &= ~(1 << chunk_pos_negated)

                literal_active[ta_chunk] &= ~(1 << chunk_pos)

                self.update(i, target, encoded_X, clause_active, literal_active)

                if self.feature_negation:
                    literal_active[ta_chunk_negated] = copy_literal_active_ta_chunk_negated
                literal_active[ta_chunk] = copy_literal_active_ta_chunk

    def predict(self, X, **kwargs):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        Y = np.ascontiguousarray(np.zeros((self.number_of_classes, X.shape[0]), dtype=np.uint32))

        for e in range(X.shape[0]):
            encoded_X = self.clause_bank.prepare_X(X_csr[e, :].toarray())

            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)[0]
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
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

        weights = self.weight_banks[the_class].get_weights()

        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()

        for e in range(number_of_examples):
            Xu, Yu = self.clause_bank.prepare_autoencoder_examples(X_csr, X_csc,
                                                                   np.array([self.output_active[the_class]],
                                                                            dtype=np.uint32), self.accumulation)
            (target, encoded_X) = Yu[0], Xu[0].reshape((1, -1))

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

        weights = self.weight_banks[the_class].get_weights()

        for e in range(number_of_examples):
            Xu, Yu = self.clause_bank.prepare_autoencoder_examples(X_csr, X_csc,
                                                                   np.array([self.output_active[the_class]],
                                                                            dtype=np.uint32), self.accumulation)
            (target, encoded_X) = Yu[0], Xu[0].reshape((1, -1))

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
        return self.weight_banks[the_class].get_weights()[clause]

    def get_weights(self, the_class):
        return self.weight_banks[the_class].get_weights()

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight

    def propositionalize(self, X):
        # Identify relations and symbols
        self.relation_id = {}
        self.id_relation = {}
        self.relation_arity = {}
        r_id = 0
        self.symbol_id = {}
        self.id_symbol = {}
        s_id = 0
        for facts in X:
            for (fact, value) in facts:
                relation = fact[0]
                if relation not in self.relation_id:
                    self.relation_id[relation] = r_id
                    self.id_relation[r_id] = relation
                    self.relation_arity[relation] = len(fact[1:])
                    r_id += 1
                for symbol in fact[1:]:
                    if symbol not in self.symbol_id:
                        self.symbol_id[symbol] = s_id
                        self.id_symbol[s_id] = symbol
                        s_id += 1

        self.fact_id = {}
        self.id_fact = {}
        self.number_of_features = 0
        for relation in self.relation_id.keys():
            number_of_relation_facts = len(self.symbol_id) ** (self.relation_arity[relation])
            for relation_fact_id in range(number_of_relation_facts):
                fact = [relation]
                symbol_id_remainder = relation_fact_id
                for argument_id in range(self.relation_arity[relation]):
                    fact.append(self.id_symbol[symbol_id_remainder % len(self.symbol_id)])
                    symbol_id_remainder = symbol_id_remainder // len(self.symbol_id)
                fact = tuple(fact)
                self.fact_id[fact] = self.number_of_features + relation_fact_id
                self.id_fact[self.number_of_features + relation_fact_id] = fact
            self.number_of_features += number_of_relation_facts

        X_propositional = lil_matrix((len(X), self.number_of_features), dtype=np.uint32)
        X_active = lil_matrix((len(X), self.number_of_features), dtype=np.uint32)
        example_id = 0
        for facts in X:
            for (fact, value) in facts:
                X_propositional[example_id, self.fact_id[fact]] = value
                X_active[example_id, self.fact_id[fact]] = 1
            example_id += 1
        X_propositional = csr_matrix(X_propositional)
        X_active = csr_matrix(X_active)

        return X_propositional, X_active
