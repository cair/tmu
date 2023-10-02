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
import typing
from collections import defaultdict

from tmu.models.base import MultiClauseBankMixin, MultiWeightBankMixin, TMBaseModel
from tmu.util.encoded_data_cache import DataEncoderCache
from tmu.util.statistics import MetricRecorder
from tmu.weight_bank import WeightBank
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class TMClassifier(TMBaseModel, MultiClauseBankMixin, MultiWeightBankMixin):
    def __init__(
            self,
            number_of_clauses: int,
            T: int,
            s: float,
            confidence_driven_updating: bool = False,
            type_i_ii_ratio: float = 1.0,
            type_i_feedback: bool = True,
            type_ii_feedback: bool = True,
            type_iii_feedback: bool = False,
            d: float = 200.0,
            platform: str = 'CPU',
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
            batch_size=100,
            incremental=True,
            type_ia_ii_feedback_ratio=0,
            absorbing=-1,
            absorbing_include=None,
            absorbing_exclude=None,
            literal_sampling=1.0,
            feedback_rate_excluded_literals=1,
            literal_insertion_state=-1,
            seed=None
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            confidence_driven_updating=confidence_driven_updating,
            type_i_ii_ratio=type_i_ii_ratio,
            type_i_feedback=type_i_feedback,
            type_ii_feedback=type_ii_feedback,
            type_iii_feedback=type_iii_feedback,
            d=d,
            platform=platform,
            patch_dim=patch_dim,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost_true_positive_feedback,
            reuse_random_feedback=reuse_random_feedback,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            weighted_clauses=weighted_clauses,
            clause_drop_p=clause_drop_p,
            literal_drop_p=literal_drop_p,
            batch_size=batch_size,
            incremental=incremental,
            type_ia_ii_feedback_ratio=type_ia_ii_feedback_ratio,
            absorbing=absorbing,
            absorbing_include=absorbing_include,
            absorbing_exclude=absorbing_exclude,
            literal_sampling=literal_sampling,
            feedback_rate_excluded_literals=feedback_rate_excluded_literals,
            literal_insertion_state=literal_insertion_state,
            seed=seed
        )
        MultiClauseBankMixin.__init__(self, seed=seed)
        MultiWeightBankMixin.__init__(self, seed=seed)

        # These data structures cache the encoded data for the training and test sets. It also makes a fast-check if
        # training data has changed, and only re-encodes if it has.
        self.test_encoder_cache = DataEncoderCache(seed=self.seed)
        self.train_encoder_cache = DataEncoderCache(seed=self.seed)

        self.metrics = MetricRecorder()

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_banks.set_clause_init(clause_bank_type, clause_bank_args)
        self.clause_banks.populate(list(range(self.number_of_classes)))

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.weight_banks.set_clause_init(
            WeightBank,
            dict(
                weights=np.concatenate((
                    np.ones(self.number_of_clauses // 2, dtype=np.int32),
                    -1 * np.ones(self.number_of_clauses // 2, dtype=np.int32)
                ))
            )
        )
        self.weight_banks.populate(list(range(self.number_of_classes)))

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        self.positive_clauses = np.concatenate((np.ones(self.number_of_clauses // 2, dtype=np.int32),
                                                np.zeros(self.number_of_clauses // 2, dtype=np.int32)))

        self.negative_clauses = np.concatenate((np.zeros(self.number_of_clauses // 2, dtype=np.int32),
                                                np.ones(self.number_of_clauses // 2, dtype=np.int32)))

    def init_num_classes(self, X: np.ndarray, Y: np.ndarray):
        return int(np.max(Y) + 1)

    def mechanism_feedback(
            self,
            is_target: bool,
            target: int,
            clause_outputs: np.ndarray,
            update_p: np.ndarray,
            clause_active: np.ndarray,
            literal_active: np.ndarray,
            encoded_X_train: np.ndarray,
            sample_idx: int,
    ):
        clause_a = self.positive_clauses if is_target else self.negative_clauses
        clause_b = self.negative_clauses if is_target else self.positive_clauses

        if self.weighted_clauses:
            if is_target:
                self.weight_banks[target].increment(
                    clause_output=clause_outputs,
                    update_p=update_p,
                    clause_active=clause_active[target],
                    positive_weights=False
                )
            else:
                self.weight_banks[target].decrement(
                    clause_output=clause_outputs,
                    update_p=update_p,
                    clause_active=clause_active[target],
                    negative_weights=False
                )

        if self.type_i_feedback:
            self.clause_banks[target].type_i_feedback(
                update_p=update_p * self.type_i_p,
                clause_active=clause_active[target] * clause_a,
                literal_active=literal_active,
                encoded_X=encoded_X_train,
                e=sample_idx
            )

        if self.type_ii_feedback:
            self.clause_banks[target].type_ii_feedback(
                update_p=update_p * self.type_ii_p,
                clause_active=clause_active[target] * clause_b,
                literal_active=literal_active,
                encoded_X=encoded_X_train,
                e=sample_idx
            )

        if self.type_iii_feedback:
            self.clause_banks[target].type_iii_feedback(
                update_p=update_p,
                clause_active=clause_active[target] * clause_a,
                literal_active=literal_active,
                encoded_X=encoded_X_train,
                e=sample_idx,
                target=1
            )

            self.clause_banks[target].type_iii_feedback(
                update_p=update_p,
                clause_active=clause_active[target] * clause_b,
                literal_active=literal_active,
                encoded_X=encoded_X_train,
                e=sample_idx,
                target=0
            )

    def mechanism_clause_sum(
            self,
            target: int,
            clause_active: np.ndarray,
            literal_active: np.ndarray,
            encoded_X_train: np.ndarray,
            sample_idx: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:

        clause_outputs: np.ndarray = self.clause_banks[target].calculate_clause_outputs_update(
            literal_active=literal_active,
            encoded_X=encoded_X_train,
            e=sample_idx
        )

        class_sum: np.ndarray = np.dot(
            clause_active[target] * self.weight_banks[target].get_weights(),
            clause_outputs
        ).astype(np.int32)

        class_sum: np.ndarray = np.clip(class_sum, -self.T, self.T)

        return class_sum, clause_outputs

    def mechanism_compute_update_probabilities(
            self,
            is_target: bool,
            class_sum: np.ndarray
    ) -> float:
        # Confidence-driven updating method
        if self.confidence_driven_updating:
            return (self.T - abs(class_sum)) / self.T

        # Compute based on whether the class is the target or not
        if is_target:
            return (self.T - class_sum) / (2 * self.T)
        return (self.T + class_sum) / (2 * self.T)

    def mechanism_literal_active(self) -> np.ndarray:
        # Literals are dropped based on literal drop probability
        literal_active = np.zeros(self.clause_banks[0].number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = self.rng.rand(self.clause_banks[0].number_of_literals) >= self.literal_drop_p

        for k in range(self.clause_banks[0].number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] |= (1 << chunk_pos)

        if not self.feature_negation:
            for k in range(self.clause_banks[0].number_of_literals // 2, self.clause_banks[0].number_of_literals):
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] &= (~(1 << chunk_pos))
        literal_active = literal_active.astype(np.uint32)

        return literal_active

    def mechanism_clause_active(self) -> np.ndarray:
        # Drops clauses randomly based on clause drop probability
        return (self.rng.rand(len(self.weight_banks), self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

    def _fit_sample_target(
            self,
            class_sum: int,
            clause_outputs: np.ndarray,
            is_target_class: bool,
            class_value: int,
            sample_idx: int,
            clause_active: np.ndarray,
            literal_active: np.ndarray,
            encoded_X_train: np.ndarray
    ) -> float:
        """
        Handle operations for a given class type (target or not target).

        :param is_target_class: Boolean indicating if the class is a target.
        :param class_value: Value of the class.
        :param sample_idx: Index of the current sample.
        :param clause_active: Clause active status.
        :param literal_active: Literal active status.
        :param encoded_X_train: Encoded training data.
        :return: Computed update probability for the class.
        """

        update_p: float = self.mechanism_compute_update_probabilities(
            is_target=is_target_class,
            class_sum=class_sum
        )

        self.mechanism_feedback(
            is_target=is_target_class,
            target=class_value,
            clause_outputs=clause_outputs,
            update_p=update_p,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train,
            sample_idx=sample_idx
        )

        return update_p

    def _fit_sample(
            self,
            target: int,
            not_target: int,
            sample_idx: int,
            clause_active: np.ndarray,
            literal_active: np.ndarray,
            encoded_X_train
    ) -> dict:

        class_sum, clause_outputs = self.mechanism_clause_sum(
            target=target,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train,
            sample_idx=sample_idx
        )

        update_p_target: float = self._fit_sample_target(
            class_sum=class_sum,
            clause_outputs=clause_outputs,
            is_target_class=True,
            class_value=target,
            sample_idx=sample_idx,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train
        )

        # for incremental, and when we only have 1 sample, there is no other targets
        if self.weight_banks.n_classes == 1:
            return dict(
                update_p_target=update_p_target,
                update_p_not_target=None
            )

        class_sum_not, clause_outputs_not = self.mechanism_clause_sum(
            target=not_target,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train,
            sample_idx=sample_idx
        )

        update_p_not_target: float = self._fit_sample_target(
            class_sum=class_sum_not,
            clause_outputs=clause_outputs_not,
            is_target_class=False,
            class_value=not_target,
            sample_idx=sample_idx,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train
        )

        return dict(
            update_p_not_target=update_p_not_target,
            update_p_target=update_p_target
        )

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            shuffle: bool = True,
            metrics: typing.Optional[list] = None,
            *args,
            **kwargs
    ):
        metrics = metrics or []
        assert len(X) == len(Y), "X and Y must have the same length"
        assert len(X.shape) >= 2, "X must be a 2D array"
        assert len(Y.shape) == 1, "Y must be a 1D array"

        self.init(X, Y)
        self.metrics.clear()

        Ym = Y.astype(np.uint32)

        encoded_X_train: np.ndarray = self.train_encoder_cache.get_encoded_data(
            data=X,
            encoder_func=lambda x: self.clause_banks[0].prepare_X(x)
        )

        clause_active: np.ndarray = self.mechanism_clause_active()
        literal_active: np.ndarray = self.mechanism_literal_active()

        sample_indices: np.ndarray = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(sample_indices)

        for sample_idx in sample_indices:
            target: int = Ym[sample_idx]
            not_target: int = self.weight_banks.sample(exclude=[target])

            history: dict = self._fit_sample(
                target=target,
                not_target=not_target,
                sample_idx=sample_idx,
                clause_active=clause_active,
                literal_active=literal_active,
                encoded_X_train=encoded_X_train
            )

            if "update_p" in metrics:
                self.metrics.add_scalar(group="update_p", key=target, value=history["update_p_target"])
                self.metrics.add_scalar(group="update_p", key=target, value=history["update_p_not_target"])

        return self.metrics.export(
            mean=True,
            std=True
        )

    def predict(
            self,
            X: np.ndarray,
            clip_class_sum: bool = False,
            return_class_sums: bool = False,
            **kwargs
    ):

        encoded_X_test = self.test_encoder_cache.get_encoded_data(
            data=X,
            encoder_func=lambda x: self.clause_banks[0].prepare_X(x)
        )

        class_sums = np.array([
            self.predict_compute_class_sums(
                encoded_X_test=encoded_X_test,
                ith_sample=i,
                clip_class_sum=clip_class_sum
            ) for i in range(X.shape[0])
        ])

        max_classes = np.argmax(class_sums, axis=1)

        if return_class_sums:
            return max_classes, class_sums
        else:
            return max_classes

    def predict_compute_class_sums(
            self,
            encoded_X_test: np.array,
            ith_sample: int,
            clip_class_sum: bool
    ) -> typing.List[int]:
        """The following function evaluates the resulting class sum votes.

        Args:
            ith_sample (int): The index of the sample
            clip_class_sum (bool): Wether to clip class sums

        Returns:
            list[int]: list of all class sums
        """
        class_sums = []
        for ith_class in self.weight_banks.classes():

            class_sum = np.dot(
                self.weight_banks[ith_class].get_weights(),
                self.clause_banks[ith_class].calculate_clause_outputs_predict(encoded_X_test, ith_sample)
            ).astype(np.int32)

            if clip_class_sum:
                class_sum = np.clip(class_sum, -self.T, self.T)
            class_sums.append(class_sum)
        return class_sums

    def transform(self, X):
        encoded_X = self.clause_banks[0].prepare_X(X)
        transformed_X = np.empty((X.shape[0], len(self.weight_banks), self.number_of_clauses), dtype=np.uint32)
        for e in range(X.shape[0]):
            for i in self.weight_banks.classes():
                transformed_X[e, i, :] = self.clause_banks[i].calculate_clause_outputs_predict(encoded_X, e)
        return transformed_X.reshape((X.shape[0], len(self.weight_banks) * self.number_of_clauses))

    def transform_patchwise(self, X):
        encoded_X = self.clause_banks[0].prepare_X(X)

        transformed_X = np.empty(
            (X.shape[0], len(self.weight_banks), self.number_of_clauses * self.clause_banks[0].number_of_patches),
            dtype=np.uint32)
        for e in range(X.shape[0]):
            for i in self.weight_banks.classes():
                transformed_X[e, i, :] = self.clause_banks[i].calculate_clause_outputs_patchwise(encoded_X, e)
        return transformed_X.reshape(
            (X.shape[0], len(self.weight_banks) * self.number_of_clauses, self.clause_banks[0].number_of_patches))

    def literal_clause_frequency(self):
        clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
        literal_frequency = np.zeros(self.clause_banks[0].number_of_literals, dtype=np.uint32)
        for i in self.weight_banks.classes():
            literal_frequency += self.clause_banks[i].calculate_literal_clause_frequency(clause_active)
        return literal_frequency

    def literal_importance(self, the_class, negated_features=False, negative_polarity=False):
        literal_frequency = np.zeros(self.clause_banks[0].number_of_literals, dtype=np.uint32)
        if negated_features:
            if negative_polarity:
                literal_frequency[self.clause_banks[the_class].number_of_literals // 2:] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.negative_clauses)[self.clause_banks[the_class].number_of_literals // 2:]
            else:
                literal_frequency[self.clause_banks[the_class].number_of_literals // 2:] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.positive_clauses)[self.clause_banks[the_class].number_of_literals // 2:]
        else:
            if negative_polarity:
                literal_frequency[:self.clause_banks[the_class].number_of_literals // 2] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.negative_clauses)[:self.clause_banks[the_class].number_of_literals // 2]
            else:
                literal_frequency[:self.clause_banks[the_class].number_of_literals // 2] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.positive_clauses)[:self.clause_banks[the_class].number_of_literals // 2]

        return literal_frequency

    def clause_precision(self, the_class, polarity, X, Y):
        clause_outputs = self.transform(X).reshape(
            X.shape[0],
            len(self.weight_banks),
            2,
            self.number_of_clauses // 2
        )[:, the_class, polarity, :]

        if polarity == 0:
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
        else:
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
        return np.where(
            true_positive_clause_outputs + false_positive_clause_outputs == 0,
            0,
            true_positive_clause_outputs / (true_positive_clause_outputs + false_positive_clause_outputs)
        )

    def clause_recall(self, the_class, polarity, X, Y):
        clause_outputs = self.transform(X).reshape(
            X.shape[0],
            len(self.weight_banks),
            2,
            self.number_of_clauses // 2
        )[:, the_class, polarity, :]

        if polarity == 0:
            return clause_outputs[Y == the_class].sum(axis=0) / Y[Y == the_class].shape[0]
        else:
            return clause_outputs[Y != the_class].sum(axis=0) / Y[Y != the_class].shape[0]

    def get_weight(self, the_class, polarity, clause):
        polarized_clause = self._get_polarized_clause_index(clause, polarity)
        return self.weight_banks[the_class].get_weights()[polarized_clause]

    def set_weight(self, the_class, polarity, clause, weight):
        polarized_clause = self._get_polarized_clause_index(clause, polarity)
        self.weight_banks[the_class].get_weights()[polarized_clause] = weight

    def get_ta_action(self, clause, ta, the_class=0, polarity=0):
        return self.clause_banks[the_class].get_ta_action(
            self._get_polarized_clause_index(clause, polarity), ta)

    def get_ta_state(self, clause, ta, the_class=0, polarity=0):
        return self.clause_banks[the_class].get_ta_state(
            self._get_polarized_clause_index(clause, polarity), ta)

    def set_ta_state(self, clause, ta, state, the_class=0, polarity=0):
        self.clause_banks[the_class].set_ta_state(
            self._get_polarized_clause_index(clause, polarity), ta, state)

    def number_of_include_actions(self, the_class, clause):
        return self.clause_banks[the_class].number_of_include_actions(clause)

    def number_of_exclude_actions(self, the_class, clause):
        return self.clause_banks[the_class].number_of_exclude_actions(clause)

    def number_of_unallocated_literals(self, the_class, clause):
        return self.clause_banks[the_class].number_of_unallocated_literals(clause)

    def _get_polarized_clause_index(self, clause, polarity):
        return clause if polarity == 0 else self.number_of_clauses // 2 + clause

    def number_of_absorbed_include_actions(self, the_class, clause):
        return self.clause_banks[the_class].number_of_absorbed_include_actions(clause)
