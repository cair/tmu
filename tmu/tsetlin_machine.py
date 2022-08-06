# Copyright (c) 2022 Ole-Christoffer Granmo

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

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import sys
import numpy as np
from tmu.clause_bank import ClauseBank
from tmu.weight_bank import WeightBank
from scipy.sparse import csr_matrix, csc_matrix
from sys import maxsize

from time import time

class TMBasis():
	def __init__(self, number_of_clauses, T, s, confidence_driven_updating=False, type_i_ii_ratio = 1.0, type_iii_feedback=False, focused_negative_sampling=False, output_balancing=False, d=200.0, platform='CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits_ta = number_of_state_bits_ta
		self.number_of_state_bits_ind = number_of_state_bits_ind
		self.T = int(T)
		self.s = s

		self.confidence_driven_updating = confidence_driven_updating

		if type_i_ii_ratio >= 1.0:
			self.type_i_p = 1.0
			self.type_ii_p = 1.0/type_i_ii_ratio
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

		self.X_train = np.zeros(0, dtype=np.uint32)
		self.X_test = np.zeros(0, dtype=np.uint32)

		self.initialized = False

	def clause_co_occurrence(self, X, percentage=False):
		clause_outputs = csr_matrix(self.transform(X))
		if percentage:
			return clause_outputs.transpose().dot(clause_outputs).multiply(1.0/clause_outputs.sum(axis=0))
		else:
			return clause_outputs.transpose().dot(clause_outputs)

	def transform(self, X):
		encoded_X = self.clause_bank.prepare_X(X)
		transformed_X = np.empty((X.shape[0], self.number_of_clauses), dtype=np.uint32)
		for e in range(X.shape[0]):
			transformed_X[e,:] = self.clause_bank.calculate_clause_outputs_predict(encoded_X, e)
		return transformed_X

	def transform_patchwise(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.clause_bank.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		transformed_X = np.empty((X.shape[0], self.number_of_clauses*self.number_of_patches), dtype=np.uint32)
		for e in range(X.shape[0]):
			transformed_X[e,:] = self.clause_bank.calculate_clause_outputs_patchwise(encoded_X, e)
		return transformed_X.reshape((X.shape[0], self.number_of_clauses, self.number_of_patches))

	def literal_clause_frequency(self):
		clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
		return self.clause_bank.calculate_literal_clause_frequency(clause_active)
	
	def get_ta_action(self, clause, ta):
		return self.clause_bank.get_ta_action(clause, ta)

	def get_ta_state(self, clause, ta):
		return self.clause_bank.get_ta_state(clause, ta)

	def set_ta_state(self, clause, ta, state):
		return self.clause_bank.set_ta_state(clause, ta, state)

class TMClassifier(TMBasis):
	def __init__(self, number_of_clauses, T, s, confidence_driven_updating=False, type_i_ii_ratio = 1.0, type_iii_feedback=False, d=200.0, platform='CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, confidence_driven_updating=confidence_driven_updating, type_i_ii_ratio = type_i_ii_ratio, type_iii_feedback=type_iii_feedback, d=d, platform=platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta, number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		self.number_of_classes = int(np.max(Y) + 1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.concatenate((np.ones(self.number_of_clauses//2, dtype=np.int32), -1*np.ones(self.number_of_clauses//2, dtype=np.int32)))))
		
		self.clause_banks = []
		if self.platform == 'CPU':
			for i in range(self.number_of_classes):
				self.clause_banks.append(ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim))
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			for i in range(self.number_of_classes):
				self.clause_banks.append(ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim))
		else:
			print("Unknown Platform")
			sys.exit(-1)

		if self.max_included_literals == None:
			self.max_included_literals = self.clause_banks[0].number_of_literals

		self.positive_clauses = np.concatenate((np.ones(self.number_of_clauses//2, dtype=np.int32), np.zeros(self.number_of_clauses//2, dtype=np.int32)))
		self.negative_clauses = np.concatenate((np.zeros(self.number_of_clauses//2, dtype=np.int32), np.ones(self.number_of_clauses//2, dtype=np.int32)))

	def fit(self, X, Y, shuffle=True):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		if not np.array_equal(self.X_train, X):
			self.encoded_X_train = self.clause_banks[0].prepare_X(X)
			self.X_train = X.copy()

		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		clause_active = []
		for i in range(self.number_of_classes):
		 	# Clauses are dropped based on their weights
			class_clause_active = np.ascontiguousarray(np.ones(self.number_of_clauses, dtype=np.int32))
			clause_score = np.abs(self.weight_banks[i].get_weights())
			deactivate = np.random.choice(np.arange(self.number_of_clauses), size=int(self.number_of_clauses*self.clause_drop_p), p = clause_score / clause_score.sum())
			for d in range(deactivate.shape[0]):
				class_clause_active[deactivate[d]] = 0
			clause_active.append(class_clause_active)

		# Literals are dropped based on their frequency
		literal_active = (np.zeros(self.clause_banks[0].number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		literal_clause_frequency = self.literal_clause_frequency()
		if literal_clause_frequency.sum() > 0:
			deactivate = np.random.choice(np.arange(self.clause_banks[0].number_of_literals), size=int(self.clause_banks[0].number_of_literals*self.literal_drop_p), p = literal_clause_frequency / literal_clause_frequency.sum())
		else:
			deactivate = np.random.choice(np.arange(self.clause_banks[0].number_of_literals), size=int(self.clause_banks[0].number_of_literals*self.literal_drop_p))
		for d in range(deactivate.shape[0]):
			ta_chunk = deactivate[d] // 32
			chunk_pos = deactivate[d] % 32
			literal_active[ta_chunk] &= (~(1 << chunk_pos))

		if not self.feature_negation:
			for k in range(self.clause_banks[0].number_of_literals//2, self.clause_banks[0].number_of_literals):
				ta_chunk = k // 32
				chunk_pos = k % 32
				literal_active[ta_chunk] &= (~(1 << chunk_pos))
		literal_active = literal_active.astype(np.uint32)

		shuffled_index = np.arange(X.shape[0])
		if shuffle:
			np.random.shuffle(shuffled_index)

		for e in shuffled_index:
			target = Ym[e]
			
			clause_outputs = self.clause_banks[target].calculate_clause_outputs_update(literal_active, self.encoded_X_train, e)
			class_sum = np.dot(clause_active[target] * self.weight_banks[target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)

			if self.confidence_driven_updating:
				update_p = 1.0*(self.T - np.absolute(class_sum))/self.T
			else:
				update_p = (self.T - class_sum)/(2*self.T)

			if self.weighted_clauses:
				self.weight_banks[target].increment(clause_outputs, update_p, clause_active[target], False)
			self.clause_banks[target].type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, clause_active[target]*self.positive_clauses, literal_active, self.encoded_X_train, e)
			self.clause_banks[target].type_ii_feedback(update_p*self.type_ii_p, clause_active[target]*self.negative_clauses, literal_active, self.encoded_X_train, e)
			if self.type_iii_feedback:
				self.clause_banks[target].type_iii_feedback(update_p, self.d, clause_active[target]*self.positive_clauses, literal_active, self.encoded_X_train, e, 1)
				self.clause_banks[target].type_iii_feedback(update_p, self.d, clause_active[target]*self.negative_clauses, literal_active, self.encoded_X_train, e, 0)

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			clause_outputs = self.clause_banks[not_target].calculate_clause_outputs_update(literal_active, self.encoded_X_train, e)
			class_sum = np.dot(clause_active[not_target] * self.weight_banks[not_target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)

			if self.confidence_driven_updating:
				update_p = 1.0*(self.T - np.absolute(class_sum))/self.T
			else:
				update_p = (self.T + class_sum)/(2*self.T)
		
			if self.weighted_clauses:
				self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active[not_target], False)			
			self.clause_banks[not_target].type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, clause_active[not_target]*self.negative_clauses, literal_active, self.encoded_X_train, e)
			self.clause_banks[not_target].type_ii_feedback(update_p*self.type_ii_p, clause_active[not_target]*self.positive_clauses, literal_active, self.encoded_X_train, e)
			if self.type_iii_feedback:
				self.clause_banks[not_target].type_iii_feedback(update_p, self.d, clause_active[not_target]*self.negative_clauses, literal_active, self.encoded_X_train, e, 1)
				self.clause_banks[not_target].type_iii_feedback(update_p, self.d, clause_active[not_target]*self.positive_clauses, literal_active, self.encoded_X_train, e, 0)
		return

	def predict(self, X):
		if not np.array_equal(self.X_test, X):
			self.encoded_X_test = self.clause_banks[0].prepare_X(X)
			self.X_test = X.copy()

		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
		for e in range(X.shape[0]):
			max_class_sum = -self.T
			max_class = 0
			for i in range(self.number_of_classes):
				class_sum = np.dot(self.weight_banks[i].get_weights(), self.clause_banks[i].calculate_clause_outputs_predict(self.encoded_X_test, e)).astype(np.int32)
				class_sum = np.clip(class_sum, -self.T, self.T)
				if class_sum > max_class_sum:
					max_class_sum = class_sum
					max_class = i
			Y[e] = max_class
		return Y

	def transform(self, X):
		encoded_X = self.clause_banks[0].prepare_X(X)
		transformed_X = np.empty((X.shape[0], self.number_of_classes, self.number_of_clauses), dtype=np.uint32)
		for e in range(X.shape[0]):
			for i in range(self.number_of_classes):
				transformed_X[e,i,:] = self.clause_banks[i].calculate_clause_outputs_predict(encoded_X, e)
		return transformed_X.reshape((X.shape[0], self.number_of_classes*self.number_of_clauses))

	def transform_patchwise(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		transformed_X = np.empty((X.shape[0], self.number_of_classes, self.number_of_clauses//2*self.number_of_patches), dtype=np.uint32)
		for e in range(X.shape[0]):
			for i in range(self.number_of_classes):
				transformed_X[e,i,:] = self.clause_bank[i].calculate_clause_outputs_patchwise(encoded_X, e)
		return transformed_X.reshape((X.shape[0], self.number_of_classes*self.number_of_clauses, self.number_of_patches))

	def literal_clause_frequency(self):
		clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
		literal_frequency = np.zeros(self.clause_banks[0].number_of_literals, dtype=np.uint32)
		for i in range(self.number_of_classes):
			literal_frequency += self.clause_banks[i].calculate_literal_clause_frequency(clause_active)
		return literal_frequency

	def literal_importance(self, the_class, negated_features=False, negative_polarity=False):
		literal_frequency = np.zeros(self.clause_banks[0].number_of_literals, dtype=np.uint32)
		if negated_features:
			if negative_polarity:
				literal_frequency[self.clause_banks[the_class].number_of_literals//2:] += self.clause_banks[the_class].calculate_literal_clause_frequency(self.negative_clauses)[self.clause_banks[the_class].number_of_literals//2:]
			else:
				literal_frequency[self.clause_banks[the_class].number_of_literals//2:] += self.clause_banks[the_class].calculate_literal_clause_frequency(self.positive_clauses)[self.clause_banks[the_class].number_of_literals//2:]
		else:
			if negative_polarity:
				literal_frequency[:self.clause_banks[the_class].number_of_literals//2] += self.clause_banks[the_class].calculate_literal_clause_frequency(self.negative_clauses)[:self.clause_banks[the_class].number_of_literals//2]
			else:
				literal_frequency[:self.clause_banks[the_class].number_of_literals//2] += self.clause_banks[the_class].calculate_literal_clause_frequency(self.positive_clauses)[:self.clause_banks[the_class].number_of_literals//2]

		return literal_frequency

	def clause_precision(self, the_class, polarity, X, Y):
		clause_outputs = self.transform(X).reshape(X.shape[0], self.number_of_classes, 2, self.number_of_clauses//2)[:,the_class, polarity,:]
		if polarity == 0:
			true_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
			false_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
		else:
			true_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
			false_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
		return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0, true_positive_clause_outputs/(true_positive_clause_outputs + false_positive_clause_outputs))

	def clause_recall(self, the_class, polarity, X, Y):
		clause_outputs = self.transform(X).reshape(X.shape[0], self.number_of_classes, 2, self.number_of_clauses//2)[:,the_class, polarity,:]
		if polarity == 0:
			true_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0) / Y[Y==the_class].shape[0]
		else:
			true_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0) / Y[Y!=the_class].shape[0]
		return true_positive_clause_outputs
	
	def get_weight(self, the_class, polarity, clause):
		if polarity == 0:
			return self.weight_banks[the_class].get_weights()[clause]
		else:
			return self.weight_banks[the_class].get_weights()[self.number_of_clauses//2 + clause]

	def set_weight(self, the_class, polarity, clause, weight):
		if polarity == 0:
			self.weight_banks[the_class].get_weights()[clause] = weight
		else:
			self.weight_banks[the_class].get_weights()[self.number_of_clauses//2 + clause] = weight

	def get_ta_action(self, the_class, polarity, clause, ta):
		if polarity == 0:
			return self.clause_banks[the_class].get_ta_action(clause, ta)
		else:
			return self.clause_banks[the_class].get_ta_action(self.number_of_clauses//2 + clause, ta)

	def get_ta_state(self, the_class, polarity, clause, ta):
		if polarity == 0:
			return self.clause_banks[the_class].get_ta_state(clause, ta)
		else:
			return self.clause_banks[the_class].get_ta_state(self.number_of_clauses//2 + clause, ta)

	def set_ta_state(self, the_class, polarity, clause, ta, state):
		if polarity == 0:
			return self.clause_banks[the_class].set_ta_state(clause, ta, state)
		else:
			return self.clause_banks[the_class].set_ta_state(self.number_of_clauses//2 + clause, ta, state)

	def number_of_include_actions(self, the_class, clause):
		return self.clause_banks[the_class].number_of_include_actions(clause)
		
class TMCoalescedClassifier(TMBasis):
	def __init__(self, number_of_clauses, T, s, type_i_ii_ratio = 1.0, type_iii_feedback=False, focused_negative_sampling=False, output_balancing=False, d=200.0, platform = 'CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, type_i_ii_ratio = type_i_ii_ratio, type_iii_feedback=type_iii_feedback, focused_negative_sampling=focused_negative_sampling, output_balancing=output_balancing, d=d, platform = platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta, number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		self.number_of_classes = int(np.max(Y) + 1)
	
		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.random.choice([-1,1], size=self.number_of_clauses).astype(np.int32)))
	
		if self.max_included_literals == None:
			self.max_included_literals = self.clause_bank.number_of_literals

	def update(self, target, e):
		clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.literal_active, self.encoded_X_train, e)
			
		class_sum = np.dot(self.clause_active * self.weight_banks[target].get_weights(), clause_outputs).astype(np.int32)
		class_sum = np.clip(class_sum, -self.T, self.T)
		update_p = (self.T - class_sum)/(2*self.T)

		type_iii_feedback_selection = np.random.choice(2)

		self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, self.clause_active*(self.weight_banks[target].get_weights() >= 0), self.literal_active, self.encoded_X_train, e)
		self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, self.clause_active*(self.weight_banks[target].get_weights() < 0), self.literal_active, self.encoded_X_train, e)
		self.weight_banks[target].increment(clause_outputs, update_p, self.clause_active, True)
		if self.type_iii_feedback and type_iii_feedback_selection == 0:
			self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[target].get_weights() >= 0), self.literal_active, self.encoded_X_train, e, 1)
			self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[target].get_weights() < 0), self.literal_active, self.encoded_X_train, e, 0)

		for i in range(self.number_of_classes):
			if i == target:
				self.update_ps[i] = 0.0
			else:
				self.update_ps[i] = np.dot(self.clause_active * self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
				self.update_ps[i] = np.clip(self.update_ps[i], -self.T, self.T)
				self.update_ps[i] = 1.0*(self.T + self.update_ps[i])/(2*self.T)

		if self.update_ps.sum() == 0:
			return

		if self.focused_negative_sampling:
			not_target = np.random.choice(self.number_of_classes, p=self.update_ps/self.update_ps.sum())
			update_p = self.update_ps[not_target] 
		else: 
			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)
			update_p = self.update_ps[not_target]
	
		self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, self.clause_active * (self.weight_banks[not_target].get_weights() < 0), self.literal_active, self.encoded_X_train, e)
		self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, self.clause_active*(self.weight_banks[not_target].get_weights() >= 0), self.literal_active, self.encoded_X_train, e)
		if self.type_iii_feedback and type_iii_feedback_selection == 1:
			self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[not_target].get_weights() < 0), self.literal_active, self.encoded_X_train, e, 1)
			self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[not_target].get_weights() >= 0), self.literal_active, self.encoded_X_train, e, 0)

		self.weight_banks[not_target].decrement(clause_outputs, update_p, self.clause_active, True)

	def fit(self, X, Y, shuffle=True):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		if not np.array_equal(self.X_train, X):
			self.encoded_X_train = self.clause_bank.prepare_X(X)
			self.X_train = X.copy()

		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		# Clauses are dropped based on their weights
		self.clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
		clause_score = np.zeros(self.number_of_clauses, dtype=np.int32)
		for i in range(self.number_of_classes):
			clause_score += np.abs(self.weight_banks[i].get_weights())
		deactivate = np.random.choice(np.arange(self.number_of_clauses), size=int(self.number_of_clauses*self.clause_drop_p), p = clause_score / clause_score.sum())
		for d in range(deactivate.shape[0]):
			self.clause_active[deactivate[d]] = 0

		# Literals are dropped based on their frequency
		self.literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		literal_clause_frequency = self.literal_clause_frequency()
		if literal_clause_frequency.sum() > 0:
			deactivate = np.random.choice(np.arange(self.clause_bank.number_of_literals), size=int(self.clause_bank.number_of_literals*self.literal_drop_p), p = literal_clause_frequency / literal_clause_frequency.sum())
		else:
			deactivate = np.random.choice(np.arange(self.clause_bank.number_of_literals), size=int(self.clause_bank.number_of_literals*self.literal_drop_p))
		for d in range(deactivate.shape[0]):
			ta_chunk = deactivate[d] // 32
			chunk_pos = deactivate[d] % 32
			self.literal_active[ta_chunk] &= (~(1 << chunk_pos))

		if not self.feature_negation:
			for k in range(self.clause_bank.number_of_literals//2, self.clause_bank.number_of_literals):
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

	def predict(self, X):
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

	def clause_precision(self, the_class, positive_polarity, X, Y):
		clause_outputs = self.transform(X)
		weights = self.weight_banks[the_class].get_weights()
		if positive_polarity == 0:
			positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
			false_positive_clause_outputs = positive_clause_outputs[Y!=the_class].sum(axis=0)
		else:
			positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y!=the_class].sum(axis=0)
			false_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
		
		return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0, 1.0*true_positive_clause_outputs/(true_positive_clause_outputs + false_positive_clause_outputs))

	def clause_recall(self, the_class, positive_polarity, X, Y):
		clause_outputs = self.transform(X)
		weights = self.weight_banks[the_class].get_weights()
		
		if positive_polarity == 0:
			positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0) / Y[Y==the_class].shape[0]
		else:
			positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y!=the_class].sum(axis=0) / Y[Y!=the_class].shape[0]
			
		return true_positive_clause_outputs 

	def get_weight(self, the_class, clause):
		return self.weight_banks[the_class].get_weights()[clause]

	def set_weight(self, the_class, clause, weight):
		self.weight_banks[the_class].get_weights()[clause] = weight

	def number_of_include_actions(self, clause):
		return self.clause_bank.number_of_include_actions(clause)

class TMAutoEncoder(TMBasis):
	def __init__(self, number_of_clauses, T, s, output_active, accumulation=1, type_i_ii_ratio = 1.0, type_iii_feedback=False, focused_negative_sampling=False, output_balancing=False, d=200.0, platform = 'CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		self.output_active = output_active
		self.accumulation = accumulation
		super().__init__(number_of_clauses, T, s, type_i_ii_ratio = type_i_ii_ratio, type_iii_feedback=type_iii_feedback, focused_negative_sampling=focused_negative_sampling, output_balancing=output_balancing, d=d, platform = platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta, number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X):
		self.number_of_classes = self.output_active.shape[0]
		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.random.choice([-1,1], size=self.number_of_clauses).astype(np.int32)))
	
		if self.max_included_literals == None:
			self.max_included_literals = self.clause_bank.number_of_literals

	def update(self, target_output, target_value, encoded_X, clause_active, literal_active):
		all_literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		clause_outputs = self.clause_bank.calculate_clause_outputs_update(all_literal_active, encoded_X, 0)
			
		class_sum = np.dot(clause_active * self.weight_banks[target_output].get_weights(), clause_outputs).astype(np.int32)
		class_sum = np.clip(class_sum, -self.T, self.T)
		
		type_iii_feedback_selection = np.random.choice(2)

		if target_value == 1:
			update_p = (self.T - class_sum)/(2*self.T)

			self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, clause_active*(self.weight_banks[target_output].get_weights() >= 0), literal_active, encoded_X, 0)
			self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, clause_active*(self.weight_banks[target_output].get_weights() < 0), literal_active, encoded_X, 0)
			self.weight_banks[target_output].increment(clause_outputs, update_p, clause_active, True)
			if self.type_iii_feedback and type_iii_feedback_selection == 0:
				self.clause_bank.type_iii_feedback(update_p, self.d, clause_active*(self.weight_banks[target_output].get_weights() >= 0), literal_active, encoded_X, 0, 1)
				self.clause_bank.type_iii_feedback(update_p, self.d, clause_active*(self.weight_banks[target_output].get_weights() < 0), literal_active, encoded_X, 0, 0)
		else:
			update_p = (self.T + class_sum)/(2*self.T)

			self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, clause_active * (self.weight_banks[target_output].get_weights() < 0), literal_active, encoded_X, 0)
			self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, clause_active*(self.weight_banks[target_output].get_weights() >= 0), literal_active, encoded_X, 0)
			self.weight_banks[target_output].decrement(clause_outputs, update_p, clause_active, True)
			if self.type_iii_feedback and type_iii_feedback_selection == 1:
				self.clause_bank.type_iii_feedback(update_p, self.d, clause_active*(self.weight_banks[target_output].get_weights() < 0), literal_active, encoded_X, 0, 1)
				self.clause_bank.type_iii_feedback(update_p, self.d, clause_active*(self.weight_banks[target_output].get_weights() >= 0), literal_active, encoded_X, 0, 0)
		return

	def produce_example(self, the_class, X_csc, X_csr):
		if self.output_balancing:
			target = np.random.choice(2)
			if target == 1:
				target_indices = X_csc[:,self.output_active[the_class]].indices
			else:
				target_indices = np.setdiff1d(np.random.choice(X_csr.shape[0], size=self.accumulation*10, replace=True), X_csc[:,self.output_active[the_class]].indices)
				while target_indices.shape[0] == 0:
					target_indices = np.setdiff1d(np.random.choice(X_csr.shape[0], size=self.accumulation*10, replace=True), X_csc[:,self.output_active[the_class]].indices)

			examples = np.random.choice(target_indices, size=self.accumulation, replace=True)
		else:
			examples = np.random.choice(X_csr.shape[0], replace=True)
			target = X_csr[examples,self.output_active[i]]

		accumulated_X = (X_csr[examples].toarray().sum(axis=0) > 0).astype(np.uint32)

		return (target, self.clause_bank.prepare_X(accumulated_X.reshape((1,-1))))

	def activate_clauses(self):
		# Clauses are dropped based on their weights
		clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
		deactivate = np.random.choice(self.number_of_clauses, size=int(self.number_of_clauses*self.clause_drop_p))
		for d in range(deactivate.shape[0]):
			clause_active[deactivate[d]] = 0

		return clause_active

	def activate_literals(self):		
		# Literals are dropped based on their frequency
		literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		literal_clause_frequency = self.literal_clause_frequency()
		deactivate = np.random.choice(self.clause_bank.number_of_literals, size=int(self.clause_bank.number_of_literals*self.literal_drop_p))
		for d in range(deactivate.shape[0]):
			ta_chunk = deactivate[d] // 32
			chunk_pos = deactivate[d] % 32
			literal_active[ta_chunk] &= (~(1 << chunk_pos))

		if not self.feature_negation:
			for k in range(self.clause_bank.number_of_literals//2, self.clause_bank.number_of_literals):
				ta_chunk = k // 32
				chunk_pos = k % 32
				literal_active[ta_chunk] &= (~(1 << chunk_pos))
		literal_active = literal_active.astype(np.uint32)

		return(literal_active)

	def fit(self, X, number_of_examples=2000, shuffle=True):
		if self.initialized == False:
			self.initialize(X)
			self.initialized = True

		X_csr = csr_matrix(X.reshape(X.shape[0], -1))
		X_csc = csc_matrix(X.reshape(X.shape[0], -1))

		clause_active = self.activate_clauses()
		literal_active = self.activate_literals()
		
		class_index = np.arange(self.number_of_classes, dtype=np.uint32)
		for e in range(number_of_examples):
			np.random.shuffle(class_index)

			average_absolute_weights = np.zeros(self.number_of_clauses, dtype=np.float32)
			for i in class_index:
				 average_absolute_weights += np.absolute(self.weight_banks[i].get_weights())
			average_absolute_weights /= self.number_of_classes
			update_clause = np.random.random(self.number_of_clauses) <= (self.T - np.clip(average_absolute_weights, 0, self.T))/self.T

			for i in class_index:
				(target, encoded_X) = self.produce_example(i, X_csc, X_csr)
				ta_chunk = self.output_active[i] // 32
				chunk_pos = self.output_active[i] % 32
				copy_literal_active_ta_chunk = literal_active[ta_chunk]
				literal_active[ta_chunk] &= ~(1 << chunk_pos)

				self.update(i, target, encoded_X, update_clause*clause_active, literal_active)
				literal_active[ta_chunk] = copy_literal_active_ta_chunk
		return

	def predict(self, X):
		X_csr = csr_matrix(X.reshape(X.shape[0], -1))
		Y = np.ascontiguousarray(np.zeros((self.number_of_classes, X.shape[0]), dtype=np.uint32))

		for e in range(X.shape[0]):
			encoded_X = self.clause_bank.prepare_X(X_csr[e,:].toarray())		

			clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)[0]		
			for i in range(self.number_of_classes):
				class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
				Y[i, e] = (class_sum >= 0)
		return Y

	def literal_importance(self, the_class, negated_features=False, negative_polarity=False):
		literal_frequency = np.zeros(self.clause_bank.number_of_literals, dtype=np.uint32)
		if negated_features:
			if negative_polarity:
				literal_frequency[self.clause_bank.number_of_literals//2:] += self.clause_bank.calculate_literal_clause_frequency(self.weight_banks[the_class].get_weights() < 0)[self.clause_bank.number_of_literals//2:]
			else:
				literal_frequency[self.clause_bank.number_of_literals//2:] += self.clause_bank.calculate_literal_clause_frequency(self.weight_banks[the_class].get_weights() >= 0)[self.clause_bank.number_of_literals//2:]
		else:
			if negative_polarity:
				literal_frequency[:self.clause_bank.number_of_literals//2] += self.clause_bank.calculate_literal_clause_frequency(self.weight_banks[the_class].get_weights() < 0)[:self.clause_bank.number_of_literals//2]
			else:
				literal_frequency[:self.clause_bank.number_of_literals//2] += self.clause_bank.calculate_literal_clause_frequency(self.weight_banks[the_class].get_weights() >= 0)[:self.clause_bank.number_of_literals//2]

		return literal_frequency

	def clause_precision(self, the_class, positive_polarity, X, number_of_examples=2000):
		X_csr = csr_matrix(X.reshape(X.shape[0], -1))
		X_csc = csc_matrix(X.reshape(X.shape[0], -1))

		true_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)
		false_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)

		weights = self.weight_banks[the_class].get_weights()

		clause_active = self.activate_clauses()
		literal_active = self.activate_literals()

		for e in range(number_of_examples):
			(target, encoded_X) = self.produce_example(the_class, X_csc, X_csr)
			
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

		return 1.0*true_positive/(true_positive + false_positive)

	def clause_recall(self, the_class, positive_polarity, X, number_of_examples=2000):
		X_csr = csr_matrix(X.reshape(X.shape[0], -1))
		X_csc = csc_matrix(X.reshape(X.shape[0], -1))

		true_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)
		false_negative = np.zeros(self.number_of_clauses, dtype=np.uint32)

		weights = self.weight_banks[the_class].get_weights()

		for e in range(number_of_examples):
			(target, encoded_X) = self.produce_example(the_class, X_csc, X_csr)

			clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)

			if positive_polarity:
				if target == 1:
					true_positive += (weights >= 0) * clause_outputs
					false_negative += (weights >= 0) * (1 - clause_outputs)
			else:
				if target == 0:
					true_positive += (weights < 0) * clause_outputs
					false_negative += (weights < 0) * (1 - clause_outputs)

		return true_positive/(true_positive + false_negative)

	def get_weight(self, the_class, clause):
		return self.weight_banks[the_class].get_weights()[clause]

	def get_weights(self, the_class):
		return self.weight_banks[the_class].get_weights()

	def set_weight(self, the_class, clause, weight):
		self.weight_banks[the_class].get_weights()[clause] = weight

class TMMultiTaskClassifier(TMBasis):
	def __init__(self, number_of_clauses, T, s, confidence_driven_updating=False, type_i_ii_ratio = 1.0, type_iii_feedback=False, focused_negative_sampling=False, output_balancing=False, d=200.0, platform = 'CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, confidence_driven_updating=confidence_driven_updating, type_i_ii_ratio = type_i_ii_ratio, type_iii_feedback=type_iii_feedback, focused_negative_sampling=focused_negative_sampling, output_balancing=output_balancing, d=d, platform = platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta, number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		self.number_of_classes = len(X)
	
		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(X[0], self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(X[0], self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.random.choice([-1,1], size=self.number_of_clauses).astype(np.int32)))
		
		if self.max_included_literals == None:
			self.max_included_literals = self.clause_bank.number_of_literals

	def fit(self, X, Y, shuffle=True):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		X_csr = {}
		for i in range(self.number_of_classes):
			X_csr[i] = csr_matrix(X[i].reshape(X[i].shape[0], -1))

		# Clauses are dropped based on their weights
		self.clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
		clause_score = np.zeros(self.number_of_clauses, dtype=np.int32)
		for i in range(self.number_of_classes):
			clause_score += np.abs(self.weight_banks[i].get_weights())
		deactivate = np.random.choice(np.arange(self.number_of_clauses), size=int(self.number_of_clauses*self.clause_drop_p), p = clause_score / clause_score.sum())
		for d in range(deactivate.shape[0]):
			self.clause_active[deactivate[d]] = 0

		# Literals are dropped based on their frequency
		self.literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		literal_clause_frequency = self.literal_clause_frequency()
		if literal_clause_frequency.sum() > 0:
			deactivate = np.random.choice(np.arange(self.clause_bank.number_of_literals), size=int(self.clause_bank.number_of_literals*self.literal_drop_p), p = literal_clause_frequency / literal_clause_frequency.sum())
		else:
			deactivate = np.random.choice(np.arange(self.clause_bank.number_of_literals), size=int(self.clause_bank.number_of_literals*self.literal_drop_p))
		for d in range(deactivate.shape[0]):
			ta_chunk = deactivate[d] // 32
			chunk_pos = deactivate[d] % 32
			self.literal_active[ta_chunk] &= (~(1 << chunk_pos))

		if not self.feature_negation:
			for k in range(self.clause_bank.number_of_literals//2, self.clause_bank.number_of_literals):
				ta_chunk = k // 32
				chunk_pos = k % 32
				self.literal_active[ta_chunk] &= (~(1 << chunk_pos))

		self.literal_active = self.literal_active.astype(np.uint32)

		shuffled_index = np.arange(X[0].shape[0])
		if shuffle:
			np.random.shuffle(shuffled_index)

		class_index = np.arange(self.number_of_classes, dtype=np.uint32)
		for e in shuffled_index:
			np.random.shuffle(class_index)
			for i in class_index:
				encoded_X = self.clause_bank.prepare_X(X_csr[i][e,:].toarray())
				clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.literal_active, encoded_X, 0)
			
				class_sum = np.dot(self.clause_active * self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
				class_sum = np.clip(class_sum, -self.T, self.T)

				type_iii_feedback_selection = np.random.choice(2)

				if Y[i, e] == 1:
					if self.confidence_driven_updating:
						update_p = 1.0*(self.T - np.absolute(class_sum))/self.T
					else:
						update_p = (self.T - class_sum)/(2*self.T)

					self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, self.clause_active*(self.weight_banks[i].get_weights() >= 0), self.literal_active, encoded_X, 0)
					self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, self.clause_active*(self.weight_banks[i].get_weights() < 0), self.literal_active, encoded_X, 0)
					self.weight_banks[i].increment(clause_outputs, update_p, self.clause_active, True)
					if self.type_iii_feedback and type_iii_feedback_selection == 0:
						self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[i].get_weights() >= 0), self.literal_active, encoded_X, 0, 1)
						self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[i].get_weights() < 0), self.literal_active, encoded_X, 0, 0)
				else:
					if self.confidence_driven_updating:
						update_p = 1.0*(self.T - np.absolute(class_sum))/self.T
					else:
						update_p = (self.T + class_sum)/(2*self.T)

					self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, self.max_included_literals, self.clause_active * (self.weight_banks[i].get_weights() < 0), self.literal_active, encoded_X, 0)
					self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, self.clause_active*(self.weight_banks[i].get_weights() >= 0), self.literal_active, encoded_X, 0)
					self.weight_banks[i].decrement(clause_outputs, update_p, self.clause_active, True)
					if self.type_iii_feedback and type_iii_feedback_selection == 1:
						self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[i].get_weights() < 0), self.literal_active, encoded_X, 0, 1)
						self.clause_bank.type_iii_feedback(update_p, self.d, self.clause_active*(self.weight_banks[i].get_weights() >= 0), self.literal_active, encoded_X, 0, 0)
		return

	def predict(self, X):
		Y = np.ascontiguousarray(np.zeros((len(X), X[0].shape[0]), dtype=np.uint32))

		X_csr = {}
		for i in range(self.number_of_classes):
			X_csr[i] = csr_matrix(X[i].reshape(X[i].shape[0], -1))
		
		for i in range(self.number_of_classes):
			for e in range(X[i].shape[0]):
				encoded_X = self.clause_bank.prepare_X(X_csr[i][e,:].toarray())		
				clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)			
				class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
				Y[i, e] = (class_sum >= 0)
		return Y

	def clause_precision(self, the_class, positive_polarity, X, Y):
		clause_outputs = self.transform(X)
		weights = self.weight_banks[the_class].get_weights()
		if positive_polarity == 0:
			positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
			false_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
		else:
			positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
			false_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
		
		return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0, 1.0*true_positive_clause_outputs/(true_positive_clause_outputs + false_positive_clause_outputs))

	def clause_recall(self, the_class, positive_polarity, X, Y):
		clause_outputs = self.transform(X)
		weights = self.weight_banks[the_class].get_weights()
		
		if positive_polarity == 0:
			positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
		else:
			positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y!=the_class].sum(axis=0)
			
		return true_positive_clause_outputs / Y[Y==the_class].shape[0]

	def get_weight(self, the_class, clause):
		return self.weight_banks[the_class].get_weights()[clause]

	def get_weights(self, the_class):
		return self.weight_banks[the_class].get_weights()

	def set_weight(self, the_class, clause, weight):
		self.weight_banks[the_class].get_weights()[clause] = weight


class TMMultiChannelClassifier(TMBasis):
	def __init__(self, number_of_clauses, global_T, T, s, type_i_ii_ratio = 1.0, platform = 'CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, number_of_state_bits_ta=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, type_i_ii_ratio = type_i_ii_ratio, platform = platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits_ta=number_of_state_bits_ta, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)
		self.global_T = global_T

	def initialize(self, X, Y):
		self.number_of_classes = int(np.max(Y) + 1)
	
		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(X[0], self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(X[0], self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.random.choice([-1,1], size=self.number_of_clauses).astype(np.int32)))
		
		self.X_train = {}
		self.X_test = {}
		for c in range(X.shape[0]):
			self.X_train[c] = np.zeros(0, dtype=np.uint32)
			self.X_test[c] = np.zeros(0, dtype=np.uint32)

		self.encoded_X_train = {}
		self.encoded_X_test = {}

	def fit(self, X, Y, shuffle=True):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		for c in range(X.shape[0]):
			if not np.array_equal(self.X_train[c], X[c]):
				self.encoded_X_train[c] = self.clause_bank.prepare_X(X[c])
				self.X_train[c] = X[c].copy()

		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		# Clauses are dropped based on their weights
		clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
		clause_score = np.zeros(self.number_of_clauses, dtype=np.int32)
		for i in range(self.number_of_classes):
			clause_score += np.abs(self.weight_banks[i].get_weights())
		deactivate = np.random.choice(np.arange(self.number_of_clauses), size=int(self.number_of_clauses*self.clause_drop_p), p = clause_score / clause_score.sum())
		for d in range(deactivate.shape[0]):
			clause_active[deactivate[d]] = 0

		# Literals are dropped based on their frequency
		literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		literal_clause_frequency = self.literal_clause_frequency()
		if literal_clause_frequency.sum() > 0:
			deactivate = np.random.choice(np.arange(self.clause_bank.number_of_literals), size=int(self.clause_bank.number_of_literals*self.literal_drop_p), p = literal_clause_frequency / literal_clause_frequency.sum())
		else:
			deactivate = np.random.choice(np.arange(self.clause_bank.number_of_literals), size=int(self.clause_bank.number_of_literals*self.literal_drop_p))
		for d in range(deactivate.shape[0]):
			ta_chunk = deactivate[d] // 32
			chunk_pos = deactivate[d] % 32
			literal_active[ta_chunk] &= (~(1 << chunk_pos))

		if not self.feature_negation:
			for k in range(self.clause_bank.number_of_literals//2, self.clause_bank.number_of_literals):
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
				clause_outputs.append(self.clause_bank.calculate_clause_outputs_update(literal_active, self.encoded_X_train[c], e).copy())
			
			global_class_sum = 0
			for c in range(X.shape[0]):
				local_class_sum[c] = np.dot(clause_active * self.weight_banks[target].get_weights(), clause_outputs[c]).astype(np.int32)
				local_class_sum[c] = np.clip(local_class_sum[c], -self.T, self.T)
				global_class_sum += local_class_sum[c]
			global_class_sum = np.clip(global_class_sum, -self.global_T[target][0], self.global_T[target][1])
			global_update_p = 1.0*(self.global_T[target][1] - global_class_sum)/(self.global_T[target][0]+self.global_T[target][1])

			for c in range(X.shape[0]):
				local_update_p = 1.0*(self.T - local_class_sum[c])/(2*self.T)
				update_p = np.minimum(local_update_p, global_update_p)
				self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s[target], self.boost_true_positive_feedback, clause_active*(self.weight_banks[target].get_weights() >= 0), literal_active, self.encoded_X_train[c], e)
				self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, clause_active*(self.weight_banks[target].get_weights() < 0), literal_active, self.encoded_X_train[c], e)
				self.weight_banks[target].increment(clause_outputs[c], update_p, clause_active, True)

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			global_class_sum = 0.0
			for c in range(X.shape[0]):				
				local_class_sum[c] = np.dot(clause_active * self.weight_banks[not_target].get_weights(), clause_outputs[c]).astype(np.int32)
				local_class_sum[c] = np.clip(local_class_sum[c], -self.T, self.T)
				global_class_sum += local_class_sum[c]
			global_class_sum = np.clip(global_class_sum, -self.global_T[not_target][0], self.global_T[not_target][1])
			global_update_p = 1.0*(self.global_T[not_target][0] + global_class_sum)/(self.global_T[not_target][0]+self.global_T[not_target][1])

			for c in range(X.shape[0]):
				local_update_p = 1.0*(self.T + local_class_sum[c])/(2*self.T)
				update_p = np.minimum(local_update_p, global_update_p)
				self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s[not_target], self.boost_true_positive_feedback, clause_active * (self.weight_banks[not_target].get_weights() < 0), literal_active, self.encoded_X_train[c], e)
				self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, clause_active*(self.weight_banks[not_target].get_weights() >= 0), literal_active, self.encoded_X_train[c], e)
				self.weight_banks[not_target].decrement(clause_outputs[c], update_p, clause_active, True)
		return

	def predict(self, X):
		for c in range(X.shape[0]):
			if not np.array_equal(self.X_test[c], X[c]):
				self.encoded_X_test[c] = self.clause_bank.prepare_X(X[c])
				self.X_test[c] = X[c].copy()

		Y = np.ascontiguousarray(np.zeros(X.shape[1], dtype=np.uint32))

		for e in range(X.shape[1]):
			max_class_sum = -maxsize
			max_class = 0

			clause_outputs = []
			for c in range(X.shape[0]):
				clause_outputs.append(self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test[c], e).copy())
			
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
			positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
			false_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
		else:
			positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
			false_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
		
		return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0, 1.0*true_positive_clause_outputs/(true_positive_clause_outputs + false_positive_clause_outputs))

	def clause_recall(self, the_class, positive_polarity, X, Y):
		clause_outputs = self.transform(X)
		weights = self.weight_banks[the_class].get_weights()
		
		if positive_polarity == 0:
			positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
		else:
			positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
			true_positive_clause_outputs = positive_clause_outputs[Y!=the_class].sum(axis=0)
			
		return true_positive_clause_outputs / Y[Y==the_class].shape[0]

	def get_weight(self, the_class, clause):
		return self.weight_banks[the_class].get_weights()[clause]

	def set_weight(self, the_class, clause, weight):
		self.weight_banks[the_class].get_weights()[clause] = weight

class TMOneVsOneClassifier(TMBasis):
	def __init__(self, number_of_clauses, T, s, type_i_ii_ratio = 1.0, platform = 'CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, number_of_state_bits_ta=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, type_i_ii_ratio = type_i_ii_ratio, platform = platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits_ta=number_of_state_bits_ta, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		self.number_of_classes = int(np.max(Y) + 1)
		self.number_of_outputs = self.number_of_classes * (self.number_of_classes-1)

		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_outputs):
			self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))
		
	def fit(self, X, Y, shuffle=True):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		if not np.array_equal(self.X_train, X):
			self.encoded_X_train = self.clause_bank.prepare_X(X)
			self.X_train = X.copy()

		Ym = np.ascontiguousarray(Y).astype(np.uint32)
		
		clause_active = np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))
		literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		if not self.feature_negation:
			for k in range(self.clause_bank.number_of_literals//2, self.clause_bank.number_of_literals):
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

			output = target * (self.number_of_classes-1) + not_target - (not_target > target)

			class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			update_p = (self.T - class_sum)/(2*self.T)

			self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, clause_active*(self.weight_banks[output].get_weights() >= 0), literal_active, self.encoded_X_train, e)
			self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, clause_active*(self.weight_banks[output].get_weights() < 0), literal_active, self.encoded_X_train, e)
			self.weight_banks[output].increment(clause_outputs, update_p, clause_active, True)

			output = not_target * (self.number_of_classes-1) + target - (target > not_target)

			class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			update_p = (self.T + class_sum)/(2*self.T)
		
			self.clause_bank.type_i_feedback(update_p*self.type_i_p, self.s, self.boost_true_positive_feedback, clause_active * (self.weight_banks[output].get_weights() < 0), literal_active, self.encoded_X_train, e)
			self.clause_bank.type_ii_feedback(update_p*self.type_ii_p, clause_active*(self.weight_banks[output].get_weights() >= 0), literal_active, self.encoded_X_train, e)
			self.weight_banks[output].decrement(clause_outputs, update_p, clause_active, True)
		return

	def predict(self, X):
		if not np.array_equal(self.X_test, X):
			self.encoded_X_test = self.clause_bank.prepare_X(X)
			self.X_test = X.copy()

		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

		for e in range(X.shape[0]):
			clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)

			max_class_sum = -self.T*self.number_of_classes
			max_class = 0
			for i in range(self.number_of_classes):
				class_sum = 0
				for output in range(i * (self.number_of_classes - 1), (i+1) * (self.number_of_classes-1)):
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
				positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
				true_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
				false_positive_clause_outputs = positive_clause_outputs[Y==other_class].sum(axis=0)
				precision[i] = np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0, true_positive_clause_outputs/(true_positive_clause_outputs + false_positive_clause_outputs))
			else:
				positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
				true_positive_clause_outputs = positive_clause_outputs[Y==other_class].sum(axis=0)
				false_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
				precision[i] = np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0, true_positive_clause_outputs/(true_positive_clause_outputs + false_positive_clause_outputs))
	
		return precision

	def clause_recall(self, the_class, positive_polarity, X, Y):
		clause_outputs = self.transform(X)
		recall = np.zeros((self.number_of_classes - 1, self.number_of_clauses))
		for i in range(self.number_of_classes - 1):
			other_class = i + (i >= the_class)
			output = the_class * (self.number_of_classes - 1) + i
			weights = self.weight_banks[output].get_weights()
			if positive_polarity:
				positive_clause_outputs = (weights >= 0)[:,np.newaxis].transpose() * clause_outputs
				true_positive_clause_outputs = positive_clause_outputs[Y==the_class].sum(axis=0)
				recall[i] =	true_positive_clause_outputs / Y[Y==the_class].shape[0]
			else:
				positive_clause_outputs = (weights < 0)[:,np.newaxis].transpose() * clause_outputs
				true_positive_clause_outputs = positive_clause_outputs[Y==other_class].sum(axis=0)
				recall[i] = true_positive_clause_outputs / Y[Y==other_class].shape[0]
		return recall

	def get_weight(self, output, clause):
		return self.weight_banks[output].get_weights()[clause]

	def set_weight(self, output, weight):
		self.weight_banks[output].get_weights()[output] = weight

class TMRegressor(TMBasis):
	def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, feature_negation=True, boost_true_positive_feedback=1, number_of_state_bits_ta=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim, feature_negation=feature_negation, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits_ta=number_of_state_bits_ta, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		self.max_y = np.max(Y)
		self.min_y = np.min(Y)

		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind, self.patch_dim)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim)
		else:
			print("Unknown Platform")
			sys.exit(-1)
			
		self.weight_bank = WeightBank(np.ones(self.number_of_clauses).astype(np.int32))

	def fit(self, X, Y, shuffle=True):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		if not np.array_equal(self.X_train, X):
			self.encoded_X_train = self.clause_bank.prepare_X(X)
			self.X_train = X.copy()
		encoded_Y = np.ascontiguousarray(((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32))

		clause_active = np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))
		literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
		if not self.feature_negation:
			for k in range(self.clause_bank.number_of_literals//2, self.clause_bank.number_of_literals):
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

			update_p = (1.0*prediction_error/self.T)**2

			if pred_y < encoded_Y[e]:
				self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active, literal_active, self.encoded_X_train, e)
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
			Y[e] = 1.0*pred_y * (self.max_y - self.min_y)/(self.T) + self.min_y
		return Y

	def get_weight(self, clause):
		return self.weight_bank.get_weights()[clause]

	def set_weight(self, clause, weight):
		self.weight_banks.get_weights()[clause] = weight

