# Copyright (c) 2021 Ole-Christoffer Granmo

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

import tmu.tools
import sys
import numpy as np
from tmu.clause_bank import ClauseBank
from tmu.weight_bank import WeightBank
from scipy.sparse import csr_matrix

from time import time

class TMBasis():
	def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.platform = platform
		self.patch_dim = patch_dim
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.weighted_clauses = weighted_clauses

		self.clause_drop_p = clause_drop_p
		self.literal_drop_p = literal_drop_p

		self.initialized = False

	def initialize(self, X, patch_dim):
		if len(X.shape) == 2:
			self.dim = (X.shape[1], 1 ,1)
		elif len(X.shape) == 3:
			self.dim = (X.shape[1], X.shape[2], 1)
		elif len(X.shape) == 4:
			self.dim = (X.shape[1], X.shape[2], X.shape[3])

		if self.patch_dim == None:
			self.patch_dim = (X.shape[1], 1)

		self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))
		self.number_of_literals = self.number_of_features*2
		
		self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1)*(self.dim[1] - self.patch_dim[1] + 1))
		self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)

	def clause_co_occurrence(self, X, percentage=False):
		clause_outputs = csr_matrix(self.transform(X))
		if percentage:
			return clause_outputs.transpose().dot(clause_outputs).multiply(1.0/clause_outputs.sum(axis=0))
		else:
			return clause_outputs.transpose().dot(clause_outputs)

	def transform(self, X):
		encoded_X = self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		transformed_X = np.empty((X.shape[0], self.number_of_clauses), dtype=np.uint32)
		for e in range(X.shape[0]):
			transformed_X[e,:] = self.clause_bank.calculate_clause_outputs_update(encoded_X, e)
		return transformed_X

	def transform_patchwise(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		transformed_X = np.empty((X.shape[0], self.number_of_clauses*self.number_of_patches), dtype=np.uint32)
		for e in range(X.shape[0]):
			transformed_X[e,:] = self.clause_bank.calculate_clause_outputs_patchwise(encoded_X, e)
		return transformed_X.reshape((X.shape[0], self.number_of_clauses, self.number_of_patches))

	def get_ta_action(self, clause, ta):
		return self.clause_bank.get_ta_action(clause, ta)

	def get_ta_state(self, clause, ta):
		return self.clause_bank.get_ta_state(clause, ta)

	def set_ta_state(self, clause, ta, state):
		return self.clause_bank.set_ta_state(clause, ta, state)

class TMClassifier(TMBasis):
	def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		super().initialize(X, self.patch_dim)

		self.number_of_classes = int(np.max(Y) + 1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.concatenate((np.ones(self.number_of_clauses//2, dtype=np.int32), -1*np.ones(self.number_of_clauses//2, dtype=np.int32)))))
		
		self.clause_banks = []
		if self.platform == 'CPU':
			for i in range(self.number_of_classes):
				self.clause_banks.append(ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches))
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			for i in range(self.number_of_classes):
				self.clause_banks.append(ClauseBankCUDA(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, X, Y))
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.positive_clauses = np.concatenate((np.ones(self.number_of_clauses//2, dtype=np.int32), np.zeros(self.number_of_clauses//2, dtype=np.int32)))
		self.negative_clauses = np.concatenate((np.zeros(self.number_of_clauses//2, dtype=np.int32), np.ones(self.number_of_clauses//2, dtype=np.int32)))

	def fit(self, X, Y):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		encoded_X = self.clause_banks[0].prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		Ym = np.ascontiguousarray(Y).astype(np.uint32)
		
		clause_active = []
		for i in range(self.number_of_classes):
			clause_active.append(np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32)))

		for e in range(X.shape[0]):
			target = Ym[e]
			
			clause_outputs = self.clause_banks[target].calculate_clause_outputs_update(encoded_X, e)
			class_sum = np.dot(clause_active[target] * self.weight_banks[target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)

			update_p = (self.T - class_sum)/(2*self.T)

			if self.weighted_clauses:
				self.weight_banks[target].increment(clause_outputs, update_p, clause_active[target], False)
			self.clause_banks[target].type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active[target]*self.positive_clauses, encoded_X, e)
			self.clause_banks[target].type_ii_feedback(update_p, clause_active[target]*self.negative_clauses, encoded_X, e)

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			clause_outputs = self.clause_banks[not_target].calculate_clause_outputs_update(encoded_X, e)
			class_sum = np.dot(clause_active[not_target] * self.weight_banks[not_target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)

			update_p = (self.T + class_sum)/(2*self.T)
		
			if self.weighted_clauses:
				self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active[not_target], False)			
			self.clause_banks[not_target].type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active[not_target]*self.negative_clauses, encoded_X, e)
			self.clause_banks[not_target].type_ii_feedback(update_p, clause_active[not_target]*self.positive_clauses, encoded_X, e)
		
		return

	def predict(self, X):		
		encoded_X = self.clause_banks[0].prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))

		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
		for e in range(X.shape[0]):
			max_class_sum = -self.T
			max_class = 0
			for i in range(self.number_of_classes):
				class_sum = np.dot(self.weight_banks[i].get_weights(), self.clause_banks[i].calculate_clause_outputs_predict(encoded_X, e)).astype(np.int32)
				class_sum = np.clip(class_sum, -self.T, self.T)
				if class_sum > max_class_sum:
					max_class_sum = class_sum
					max_class = i
			Y[e] = max_class
		return Y

	def transform(self, X):
		encoded_X = self.clause_banks[0].prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		transformed_X = np.empty((X.shape[0], self.number_of_classes, self.number_of_clauses), dtype=np.uint32)
		for e in range(X.shape[0]):
			for i in range(self.number_of_classes):
				transformed_X[e,i,:] = self.clause_banks[i].calculate_clause_outputs_update(encoded_X, e)
		return transformed_X.reshape((X.shape[0], self.number_of_classes*self.number_of_clauses))

	def transform_patchwise(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		transformed_X = np.empty((X.shape[0], self.number_of_classes, self.number_of_clauses//2*self.number_of_patches), dtype=np.uint32)
		for e in range(X.shape[0]):
			for i in range(self.number_of_classes):
				transformed_X[e,i,:] = self.clause_bank[i].calculate_clause_outputs_patchwise(encoded_X, e)
		return transformed_X.reshape((X.shape[0], self.number_of_classes*self.number_of_clauses, self.number_of_patches))

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
			true_positive_clause_outputs = clause_outputs[Y==the_class].sum(axis=0)
		else:
			true_positive_clause_outputs = clause_outputs[Y!=the_class].sum(axis=0)
		return true_positive_clause_outputs / Y[Y==the_class].shape[0]

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

class TMCoalescedClassifier(TMBasis):
	def __init__(self, number_of_clauses, T, s, platform = 'CPU', patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, platform = platform, patch_dim=patch_dim, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		super().initialize(X, self.patch_dim)

		self.number_of_classes = int(np.max(Y) + 1)
	
		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, X, Y)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))
		
	def fit(self, X, Y):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		encoded_X = self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))

		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		clause_active = np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))
		for e in range(X.shape[0]):
			target = Ym[e]

			clause_outputs = self.clause_bank.calculate_clause_outputs_update(encoded_X, e)
			
			class_sum = np.dot(clause_active * self.weight_banks[target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			update_p = (self.T - class_sum)/(2*self.T)

			self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active*(self.weight_banks[target].get_weights() >= 0), encoded_X, e)
			self.clause_bank.type_ii_feedback(update_p, clause_active*(self.weight_banks[target].get_weights() < 0), encoded_X, e)
			self.weight_banks[target].increment(clause_outputs, update_p, clause_active, True)

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			class_sum = np.dot(clause_active * self.weight_banks[not_target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			update_p = (self.T + class_sum)/(2*self.T)
		
			self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active * (self.weight_banks[not_target].get_weights() < 0), encoded_X, e)
			self.clause_bank.type_ii_feedback(update_p, clause_active*(self.weight_banks[not_target].get_weights() >= 0), encoded_X, e)
			
			self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active, True)
		return

	def predict(self, X):		
		encoded_X =	self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))

		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
		for e in range(X.shape[0]):
			max_class_sum = -self.T
			max_class = 0
			clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, e)			
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
	def __init__(self, number_of_clauses, T, s, platform = 'CPU', patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, platform = platform, patch_dim=patch_dim, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		super().initialize(X, self.patch_dim)

		self.number_of_classes = int(np.max(Y) + 1)
		self.number_of_outputs = self.number_of_classes * (self.number_of_classes-1)

		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, X, Y)
		else:
			print("Unknown Platform")
			sys.exit(-1)

		self.weight_banks = []
		for i in range(self.number_of_outputs):
			self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))
		
	def fit(self, X, Y):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		encoded_X = self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		Ym = np.ascontiguousarray(Y).astype(np.uint32)
		
		clause_active = np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))
		for e in range(X.shape[0]):
			clause_outputs = self.clause_bank.calculate_clause_outputs_update(encoded_X, e)
			
			target = Ym[e]
			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			output = target * (self.number_of_classes-1) + not_target - (not_target > target)

			class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			update_p = (self.T - class_sum)/(2*self.T)

			self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active*(self.weight_banks[output].get_weights() >= 0), encoded_X, e)
			self.clause_bank.type_ii_feedback(update_p, clause_active*(self.weight_banks[output].get_weights() < 0), encoded_X, e)
			self.weight_banks[output].increment(clause_outputs, update_p, clause_active, True)

			output = not_target * (self.number_of_classes-1) + target - (target > not_target)

			class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			update_p = (self.T + class_sum)/(2*self.T)
		
			self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active * (self.weight_banks[output].get_weights() < 0), encoded_X, e)
			self.clause_bank.type_ii_feedback(update_p, clause_active*(self.weight_banks[output].get_weights() >= 0), encoded_X, e)
			self.weight_banks[output].decrement(clause_outputs, update_p, clause_active, True)
		return

	def predict(self, X):
		encoded_X = self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

		for e in range(X.shape[0]):
			clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, e)

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
	def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses, clause_drop_p = clause_drop_p, literal_drop_p = literal_drop_p)

	def initialize(self, X, Y):
		super().initialize(X, self.patch_dim)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)

		if self.platform == 'CPU':
			self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches)
		elif self.platform == 'CUDA':
			from tmu.clause_bank_cuda import ClauseBankCUDA
			self.clause_bank = ClauseBankCUDA(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, X, Y)
		else:
			print("Unknown Platform")
			sys.exit(-1)
			
		self.weight_bank = WeightBank(np.ones(self.number_of_clauses).astype(np.int32))

	def fit(self, X, Y):
		if self.initialized == False:
			self.initialize(X, Y)
			self.initialized = True

		encoded_X = self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		encoded_Y = np.ascontiguousarray(((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32))

		clause_active = np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))
		for e in range(X.shape[0]):
			clause_outputs = self.clause_bank.calculate_clause_outputs_update(encoded_X, e)

			pred_y = np.dot(clause_active * self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
			pred_y = np.clip(pred_y, 0, self.T)
			prediction_error = pred_y - encoded_Y[e]; 

			update_p = (1.0*prediction_error/self.T)**2

			if pred_y < encoded_Y[e]:
				self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active, encoded_X, e)
				if self.weighted_clauses:
					self.weight_bank.increment(clause_outputs, update_p, clause_active, False)
			elif pred_y > encoded_Y[e]:
				self.clause_bank.type_ii_feedback(update_p, clause_active, encoded_X, e)
				if self.weighted_clauses:
					self.weight_bank.decrement(clause_outputs, update_p, clause_active, False)
		return

	def predict(self, X):
		encoded_X = self.clause_bank.prepare_X(tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0))
		Y = np.ascontiguousarray(np.zeros(X.shape[0]))
		for e in range(X.shape[0]):
			clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, e)
			pred_y = np.dot(self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
			Y[e] = 1.0*pred_y * (self.max_y - self.min_y)/(self.T) + self.min_y
		return Y

	def get_weight(self, clause):
		return self.weight_bank.get_weights()[clause]

	def set_weight(self, clause, weight):
		self.weight_banks.get_weights()[clause] = weight

