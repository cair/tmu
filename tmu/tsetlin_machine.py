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

import numpy as np

from tmu.clause_bank import ClauseBank
from tmu.weight_bank import WeightBank

class TMClassifier():
	def __init__(self, number_of_clauses, T, s, patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.patch_dim = patch_dim
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.weighted_clauses = weighted_clauses

		self.clause_drop_p = clause_drop_p
		self.literal_drop_p = literal_drop_p

		self.initialized = False

	def initialize(self, X, Y):
		self.number_of_classes = int(np.max(Y) + 1)

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

		self.clause_banks = []
		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append((WeightBank(np.ones(self.number_of_clauses//2).astype(np.int32)), WeightBank(np.ones(self.number_of_clauses//2).astype(np.int32))))
			self.clause_banks.append((ClauseBank(self.number_of_clauses//2, self.number_of_literals, self.number_of_state_bits, self.number_of_patches), ClauseBank(self.number_of_clauses//2, self.number_of_literals, self.number_of_state_bits, self.number_of_patches)))
		
	def fit(self, X, Y, incremental=False):
		if self.initialized == False or incremental == False:
			self.initialize(X, Y)
			self.initialized = True

		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)
		
		clause_active = []
		for i in range(self.number_of_classes):
			clause_active.append((np.ascontiguousarray(np.random.choice(2, self.number_of_clauses//2, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32)), np.ascontiguousarray(np.random.choice(2, self.number_of_clauses//2, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))))

		for e in range(X.shape[0]):
			target = Ym[e]

			positive_clause_outputs = self.clause_banks[target][0].calculate_clause_outputs_update(encoded_X[e,:])
			negative_clause_outputs = self.clause_banks[target][1].calculate_clause_outputs_update(encoded_X[e,:]) 
			class_sum = np.dot(clause_active[target][0] * self.weight_banks[target][0].get_weights(), positive_clause_outputs).astype(np.int32)
			class_sum -= np.dot(clause_active[target][1] * self.weight_banks[target][1].get_weights(), negative_clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)

			update_p = (self.T - class_sum)/(2*self.T)

			if self.weighted_clauses:
				self.weight_banks[target][0].increment(positive_clause_outputs, update_p, clause_active[target][0])
				self.weight_banks[target][1].decrement(negative_clause_outputs, update_p, clause_active[target][1], False)

			self.clause_banks[target][0].type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active[target][0], encoded_X[e,:])
			self.clause_banks[target][1].type_ii_feedback(update_p, clause_active[target][1], encoded_X[e,:])

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			positive_clause_outputs = self.clause_banks[not_target][0].calculate_clause_outputs_update(encoded_X[e,:])
			negative_clause_outputs = self.clause_banks[not_target][1].calculate_clause_outputs_update(encoded_X[e,:]) 
			class_sum = np.dot(clause_active[not_target][0] * self.weight_banks[not_target][0].get_weights(), positive_clause_outputs).astype(np.int32)
			class_sum -= np.dot(clause_active[not_target][1] * self.weight_banks[not_target][1].get_weights(), negative_clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			
			update_p = (self.T + class_sum)/(2*self.T)
		
			if self.weighted_clauses:
				self.weight_banks[not_target][1].increment(negative_clause_outputs, update_p, clause_active[not_target][1])
				self.weight_banks[not_target][0].decrement(positive_clause_outputs, update_p, clause_active[not_target][0], False)			

			self.clause_banks[not_target][1].type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active[not_target][1], encoded_X[e,:])
			self.clause_banks[not_target][0].type_ii_feedback(update_p, clause_active[not_target][0], encoded_X[e,:])			
		return

	def predict(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
		for e in range(X.shape[0]):
			max_class_sum = -self.T
			max_class = 0
			for i in range(self.number_of_classes):
				class_sum = np.dot(self.weight_banks[i][0].get_weights(), self.clause_banks[i][0].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
				class_sum -= np.dot(self.weight_banks[i][1].get_weights(), self.clause_banks[i][1].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
				class_sum = np.clip(class_sum, -self.T, self.T)
				if class_sum > max_class_sum:
					max_class_sum = class_sum
					max_class = i
			Y[e] = max_class
		return Y

	def ta_action(self, data_class, clause, ta):
		polarity = clause >= self.number_of_clauses//2
		clause_pos = clause % (self.number_of_clauses//2)
		return self.clause_banks[data_class][polarity].ta_action(clause_pos, ta)

class TMCoalescedClassifier():
	def __init__(self, number_of_clauses, T, s, patch_dim=None, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.patch_dim = patch_dim
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.weighted_clauses = weighted_clauses

		self.clause_drop_p = clause_drop_p
		self.literal_drop_p = literal_drop_p

		self.initialized = False

	def initialize(self, X, Y):
		self.number_of_classes = int(np.max(Y) + 1)

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

		self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches)
		self.weight_banks = []
		for i in range(self.number_of_classes):
			self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))
		
	def fit(self, X, Y, incremental=False):
		if self.initialized == False or incremental == False:
			self.initialize(X, Y)
			self.initialized = True

		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)
		
		clause_active = np.ascontiguousarray(np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(np.int32))

		for e in range(X.shape[0]):
			target = Ym[e]

			clause_outputs = self.clause_bank.calculate_clause_outputs_update(encoded_X[e,:])

			class_sum = np.dot(clause_active * self.weight_banks[target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)

			update_p = (self.T - class_sum)/(2*self.T)

			self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active*(self.weight_banks[target].get_weights() >= 0), encoded_X[e,:])
			self.clause_bank.type_ii_feedback(update_p, clause_active*(self.weight_banks[target].get_weights() < 0), encoded_X[e,:])

			self.weight_banks[target].increment(clause_outputs, update_p, clause_active)

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			class_sum = np.dot(clause_active * self.weight_banks[not_target].get_weights(), clause_outputs).astype(np.int32)
			class_sum = np.clip(class_sum, -self.T, self.T)
			
			update_p = (self.T + class_sum)/(2*self.T)
		
			self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active * (self.weight_banks[not_target].get_weights() < 0), encoded_X[e,:])
			self.clause_bank.type_ii_feedback(update_p, clause_active*(self.weight_banks[not_target].get_weights() >= 0), encoded_X[e,:])

			self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active, True)
		return

	def predict(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
		for e in range(X.shape[0]):
			max_class_sum = -self.T
			max_class = 0
			clause_outputs = self.clause_bank.calculate_clause_outputs_update(encoded_X[e,:])
			for i in range(self.number_of_classes):
				class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
				class_sum = np.clip(class_sum, -self.T, self.T)
				if class_sum > max_class_sum:
					max_class_sum = class_sum
					max_class = i
			Y[e] = max_class
		return Y

	def ta_action(self, clause, ta):
		return self.clause_bank.ta_action(clause, ta)