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

class TMBase():
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback, number_of_state_bits, weighted_clauses, clause_drop_p, literal_drop_p):
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.weighted_clauses = weighted_clauses

		self.clause_drop_p = clause_drop_p
		self.literal_drop_p = literal_drop_p

		self.initialize = True

class TMClassifier(TMBase):
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, boost_true_positive_feedback, number_of_state_bits, weighted_clauses, clause_drop_p, literal_drop_p)
		
	def fit(self, X, Y, incremental=False):
		number_of_examples = X.shape[0]

		if self.initialize == True:
			self.initialize = False
			self.number_of_classes = int(np.max(Y) + 1)
			self.number_of_literals = X.shape[1]*2
			self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)
			self.number_of_patches = 1
			self.clause_banks = []
			for i in range(self.number_of_classes):
				self.clause_banks.append([ClauseBank(self.number_of_clauses//2, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.weighted_clauses), ClauseBank(self.number_of_clauses//2, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.weighted_clauses)])
		elif incremental == False:
			for i in range(self.number_of_classes):
				self.clause_banks[i][0].initialize_clauses()
				self.clause_banks[i][1].initialize_clauses()

		encoded_X = tmu.tools.encode(X, number_of_examples, self.number_of_ta_chunks, self.number_of_literals//2, 1, 1, self.number_of_literals//2, 1, 0)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)
		 
		for e in range(number_of_examples):
			target = Ym[e]

			class_sum = np.dot(self.clause_banks[target][0].get_clause_weights(), self.clause_banks[target][0].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
			class_sum -= np.dot(self.clause_banks[target][1].get_clause_weights(), self.clause_banks[target][1].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
					
			if class_sum > self.T:
				class_sum = self.T
			elif class_sum < -self.T:
				class_sum = -self.T

			update_p = (self.T - class_sum)/(2*self.T)

			self.clause_banks[target][0].cb_type_i_feedback(update_p, self.s, encoded_X[e,:], self.boost_true_positive_feedback)
			self.clause_banks[target][1].cb_type_ii_feedback(update_p, encoded_X[e,:])

			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			class_sum = np.dot(self.clause_banks[not_target][0].get_clause_weights(), self.clause_banks[not_target][0].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
			class_sum -= np.dot(self.clause_banks[not_target][1].get_clause_weights(), self.clause_banks[not_target][1].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
	
			if class_sum > self.T:
				class_sum = self.T
			elif class_sum < -self.T:
				class_sum = -self.T

			update_p = (self.T + class_sum)/(2*self.T)
		
			self.clause_banks[not_target][1].cb_type_i_feedback(update_p, self.s, encoded_X[e,:], self.boost_true_positive_feedback)
			self.clause_banks[not_target][0].cb_type_ii_feedback(update_p, encoded_X[e,:])			
		return

	def predict(self, X):
		number_of_examples = X.shape[0]
		
		encoded_X = tmu.tools.encode(X, number_of_examples, self.number_of_ta_chunks, self.number_of_literals//2, 1, 1, self.number_of_literals//2, 1, 0)	
		Y = np.ascontiguousarray(np.zeros(number_of_examples, dtype=np.uint32))
		for e in range(number_of_examples):
			max_class_sum = -self.T
			max_class = 0
			for i in range(self.number_of_classes):
				class_sum = np.dot(self.clause_banks[i][0].get_clause_weights(), self.clause_banks[i][0].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
				class_sum -= np.dot(self.clause_banks[i][1].get_clause_weights(), self.clause_banks[i][1].calculate_clause_outputs_update(encoded_X[e,:])).astype(np.int32)
				
				if class_sum > self.T:
					class_sum = self.T
				elif class_sum < -self.T:
					class_sum = -self.T

				if class_sum > max_class_sum:
					max_class_sum = class_sum
					max_class = i
			Y[e] = max_class
		return Y

	def ta_action(self, data_class, clause, ta):
		polarity = clause >= self.number_of_clauses//2
		clause_pos = clause % (self.number_of_clauses//2)
		return self.clause_banks[data_class][polarity].ta_action(clause_pos, ta)

	