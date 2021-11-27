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

from ._tm import ffi, lib
import tmu.tools

import numpy as np

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

	def ta_action(self, data_class, clause, ta):
		ta_chunk = ta // 32
		chunk_pos = ta % 32

		polarity = clause >= self.number_of_clauses//2
		clause_pos = clause % (self.number_of_clauses//2)

		pos = int(clause_pos * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits + self.number_of_state_bits-1)

		return (self.clause_banks[data_class, int(polarity), pos] & (1 << chunk_pos)) > 0


class TMClassifier(TMBase):
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, clause_drop_p = 0.0, literal_drop_p = 0.0):
		super().__init__(number_of_clauses, T, s, boost_true_positive_feedback, number_of_state_bits, weighted_clauses, clause_drop_p, literal_drop_p)
		
	def fit(self, X, Y, incremental=False):
		number_of_examples = X.shape[0]

		if self.initialize == True:
			self.initialize = False

			self.number_of_classes = int(np.max(Y) + 1)

			self.number_of_features = X.shape[1]*2

			self.number_of_patches = 1
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.clause_banks = np.empty((int(self.number_of_classes), int(self.number_of_clauses), int(self.number_of_ta_chunks), int(self.number_of_state_bits)), dtype=np.uint32)
			self.clause_banks[:,:,:,0:self.number_of_state_bits-1] = np.uint32(~0)
			self.clause_banks[:,:,:,self.number_of_state_bits-1] = 0

			self.clause_banks = np.ascontiguousarray(self.clause_banks.reshape(int(self.number_of_classes), 2, int(self.number_of_clauses // 2 * self.number_of_ta_chunks * self.number_of_state_bits)))
			self.clause_weights = np.ascontiguousarray(np.empty((int(self.number_of_classes), int(self.number_of_clauses)), dtype=np.int32))
			self.feedback_to_ta = np.ascontiguousarray(np.empty((int(self.number_of_ta_chunks)), dtype=np.uint32))
			self.output_one_patches = np.ascontiguousarray(np.empty((int(1)), dtype=np.uint32))
			self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
		elif incremental == False:
			self.clause_banks = np.empty((int(self.number_of_classes), int(self.number_of_clauses), int(self.number_of_ta_chunks), int(self.number_of_state_bits)), dtype=np.uint32)
			self.clause_banks[:,:,:,0:self.number_of_state_bits-1] = np.uint32(~0)
			self.clause_banks[:,:,:,self.number_of_state_bits-1] = 0

			self.clause_banks = np.ascontiguousarray(self.clause_banks.reshape(int(self.number_of_classes), 2, int(self.number_of_clauses // 2 * self.number_of_ta_chunks * self.number_of_state_bits)))
			self.clause_weights = np.ascontiguousarray(np.empty((int(self.number_of_classes), int(self.number_of_clauses)), dtype=np.int32))

		encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		tmu.tools.encode(Xm, encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 0)
		
		encoded_X = np.ascontiguousarray(encoded_X.reshape((int(number_of_examples), self.number_of_ta_chunks)))

		co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)
		o1p_p = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)
		ft_p = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)

		for e in range(number_of_examples):
			target = Ym[e]

			xi_p = ffi.cast("unsigned int *", encoded_X[e,:].ctypes.data)
			
			cb_p_1 = ffi.cast("unsigned int *", self.clause_banks[target,0,:].ctypes.data)
			lib.cb_calculate_clause_outputs_update(cb_p_1, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, co_p, xi_p)
			class_sum = self.clause_output.sum().astype(np.int32)
			cb_p_2 = ffi.cast("unsigned int *", self.clause_banks[target,1,:].ctypes.data)
			lib.cb_calculate_clause_outputs_update(cb_p_2, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, co_p, xi_p)
			class_sum -= self.clause_output.sum().astype(np.int32)
					
			if class_sum > self.T:
				class_sum = self.T
			elif class_sum < -self.T:
				class_sum = -self.T

			update_p = (self.T - class_sum)/(2*self.T)

			cw_1_p= ffi.cast("int *", self.clause_weights[target,:self.number_of_clauses//2].ctypes.data)
			lib.cb_type_i_feedback(cb_p_1, cw_1_p, ft_p, o1p_p, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, update_p, self.s, self.weighted_clauses, self.boost_true_positive_feedback, xi_p)
			cw_2_p= ffi.cast("int *", self.clause_weights[target,self.number_of_clauses//2:].ctypes.data)
			lib.cb_type_ii_feedback(cb_p_2, cw_2_p, o1p_p, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, update_p, self.weighted_clauses, xi_p)
			
			not_target = np.random.randint(self.number_of_classes)
			while not_target == target:
				not_target = np.random.randint(self.number_of_classes)

			cb_p_1 = ffi.cast("unsigned int *", self.clause_banks[not_target,0,:].ctypes.data)
			lib.cb_calculate_clause_outputs_update(cb_p_1, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, co_p, xi_p)
			class_sum = self.clause_output.sum().astype(np.int32)
			cb_p_2 = ffi.cast("unsigned int *", self.clause_banks[not_target,1,:].ctypes.data)
			lib.cb_calculate_clause_outputs_update(cb_p_2, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, co_p, xi_p)
			class_sum -= self.clause_output.sum().astype(np.int32)
				
			if class_sum > self.T:
				class_sum = self.T
			elif class_sum < -self.T:
				class_sum = -self.T

			update_p = (self.T + class_sum)/(2*self.T)
		
			cw_1_p= ffi.cast("int *", self.clause_weights[not_target,:self.number_of_clauses//2].ctypes.data)
			lib.cb_type_i_feedback(cb_p_2, cw_2_p, ft_p, o1p_p, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, update_p, self.s, self.weighted_clauses, self.boost_true_positive_feedback, xi_p)
			cw_2_p= ffi.cast("int *", self.clause_weights[not_target,self.number_of_clauses//2:].ctypes.data)
			lib.cb_type_ii_feedback(cb_p_1, cw_1_p, o1p_p, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, update_p, self.weighted_clauses, xi_p)
			
		return

	def predict(self, X):
		number_of_examples = X.shape[0]
		
		encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		tmu.tools.encode(Xm, encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 0)

		encoded_X = np.ascontiguousarray(encoded_X.reshape((int(number_of_examples), self.number_of_ta_chunks)))

		Y = np.ascontiguousarray(np.zeros(number_of_examples, dtype=np.uint32))

		co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)

		for e in range(number_of_examples):
			xi_p = ffi.cast("unsigned int *", encoded_X[e,:].ctypes.data)

			max_class_sum = -self.T
			max_class = 0
			for i in range(self.number_of_classes):
				cb_p_1 = ffi.cast("unsigned int *", self.clause_banks[i,0,:].ctypes.data)
				lib.cb_calculate_clause_outputs_predict(cb_p_1, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, co_p, xi_p)
				class_sum = self.clause_output.sum().astype(np.int32)
				
				cb_p_2 = ffi.cast("unsigned int *", self.clause_banks[i,1,:].ctypes.data)
				lib.cb_calculate_clause_outputs_predict(cb_p_2, self.number_of_clauses//2, self.number_of_features, self.number_of_state_bits, self.number_of_patches, co_p, xi_p)
				class_sum -= self.clause_output.sum().astype(np.int32)
				
				if class_sum > self.T:
					class_sum = self.T
				elif class_sum < -self.T:
					class_sum = -self.T

				if class_sum > max_class_sum:
					max_class_sum = class_sum
					max_class = i
			Y[e] = max_class
		return Y