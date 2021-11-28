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

import numpy as np

class ClauseBank():
	def __init__(self, number_of_clauses, number_of_literals, number_of_state_bits, number_of_patches, weighted_clauses):
		self.number_of_clauses = int(number_of_clauses)
		self.number_of_literals = int(number_of_literals)
		self.number_of_state_bits = int(number_of_state_bits)
		self.number_of_patches = int(number_of_patches)
		self.weighted_clauses = int(weighted_clauses)

		self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)

		self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
		self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)

		self.clause_weights = np.ascontiguousarray(np.empty(self.number_of_clauses, dtype=np.int32))
		self.cw_p = ffi.cast("int *", self.clause_weights.ctypes.data)

		self.feedback_to_ta = np.ascontiguousarray(np.empty((self.number_of_ta_chunks), dtype=np.uint32))
		self.ft_p = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)

		self.output_one_patches = np.ascontiguousarray(np.empty(self.number_of_patches, dtype=np.uint32))
		self.o1p_p = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)

		self.initialize_clauses()

	def initialize_clauses(self):
		self.clause_bank = np.empty((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits), dtype=np.uint32)
		self.clause_bank[:,:,0:self.number_of_state_bits-1] = np.uint32(~0)
		self.clause_bank[:,:,self.number_of_state_bits-1] = 0
		self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape((self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits)))
		self.clause_weights[:] = 1
		self.cb_p = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)

	def calculate_clause_outputs_predict(self, Xi):
		xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
		lib.cb_calculate_clause_outputs_predict(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.co_p, xi_p)
		return self.clause_output

	def calculate_clause_outputs_update(self, Xi):
		xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
		lib.cb_calculate_clause_outputs_update(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.co_p, xi_p)
		return self.clause_output

	def cb_type_i_feedback(self, update_p, s, boost_true_positive_feedback, clause_active, Xi):
		xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
		ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
		lib.cb_type_i_feedback(self.cb_p, self.cw_p, self.ft_p, self.o1p_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, update_p, s, self.weighted_clauses, boost_true_positive_feedback, ca_p, xi_p)

	def cb_type_ii_feedback(self, update_p, clause_active, Xi):
		xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
		ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
		lib.cb_type_ii_feedback(self.cb_p, self.cw_p, self.o1p_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, update_p, self.weighted_clauses, ca_p, xi_p)

	def get_clause_weights(self):
		return self.clause_weights

	def ta_action(self, clause, ta):
		ta_chunk = ta // 32
		chunk_pos = ta % 32

		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits + self.number_of_state_bits-1)

		return (self.clause_bank[pos] & (1 << chunk_pos)) > 0
