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

from ._cb import ffi, lib

import numpy as np

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import tmu.tools

g = curandom.XORWOWRandomNumberGenerator() 

import tmu.clause_bank_cuda_kernels as kernels

class ClauseBankCUDA():
	def __init__(self, X, number_of_clauses, number_of_state_bits, patch_dim):
		self.grid = (16*13,1,1)
		self.block = (128,1,1)

		self.number_of_clauses = int(number_of_clauses)
		self.number_of_state_bits = int(number_of_state_bits)
		self.patch_dim = patch_dim

		if len(X.shape) == 2:
			self.dim = (X.shape[1], 1, 1)
		elif len(X.shape) == 3:
			self.dim = (X.shape[1], X.shape[2], 1)
		elif len(X.shape) == 4:
			self.dim = (X.shape[1], X.shape[2], X.shape[3])

		if self.patch_dim == None:
			self.patch_dim = (self.dim[0]*self.dim[1]*self.dim[2], 1)

		self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))
		self.number_of_literals = self.number_of_features*2
		
		self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1)*(self.dim[1] - self.patch_dim[1] + 1))
		self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)

		self.clause_output_patchwise = np.ascontiguousarray(np.empty((int(self.number_of_clauses*self.number_of_patches)), dtype=np.uint32))

		parameters = """
			#define NUMBER_OF_PATCHES %d
			""" % (self.number_of_patches)

		mod = SourceModule(parameters + kernels.code_calculate_clause_outputs_predict, no_extern_c=True)
		self.calculate_clause_outputs_predict_gpu = mod.get_function("calculate_clause_outputs_predict")
		self.calculate_clause_outputs_predict_gpu.prepare("PiiiPPi")
		self.calculate_literal_frequency_gpu = mod.get_function("calculate_literal_frequency")
		self.calculate_literal_frequency_gpu.prepare("PiiiPP")

		mod = SourceModule(parameters + kernels.code_calculate_clause_outputs_update, no_extern_c=True)
		self.calculate_clause_outputs_update_gpu = mod.get_function("calculate_clause_outputs_update")
		self.calculate_clause_outputs_update_gpu.prepare("PiiiPPPi")

		mod = SourceModule(parameters + kernels.code_clause_feedback, no_extern_c=True)
		self.type_i_feedback_gpu = mod.get_function("type_i_feedback")
		self.type_i_feedback_gpu.prepare("PPiiiffiiPPPi")
		self.type_ii_feedback_gpu = mod.get_function("type_ii_feedback")
		self.type_ii_feedback_gpu.prepare("PPiiifPPPi")

		self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
		self.clause_output_gpu = cuda.mem_alloc(self.clause_output.nbytes)

		self.clause_output_patchwise = np.ascontiguousarray(np.empty((int(self.number_of_clauses*self.number_of_patches)), dtype=np.uint32))

		self.clause_active_gpu = cuda.mem_alloc(self.clause_output.nbytes)
		self.literal_active_gpu = cuda.mem_alloc(self.number_of_ta_chunks*4)

		self.literal_clause_count = np.ascontiguousarray(np.empty((int(self.number_of_literals)), dtype=np.uint32))
		self.literal_clause_count_gpu = cuda.mem_alloc(self.literal_clause_count.nbytes)

		self.initialize_clauses()

	def initialize_clauses(self):
		self.clause_bank = np.empty((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits), dtype=np.uint32)
		self.clause_bank[:,:,0:self.number_of_state_bits-1] = np.uint32(~0)
		self.clause_bank[:,:,self.number_of_state_bits-1] = 0
		self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape((self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits)))
		self.clause_bank_gpu = cuda.mem_alloc(self.clause_bank.nbytes)
		cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
		self.clause_bank_synchronized = True

	def synchronize_clause_bank(self):
		if not self.clause_bank_synchronized:
			cuda.memcpy_dtoh(self.clause_bank, self.clause_bank_gpu)
			self.clause_bank_synchronized = True

	def calculate_clause_outputs_predict(self, encoded_X, e):
		self.calculate_clause_outputs_predict_gpu.prepared_call(self.grid, self.block, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.clause_output_gpu, encoded_X, np.int32(e))
		cuda.Context.synchronize()
		cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
		return self.clause_output

	def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
		cuda.memcpy_htod(self.literal_active_gpu, literal_active)
		self.calculate_clause_outputs_update_gpu.prepared_call(self.grid, self.block, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.clause_output_gpu, self.literal_active_gpu, encoded_X, np.int32(e))
		cuda.Context.synchronize()
		cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
		return self.clause_output

	def calculate_clause_outputs_patchwise(self, encoded_X, e):
		xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
		lib.cb_calculate_clause_outputs_patchwise(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.cop_p, xi_p)
		return self.clause_output_patchwise

	def type_i_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active, literal_active, encoded_X, e):
		cuda.memcpy_htod(self.clause_active_gpu, clause_active)
		cuda.memcpy_htod(self.literal_active_gpu, literal_active)
		self.type_i_feedback_gpu.prepared_call(self.grid, self.block, g.state, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, update_p, s, boost_true_positive_feedback, max_included_literals, self.clause_active_gpu, self.literal_active_gpu, encoded_X, np.int32(e))
		cuda.Context.synchronize()
		self.clause_bank_synchronized = False

	def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
		cuda.memcpy_htod(self.clause_active_gpu, np.ascontiguousarray(clause_active))
		cuda.memcpy_htod(self.literal_active_gpu, literal_active)
		self.type_ii_feedback_gpu.prepared_call(self.grid, self.block, g.state, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, update_p, self.clause_active_gpu, self.literal_active_gpu, encoded_X, np.int32(e))
		cuda.Context.synchronize()
		self.clause_bank_synchronized = False

	def calculate_literal_clause_frequency(self, clause_active):
		cuda.memcpy_htod(self.clause_active_gpu, np.ascontiguousarray(clause_active))
		self.calculate_literal_frequency_gpu.prepared_call(self.grid, self.block, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.clause_active_gpu, self.literal_clause_count_gpu)
		cuda.Context.synchronize()
		cuda.memcpy_dtoh(self.literal_clause_count, self.literal_clause_count_gpu)
		return self.literal_clause_count

	def get_ta_action(self, clause, ta):
		self.synchronize_clause_bank()
		ta_chunk = ta // 32
		chunk_pos = ta % 32
		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits + self.number_of_state_bits-1)
		return (self.clause_bank[pos] & (1 << chunk_pos)) > 0

	def number_of_include_actions(self, clause):
		self.synchronize_clause_bank()
		start = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + self.number_of_state_bits-1)
		stop = int((clause + 1) * self.number_of_ta_chunks * self.number_of_state_bits + self.number_of_state_bits-1)
		return np.unpackbits(np.ascontiguousarray(self.clause_bank[start:stop:self.number_of_state_bits]).view('uint8')).sum()

	def get_ta_state(self, clause, ta):
		self.synchronize_clause_bank()
		ta_chunk = ta // 32
		chunk_pos = ta % 32
		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits)
		state = 0
		for b in range(self.number_of_state_bits):
			if self.clause_bank[pos + b] & (1 << chunk_pos) > 0:
				state |= (1 << b)
		return state

	def set_ta_state(self, clause, ta, state):
		self.synchronize_clause_bank()
		ta_chunk = ta // 32
		chunk_pos = ta % 32
		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits)
		for b in range(self.number_of_state_bits):
			if state & (1 << b) > 0:
				self.clause_bank[pos + b] |= (1 << chunk_pos)
			else:
				self.clause_bank[pos + b] &= ~(1 << chunk_pos)
		cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)

	def prepare_X(self, X):
		encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim, 0)
		encoded_X_gpu = cuda.mem_alloc(encoded_X.nbytes)
		cuda.memcpy_htod(encoded_X_gpu, encoded_X)
		return encoded_X_gpu
