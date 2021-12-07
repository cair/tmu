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

from ._cb import ffi, lib

import numpy as np

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

g = curandom.XORWOWRandomNumberGenerator() 

import tmu.clause_bank_cuda_kernels as kernels

class ClauseBankCUDA():
	def __init__(self, number_of_clauses, number_of_literals, number_of_state_bits, number_of_patches, X, Y):
		self.grid = (16*13,1,1)
		self.block = (128,1,1)

		print("Platform: CUDA")

		self.number_of_clauses = int(number_of_clauses)
		self.number_of_literals = int(number_of_literals)
		self.number_of_state_bits = int(number_of_state_bits)
		self.number_of_patches = int(number_of_patches)

		print("Number of patches:", self.number_of_patches)

		self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)

		self.clause_output_patchwise = np.ascontiguousarray(np.empty((int(self.number_of_clauses*self.number_of_patches)), dtype=np.uint32))

		mod = SourceModule(kernels.code_calculate_clause_outputs_predict, no_extern_c=True)
		self.calculate_clause_outputs_predict_gpu = mod.get_function("calculate_clause_outputs_predict")
		self.calculate_clause_outputs_predict_gpu.prepare("PiiiiPPi")

		mod = SourceModule(kernels.code_calculate_clause_outputs_update, no_extern_c=True)
		self.calculate_clause_outputs_update_gpu = mod.get_function("calculate_clause_outputs_update")
		self.calculate_clause_outputs_update_gpu.prepare("PiiiiPPi")

		mod = SourceModule(kernels.code_clause_feedback, no_extern_c=True)
		self.type_i_feedback_gpu = mod.get_function("type_i_feedback")
		self.type_i_feedback_gpu.prepare("PPPiiiiffiPPi")
		self.type_ii_feedback_gpu = mod.get_function("type_ii_feedback")
		self.type_ii_feedback_gpu.prepare("PPPiiiifPPPPiP")

		self.clause_output = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
		self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)
		self.clause_output_gpu = cuda.mem_alloc(self.clause_output.nbytes*4)

		self.clause_patch = np.ascontiguousarray(np.empty((int(self.number_of_clauses)), dtype=np.uint32))
		self.cp_p = ffi.cast("unsigned int *", self.clause_patch.ctypes.data)
		self.clause_patch_gpu = cuda.mem_alloc(self.clause_patch.nbytes*4)

		self.clause_output_patchwise = np.ascontiguousarray(np.empty((int(self.number_of_clauses*self.number_of_patches)), dtype=np.uint32))
		self.cop_p = ffi.cast("unsigned int *", self.clause_output_patchwise.ctypes.data)

		self.feedback_to_ta = np.ascontiguousarray(np.empty((self.number_of_ta_chunks), dtype=np.uint32))
		self.ft_p = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)

		self.output_one_patches = np.ascontiguousarray(np.empty(self.number_of_patches, dtype=np.uint32))
		self.o1p_p = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)
		self.output_one_patches_gpu = cuda.mem_alloc(self.output_one_patches.nbytes*4)

		self.clause_active_gpu = cuda.mem_alloc(self.clause_output.nbytes*4)

		self.random_integers_gpu = cuda.mem_alloc(self.number_of_clauses*4*4)

		self.initialize_clauses()

	def initialize_clauses(self):
		self.clause_bank = np.empty((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits), dtype=np.uint32)
		self.clause_bank[:,:,0:self.number_of_state_bits-1] = np.uint32(~0)
		self.clause_bank[:,:,self.number_of_state_bits-1] = 0
		self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape((self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits)))
		self.clause_bank_gpu = cuda.mem_alloc(self.clause_bank.nbytes*4)
		cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
		self.cb_p = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)

	def calculate_clause_outputs_predict_cpu(self, e):
		xi_p = ffi.cast("unsigned int *", self.encoded_X[e,:].ctypes.data)
		lib.cb_calculate_clause_outputs_predict(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.co_p, xi_p)

		#cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
		#self.calculate_clause_outputs_predict_gpu.prepared_call(self.grid, self.block, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.clause_output_gpu, self.encoded_X_gpu, np.int32(e))
		#cuda.Context.synchronize()
		#cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
		return self.clause_output

	def calculate_clause_outputs_predict(self, e):
		#xi_p = ffi.cast("unsigned int *", self.encoded_X[e,:].ctypes.data)
		#lib.cb_calculate_clause_outputs_predict(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.co_p, xi_p)

		cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
		self.calculate_clause_outputs_predict_gpu.prepared_call(self.grid, self.block, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.clause_output_gpu, self.encoded_X_gpu, np.int32(e))
		cuda.Context.synchronize()
		cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
		return self.clause_output

	def calculate_clause_outputs_update(self, e):
		xi_p = ffi.cast("unsigned int *", self.encoded_X[e,:].ctypes.data)
		lib.cb_calculate_clause_outputs_update(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.co_p, xi_p)
		
		#self.calculate_clause_outputs_update_gpu.prepared_call(self.grid, self.block, self.clause_bank_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.clause_output_gpu, self.encoded_X_gpu, np.int32(e))
		#cuda.Context.synchronize()
		#cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
		return self.clause_output

	def calculate_clause_outputs_patchwise(self, Xi):
		xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
		lib.cb_calculate_clause_outputs_patchwise(self.cb_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.cop_p, xi_p)
		return self.clause_output_patchwise

	def type_i_feedback(self, update_p, s, boost_true_positive_feedback, clause_active, e):
		xi_p = ffi.cast("unsigned int *", self.encoded_X[e,:].ctypes.data)
		ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
		lib.cb_type_i_feedback(self.cb_p, self.ft_p, self.o1p_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, update_p, s, boost_true_positive_feedback, ca_p, xi_p)

		#cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
		#cuda.memcpy_htod(self.clause_active_gpu, clause_active)
		#self.type_i_feedback_gpu.prepared_call(self.grid, self.block, g.state, self.clause_bank_gpu, self.output_one_patches_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, update_p, s, boost_true_positive_feedback, self.clause_active_gpu, self.encoded_X_gpu, np.int32(e))
		#cuda.Context.synchronize()
		#cuda.memcpy_dtoh(self.clause_bank, self.clause_bank_gpu)

	def type_ii_feedback(self, update_p, clause_active, e):
		#xi_p = ffi.cast("unsigned int *", self.encoded_X[e,:].ctypes.data)
		#ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
		#lib.cb_type_ii_feedback(self.cb_p, self.o1p_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, update_p, ca_p, xi_p)

		random_integers = np.random.randint(4294967295, size=self.number_of_clauses, dtype=np.uint32)

		ri_p = ffi.cast("unsigned int *", random_integers.ctypes.data)
		xi_p = ffi.cast("unsigned int *", self.encoded_X[e,:].ctypes.data)
		lib.cb_clause_outputs_patches(self.cb_p, self.o1p_p, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, self.co_p, self.cp_p, xi_p, ri_p)

		cuda.memcpy_htod(self.clause_output_gpu, self.clause_output)
		cuda.memcpy_htod(self.clause_patch_gpu, self.clause_patch)
		cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
		cuda.memcpy_htod(self.clause_active_gpu, np.ascontiguousarray(clause_active))
		cuda.memcpy_htod(self.random_integers_gpu, random_integers)
		self.type_ii_feedback_gpu.prepared_call(self.grid, self.block, g.state, self.clause_bank_gpu, self.output_one_patches_gpu, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.number_of_patches, update_p, self.clause_active_gpu, self.clause_output_gpu, self.clause_patch_gpu, self.encoded_X_gpu, np.int32(e), self.random_integers_gpu)
		cuda.Context.synchronize()
		cuda.memcpy_dtoh(self.clause_bank, self.clause_bank_gpu)

	def get_ta_action(self, clause, ta):
		ta_chunk = ta // 32
		chunk_pos = ta % 32
		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits + self.number_of_state_bits-1)
		return (self.clause_bank[pos] & (1 << chunk_pos)) > 0

	def get_ta_state(self, clause, ta):
		ta_chunk = ta // 32
		chunk_pos = ta % 32
		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits)
		state = 0
		for b in range(self.number_of_state_bits):
			if self.clause_bank[pos + b] & (1 << chunk_pos) > 0:
				state |= (1 << b)
		return state

	def set_ta_state(self, clause, ta, state):
		ta_chunk = ta // 32
		chunk_pos = ta % 32
		pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits + ta_chunk * self.number_of_state_bits)
		for b in range(self.number_of_state_bits):
			if state & (1 << b) > 0:
				self.clause_bank[pos + b] |= (1 << chunk_pos)
			else:
				self.clause_bank[pos + b] &= (1 << chunk_pos)

	def copy_X(self, encoded_X):
		self.encoded_X = encoded_X
		self.encoded_X_gpu = cuda.mem_alloc(encoded_X.nbytes*4)
		cuda.memcpy_htod(self.encoded_X_gpu, encoded_X)
