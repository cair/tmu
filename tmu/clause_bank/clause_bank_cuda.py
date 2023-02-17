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

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688
import tmu.tools
from tmu.tmulib import ffi, lib
from tmu.tools import BenchmarkTimer
from tmu.clause_bank.clause_bank import ClauseBank
from tmu.clause_bank.base_clause_bank import BaseClauseBank

import numpy as np
import logging
import pathlib
import tempfile

current_dir = pathlib.Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

try:
    from pycuda._driver import Device, Context
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.curandom as curandom
    import pycuda.compiler as compiler
    cuda_installed = True
except ImportError as e:
    _LOGGER.warning("Could not import pycuda. This indicates that it is not installed! A possible fix is to run 'pip "
                    "install pycuda'. Fallback to CPU ClauseBanks.")
    cuda_installed = False


def load_cuda_kernel(parameters, file: str):
    path = current_dir.joinpath(file)
    temp_directory = pathlib.Path(tempfile.gettempdir()).joinpath("tm_kernels")
    temp_directory.mkdir(exist_ok=True)

    with path.open("r") as f:
        data = f.read()

    with BenchmarkTimer(_LOGGER, f"Compiled CUDA Module '{file}'."):
        return compiler.SourceModule(
            '\n'.join(parameters) + data,
            no_extern_c=True,
            keep=True,
            cache_dir=temp_directory
        )


class ImplClauseBankCUDA(BaseClauseBank):
    clause_bank_gpu = None
    clause_bank_synchronized: bool

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.grid = (16 * 13, 1, 1)
        self.block = (128, 1, 1)

        self.clause_output_patchwise = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="C"
        )

        self.rng_gen = curandom.XORWOWRandomNumberGenerator()
        self.cuda_ctx: Context = pycuda.autoinit.context

        parameters = [
            f"#define NUMBER_OF_PATCHES {self.number_of_patches}"
        ]

        mod = load_cuda_kernel(parameters, "cuda/calculate_clause_outputs_predict.cu")
        self.calculate_clause_outputs_predict_gpu = mod.get_function("calculate_clause_outputs_predict")
        self.calculate_clause_outputs_predict_gpu.prepare("PiiiPPi")
        self.calculate_literal_frequency_gpu = mod.get_function("calculate_literal_frequency")
        self.calculate_literal_frequency_gpu.prepare("PiiiPP")

        mod = load_cuda_kernel(parameters, "cuda/calculate_clause_outputs_update.cu")
        self.calculate_clause_outputs_update_gpu = mod.get_function("calculate_clause_outputs_update")
        self.calculate_clause_outputs_update_gpu.prepare("PiiiPPPi")

        mod = load_cuda_kernel(parameters, "cuda/clause_feedback.cu")
        self.type_i_feedback_gpu = mod.get_function("type_i_feedback")
        self.type_i_feedback_gpu.prepare("PPiiiffiiPPPi")
        self.type_ii_feedback_gpu = mod.get_function("type_ii_feedback")
        self.type_ii_feedback_gpu.prepare("PPiiifPPPi")

        self.clause_output = np.empty(
            int(self.number_of_clauses),
            dtype=np.uint32,
            order="c"
        )
        self.clause_output_gpu = cuda.mem_alloc(self.clause_output.nbytes)

        self.clause_output_patchwise = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )

        self.clause_active_gpu = cuda.mem_alloc(self.clause_output.nbytes)
        self.literal_active_gpu = cuda.mem_alloc(self.number_of_ta_chunks * 4)

        self.literal_clause_count = np.empty(
            int(self.number_of_literals),
            dtype=np.uint32,
            order="c"
        )
        self.literal_clause_count_gpu = cuda.mem_alloc(self.literal_clause_count.nbytes)

        self.initialize_clauses()

    def _cffi_init(self):
        pass

    def initialize_clauses(self):
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="c"
        )
        self.clause_bank[:, :, 0:self.number_of_state_bits_ta - 1] = np.uint32(~0)
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta),
            order="c"
        )
        self.clause_bank_gpu = cuda.mem_alloc(self.clause_bank.nbytes)

        cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)
        self.clause_bank_synchronized = True

    def synchronize_clause_bank(self):
        if not self.clause_bank_synchronized:
            cuda.memcpy_dtoh(self.clause_bank, self.clause_bank_gpu)
            self.clause_bank_synchronized = True

    def calculate_clause_outputs_predict(self, encoded_X, e):
        self.calculate_clause_outputs_predict_gpu.prepared_call(
            self.grid,
            self.block,
            self.clause_bank_gpu,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.clause_output_gpu,
            encoded_X,
            np.int32(e)
        )
        self.cuda_ctx.synchronize()
        cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
        return self.clause_output

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        cuda.memcpy_htod(self.literal_active_gpu, literal_active)

        self.calculate_clause_outputs_update_gpu.prepared_call(
            self.grid,
            self.block,
            self.clause_bank_gpu,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.clause_output_gpu,
            self.literal_active_gpu,
            encoded_X,
            np.int32(e)
        )

        self.cuda_ctx.synchronize()
        cuda.memcpy_dtoh(self.clause_output, self.clause_output_gpu)
        return self.clause_output

    def calculate_clause_outputs_patchwise(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", Xi.ctypes.data)
        lib.cb_calculate_clause_outputs_patchwise(
            self.cb_p,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.number_of_patches,
            self.cop_p,
            xi_p
        )
        return self.clause_output_patchwise

    def type_i_feedback(self, update_p, s, boost_true_positive_feedback, max_included_literals, clause_active,
                        literal_active, encoded_X, e):
        cuda.memcpy_htod(self.clause_active_gpu, clause_active)
        cuda.memcpy_htod(self.literal_active_gpu, literal_active)

        self.type_i_feedback_gpu.prepared_call(
            self.grid,
            self.block,
            self.rng_gen.state,
            self.clause_bank_gpu,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            update_p,
            s,
            boost_true_positive_feedback,
            max_included_literals,
            self.clause_active_gpu,
            self.literal_active_gpu,
            encoded_X,
            np.int32(e)
        )

        self.cuda_ctx.synchronize()
        self.clause_bank_synchronized = False

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        cuda.memcpy_htod(self.clause_active_gpu, np.ascontiguousarray(clause_active))
        cuda.memcpy_htod(self.literal_active_gpu, literal_active)

        self.type_ii_feedback_gpu.prepared_call(
            self.grid,
            self.block,
            self.rng_gen.state,
            self.clause_bank_gpu,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            update_p,
            self.clause_active_gpu,
            self.literal_active_gpu,
            encoded_X,
            np.int32(e)
        )

        self.cuda_ctx.synchronize()
        self.clause_bank_synchronized = False

    def calculate_literal_clause_frequency(self, clause_active):
        cuda.memcpy_htod(self.clause_active_gpu, np.ascontiguousarray(clause_active))

        self.calculate_literal_frequency_gpu.prepared_call(
            self.grid,
            self.block,
            self.clause_bank_gpu,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.clause_active_gpu,
            self.literal_clause_count_gpu
        )

        self.cuda_ctx.synchronize()

        cuda.memcpy_dtoh(self.literal_clause_count, self.literal_clause_count_gpu)

        return self.literal_clause_count

    def get_ta_action(self, clause, ta):
        self.synchronize_clause_bank()
        return super().get_ta_action(clause, ta)

    def number_of_include_actions(self, clause):
        self.synchronize_clause_bank()
        start = int(clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + self.number_of_state_bits_ta - 1)
        stop = int(
            (clause + 1) * self.number_of_ta_chunks * self.number_of_state_bits_ta + self.number_of_state_bits_ta - 1)
        return np.unpackbits(
            np.ascontiguousarray(self.clause_bank[start:stop:self.number_of_state_bits_ta]).view('uint8')).sum()

    def get_ta_state(self, clause, ta):
        self.synchronize_clause_bank()
        return super().get_ta_state(clause, ta)

    def set_ta_state(self, clause, ta, state):
        self.synchronize_clause_bank()
        super().set_ta_state(clause, ta, state)
        cuda.memcpy_htod(self.clause_bank_gpu, self.clause_bank)

    def prepare_X(self, X):
        encoded_X = tmu.tools.encode(
            X,
            X.shape[0],
            self.number_of_patches,
            self.number_of_ta_chunks,
            self.dim,
            self.patch_dim,
            0
        )

        encoded_X_gpu = cuda.mem_alloc(encoded_X.nbytes)
        cuda.memcpy_htod(encoded_X_gpu, encoded_X)
        return encoded_X_gpu


if cuda_installed:
    ClauseBankCUDA = ImplClauseBankCUDA
else:
    ClauseBankCUDA = ClauseBank
