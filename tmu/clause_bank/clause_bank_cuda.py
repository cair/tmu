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
import hashlib
import numpy as np
import logging
import pathlib
import tempfile

from tmu.util.cuda_profiler import CudaProfiler

current_dir = pathlib.Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

try:
    from pycuda._driver import Device, Context
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.curandom as curandom
    import pycuda.compiler as compiler

    cuda_installed = True
except Exception as e:
    _LOGGER.exception(e)
    _LOGGER.warning("Could not import pycuda. This indicates that it is not installed! A possible fix is to run 'pip "
                    "install pycuda'. Fallback to CPU ClauseBanks.")
    cuda_installed = False


def load_cuda_kernel(parameters, file: str):
    path = current_dir.joinpath(file)
    temp_directory = pathlib.Path(tempfile.gettempdir()).joinpath("tm_kernels")
    temp_directory.mkdir(exist_ok=True)

    with path.open("r") as f:
        data = f.read()

    source_code = '\n'.join(parameters) + data
    module_id = hashlib.sha1(source_code.encode()).hexdigest()
    ptx_file = temp_directory.joinpath(f"{module_id}.ptx")

    # If the module has been compiled, load it and return it
    if ptx_file.exists():
        _LOGGER.info(f"Loading compiled CUDA module from '{ptx_file}'.")
        module = cuda.module_from_file(str(ptx_file.absolute()))
        return module

    with BenchmarkTimer(_LOGGER, f"Compiling CUDA Module '{file}'."):
        ptx_code = pycuda.compiler.compile(source_code, no_extern_c=True)
        with open(ptx_file, 'wb') as f:
            f.write(ptx_code)
        module = cuda.module_from_file(str(ptx_file.absolute()))

    return module


class ImplClauseBankCUDA(BaseClauseBank):
    clause_bank_gpu = None
    clause_bank_synchronized: bool

    def __init__(
            self,
            seed: int,
            number_of_state_bits_ta: int,
            **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.grid = (16 * 13, 1, 1)
        self.block = (128, 1, 1)
        self.number_of_state_bits_ta = number_of_state_bits_ta

        self.clause_output_patchwise = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="C"
        )

        self.rng_gen = curandom.XORWOWRandomNumberGenerator()
        self.cuda_ctx: Context = pycuda.autoinit.context

        self._profiler: CudaProfiler = CudaProfiler()

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

        mod = load_cuda_kernel(parameters, "cuda/tools.cu")
        self.produce_autoencoder_examples_gpu = mod.get_function("produce_autoencoder_example")
        self.produce_autoencoder_examples_gpu.prepare("PPiPPiPPiPiii")

        # self.prepare_encode_gpu = mod.get_function("prepare_encode")
        # self.prepare_encode_gpu.prepare("PPiiiiiiii")

        # self.encode_gpu = mod.get_function("encode")
        # self.encode_gpu.prepare("PPiiiiiiii")

        self.clause_output = np.empty(
            int(self.number_of_clauses),
            dtype=np.uint32,
            order="c"
        )
        self.clause_output_gpu = self._profiler.profile(cuda.mem_alloc, self.clause_output.nbytes)

        self.clause_output_patchwise = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )

        self.clause_active_gpu = self._profiler.profile(cuda.mem_alloc, self.clause_output.nbytes)
        self.literal_active_gpu = self._profiler.profile(cuda.mem_alloc, self.number_of_ta_chunks * 4)

        self.literal_clause_count = np.empty(
            int(self.number_of_literals),
            dtype=np.uint32,
            order="c"
        )
        self.literal_clause_count_gpu = self._profiler.profile(cuda.mem_alloc, self.literal_clause_count.nbytes)

        self._cffi_init()

    def _cffi_init(self):
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
        self.clause_bank_gpu = self._profiler.profile(cuda.mem_alloc, self.clause_bank.nbytes)

        self._profiler.profile(cuda.memcpy_htod, self.clause_bank_gpu, self.clause_bank)
        self.clause_bank_synchronized = True

    def synchronize_clause_bank(self):
        if not self.clause_bank_synchronized:
            self._profiler.profile(cuda.memcpy_dtoh, self.clause_bank, self.clause_bank_gpu)
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
        self._profiler.profile(cuda.memcpy_dtoh, self.clause_output, self.clause_output_gpu)
        return self.clause_output

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        self._profiler.profile(cuda.memcpy_htod, self.literal_active_gpu, literal_active)

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
        self._profiler.profile(cuda.memcpy_dtoh, self.clause_output, self.clause_output_gpu)
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

    def type_i_feedback(
            self,
            update_p,
            clause_active,
            literal_active,
            encoded_X,
            e):
        self._profiler.profile(cuda.memcpy_htod, self.clause_active_gpu, clause_active)
        self._profiler.profile(cuda.memcpy_htod, self.literal_active_gpu, literal_active)

        self.type_i_feedback_gpu.prepared_call(
            self.grid,
            self.block,
            self.rng_gen.state,
            self.clause_bank_gpu,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            update_p,
            self.s,
            self.boost_true_positive_feedback,
            self.max_included_literals,
            self.clause_active_gpu,
            self.literal_active_gpu,
            encoded_X,
            np.int32(e)
        )

        self.cuda_ctx.synchronize()
        self.clause_bank_synchronized = False

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        self._profiler.profile(cuda.memcpy_htod, self.clause_active_gpu, np.ascontiguousarray(clause_active))
        self._profiler.profile(cuda.memcpy_htod, self.literal_active_gpu, literal_active)

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
        self._profiler.profile(cuda.memcpy_htod, self.clause_active_gpu, np.ascontiguousarray(clause_active))

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

        self._profiler.profile(cuda.memcpy_dtoh, self.literal_clause_count, self.literal_clause_count_gpu)

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
        self._profiler.profile(cuda.memcpy_htod, self.clause_bank_gpu, self.clause_bank)

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

        encoded_X_gpu = self._profiler.profile(cuda.mem_alloc, encoded_X.nbytes)
        self._profiler.profile(cuda.memcpy_htod, encoded_X_gpu, encoded_X)
        return encoded_X_gpu

    def prepare_X_autoencoder(self, X_csr, X_csc, active_output):
        _LOGGER.info("Copying sparse data to GPU memory")

        X_csr_indptr_gpu = self._profiler.profile(cuda.mem_alloc, X_csr.indptr.nbytes)
        self._profiler.profile(cuda.memcpy_htod, X_csr_indptr_gpu, X_csr.indptr)

        X_csr_indices_gpu = self._profiler.profile(cuda.mem_alloc, X_csr.indices.nbytes)
        self._profiler.profile(cuda.memcpy_htod, X_csr_indices_gpu, X_csr.indices)

        X_csc_indptr_gpu = self._profiler.profile(cuda.mem_alloc, X_csc.indptr.nbytes)
        self._profiler.profile(cuda.memcpy_htod, X_csc_indptr_gpu, X_csc.indptr)

        X_csc_indices_gpu = self._profiler.profile(cuda.mem_alloc, X_csc.indices.nbytes)
        self._profiler.profile(cuda.memcpy_htod, X_csc_indices_gpu, X_csc.indices)

        active_output_gpu = self._profiler.profile(cuda.mem_alloc, active_output.nbytes)
        self._profiler.profile(cuda.memcpy_htod, active_output_gpu, active_output)

        X = np.ascontiguousarray(np.zeros(int(self.number_of_ta_chunks), dtype=np.uint32))
        X_gpu = self._profiler.profile(cuda.mem_alloc, X.nbytes)

        return (
            active_output_gpu,
            active_output,
            int(active_output.shape[0]),
            X_csr_indptr_gpu,
            X_csr_indices_gpu,
            int(X_csr.shape[0]),
            X_csc_indptr_gpu,
            X_csc_indices_gpu,
            int(X_csc.shape[1]),
            X_gpu
        )

    _logged_unknown_args = set()

    def produce_autoencoder_example(
            self,
            encoded_X,
            target,
            accumulation,
            **kwargs
    ):
        # Log unknown arguments only once

        for key, value in kwargs.items():
            if key not in self._logged_unknown_args:
                self._logged_unknown_args.add(key)
                _LOGGER.error(
                    f"Unknown positional argument for {self}: argument_name={key}, argument_value={value}, class={type(self)}")

        (
            active_output_gpu,
            active_output,
            number_of_active_outputs,
            X_csr_indptr_gpu,
            X_csr_indices_gpu,
            number_of_rows,
            X_csc_indptr_gpu,
            X_csc_indices_gpu,
            number_of_columns,
            X_gpu
        ) = encoded_X

        target_value = self.rng.choice(2)

        self.produce_autoencoder_examples_gpu.prepared_call(
            self.grid,
            self.block,
            self.rng_gen.state,
            active_output_gpu,
            number_of_active_outputs,
            X_csr_indptr_gpu,
            X_csr_indices_gpu,
            number_of_rows,
            X_csc_indptr_gpu,
            X_csc_indices_gpu,
            number_of_columns,
            X_gpu,
            int(target),
            int(target_value),
            int(accumulation))

        self.cuda_ctx.synchronize()

        return X_gpu, target_value


if cuda_installed:
    ClauseBankCUDA = ImplClauseBankCUDA
else:
    ClauseBankCUDA = ClauseBank
