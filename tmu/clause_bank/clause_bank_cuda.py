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
    import pycuda.driver as cuda
    cuda_installed = True
except Exception as e:
    _LOGGER.exception(e)
    _LOGGER.warning("Could not import pycuda. This indicates that it is not installed! A possible fix is to run 'pip "
                    "install pycuda'. Fallback to CPU ClauseBanks.")
    cuda_installed = False

def load_cuda_kernel(parameters, file: str):
    import pycuda.compiler
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


# Host branch: handles numpy based data and CPU operations
class ClauseBankCudaHost(BaseClauseBank):
    def __init__(self, seed: int, number_of_state_bits_ta: int, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.number_of_state_bits_ta = number_of_state_bits_ta

        # Initialize host arrays for clause outputs and literal counts
        self.clause_output = np.empty(
            int(self.number_of_clauses),
            dtype=np.uint32,
            order="C"
        )

        self.clause_output_patchwise = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="C"
        )

        self.literal_clause_count = np.empty(
            int(self.number_of_literals),
            dtype=np.uint32,
            order="C"
        )

        self._initialize_clause_bank()

    def _initialize_clause_bank(self):
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="C"
        )

        self.clause_bank[:, :, 0:self.number_of_state_bits_ta - 1] = np.uint32(4294967295)  #np.uint32(~0)

        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta),
            order="C"
        )

    def get_ta_action(self, clause, ta):
        return super().get_ta_action(clause, ta)

    def number_of_include_actions(self, clause):
        start = int(clause * self.number_of_ta_chunks * self.number_of_state_bits_ta +
                    self.number_of_state_bits_ta - 1)
        stop = int((clause + 1) * self.number_of_ta_chunks * self.number_of_state_bits_ta +
                   self.number_of_state_bits_ta - 1)
        return np.unpackbits(
            np.ascontiguousarray(self.clause_bank[start:stop:self.number_of_state_bits_ta]).view('uint8')
        ).sum()

    def get_ta_state(self, clause, ta):
        return super().get_ta_state(clause, ta)

    def set_ta_state(self, clause, ta, state):
        super().set_ta_state(clause, ta, state)

    def _cffi_init(self):
        # No extra CFFI initialization is required on the host side.
        pass


# Device branch: encapsulates all GPU related functions and memory
class ClauseBankCudaDevice:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self._profiler = CudaProfiler()

        # Initialize CUDA in this process if needed

        cuda.init()

        # Store device reference but don't create context yet
        self.device = cuda.Device(0)
        self.cuda_ctx = None
        self._initialize_cuda()

    def _initialize_cuda(self):
        """Initialize CUDA context and resources"""
        # Create context
        self.cuda_ctx = self.device.make_context()
        try:
            import pycuda.curandom as curandom
            self.rng_gen = curandom.XORWOWRandomNumberGenerator()

            # Allocate GPU memory for clause bank from host data
            self.clause_bank_gpu = self._profiler.profile(
                cuda.mem_alloc,
                self.coordinator.host.clause_bank.nbytes
            )
            self._profiler.profile(
                cuda.memcpy_htod,
                self.clause_bank_gpu,
                self.coordinator.host.clause_bank
            )
            # Allocate other device memories
            self.clause_output_gpu = self._profiler.profile(
                cuda.mem_alloc,
                self.coordinator.host.clause_output.nbytes
            )
            self.clause_output_patchwise_gpu = self._profiler.profile(
                cuda.mem_alloc,
                self.coordinator.host.clause_output_patchwise.nbytes
            )
            self.clause_active_gpu = self._profiler.profile(
                cuda.mem_alloc,
                self.coordinator.host.clause_output.nbytes
            )
            self.literal_active_gpu = self._profiler.profile(
                cuda.mem_alloc,
                self.coordinator.number_of_ta_chunks * 4
            )
            self.literal_clause_count_gpu = self._profiler.profile(
                cuda.mem_alloc,
                self.coordinator.host.literal_clause_count.nbytes
            )

            parameters = [f"#define NUMBER_OF_PATCHES {self.coordinator.number_of_patches}"]

            mod = load_cuda_kernel(parameters, "cuda/calculate_clause_outputs_predict.cu")
            self.calculate_clause_outputs_predict_gpu = mod.get_function("calculate_clause_outputs_predict")
            self.calculate_clause_outputs_predict_gpu.prepare("PiiiPPi")
            self.calculate_literal_frequency_gpu = mod.get_function("calculate_literal_frequency")
            self.calculate_literal_frequency_gpu.prepare("PiiiPP")

            mod = load_cuda_kernel(parameters, "cuda/calculate_clause_outputs_update.cu")
            self.calculate_clause_outputs_update_gpu = mod.get_function("calculate_clause_outputs_update")
            self.calculate_clause_outputs_update_gpu.prepare("PiiiPPPi")

            mod = load_cuda_kernel(parameters, "cuda/calculate_clause_outputs_patchwise.cu")
            self.calculate_clause_outputs_patchwise_gpu = mod.get_function("calculate_clause_outputs_patchwise")
            self.calculate_clause_outputs_patchwise_gpu.prepare("PiiiPPi")

            mod = load_cuda_kernel(parameters, "cuda/clause_feedback.cu")
            self.type_i_feedback_gpu = mod.get_function("type_i_feedback")
            self.type_i_feedback_gpu.prepare("PPiiiffiiPPPi")
            self.type_ii_feedback_gpu = mod.get_function("type_ii_feedback")
            self.type_ii_feedback_gpu.prepare("PPiiifPPPi")

            mod = load_cuda_kernel(parameters, "cuda/tools.cu")
            self.produce_autoencoder_examples_gpu = mod.get_function("produce_autoencoder_example")
            self.produce_autoencoder_examples_gpu.prepare("PPiPPiPPiPiii")

        finally:
            # Pop context after initialization
            self.cuda_ctx.pop()

    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'cuda_ctx') and self.cuda_ctx is not None:
            try:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach() # Detach context from this thread
            except:
                pass
            self.cuda_ctx = None

    def _ensure_context(self):
        """Ensure CUDA context is initialized"""
        if self.cuda_ctx is None:
            self._initialize_cuda()
        else:
            self.cuda_ctx.push()

    def synchronize_clause_bank(self):
        try:
            self._ensure_context()
            # Synchronize all device state to host
            self._profiler.profile(
                cuda.memcpy_dtoh,
                self.coordinator.host.clause_bank,
                self.clause_bank_gpu
            )
            self._profiler.profile(
                cuda.memcpy_dtoh,
                self.coordinator.host.clause_output,
                self.clause_output_gpu
            )
            self._profiler.profile(
                cuda.memcpy_dtoh,
                self.coordinator.host.clause_output_patchwise,
                self.clause_output_patchwise_gpu
            )
            self._profiler.profile(
                cuda.memcpy_dtoh,
                self.coordinator.host.literal_clause_count,
                self.literal_clause_count_gpu
            )
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def upload_clause_bank(self, host_array):
        self._profiler.profile(cuda.memcpy_htod, self.clause_bank_gpu, host_array)

    def calculate_clause_outputs_predict(self, encoded_X, e, host_output):
        try:
            self._ensure_context()
            self.calculate_clause_outputs_predict_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
                self.clause_bank_gpu,
                self.coordinator.host.number_of_clauses,
                self.coordinator.host.number_of_literals,
                self.coordinator.host.number_of_state_bits_ta,
                self.clause_output_gpu,
                encoded_X,
                np.int32(e)
            )
            self.cuda_ctx.synchronize()
            self._profiler.profile(
                cuda.memcpy_dtoh,
                host_output,
                self.clause_output_gpu
            )
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e, host_output):
        try:
            self._ensure_context()
            self._profiler.profile(
                cuda.memcpy_htod,
                self.literal_active_gpu,
                literal_active
            )
            self.calculate_clause_outputs_update_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
                self.clause_bank_gpu,
                self.coordinator.host.number_of_clauses,
                self.coordinator.host.number_of_literals,
                self.coordinator.host.number_of_state_bits_ta,
                self.clause_output_gpu,
                self.literal_active_gpu,
                encoded_X,
                np.int32(e)
            )
            self.cuda_ctx.synchronize()
            self._profiler.profile(cuda.memcpy_dtoh, host_output, self.clause_output_gpu)
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def calculate_clause_outputs_patchwise(self, encoded_X, e, host_output):
        try:
            self._ensure_context()
            self.calculate_clause_outputs_patchwise_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
                self.clause_bank_gpu,
                self.coordinator.host.number_of_clauses,
                self.coordinator.host.number_of_literals,
                self.coordinator.host.number_of_state_bits_ta,
                self.clause_output_patchwise_gpu,
                encoded_X,
                np.int32(e)
            )
            self.cuda_ctx.synchronize()
            self._profiler.profile(cuda.memcpy_dtoh, host_output, self.clause_output_patchwise_gpu)
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def type_i_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        try:
            self._ensure_context()
            self._profiler.profile(
                cuda.memcpy_htod,
                self.clause_active_gpu,
                clause_active
            )
            self._profiler.profile(
                cuda.memcpy_htod,
                self.literal_active_gpu,
                literal_active
            )
            self.type_i_feedback_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
                self.rng_gen.state,
                self.clause_bank_gpu,
                self.coordinator.host.number_of_clauses,
                self.coordinator.host.number_of_literals,
                self.coordinator.host.number_of_state_bits_ta,
                update_p,
                self.coordinator.host.s,
                self.coordinator.host.boost_true_positive_feedback,
                self.coordinator.host.max_included_literals,
                self.clause_active_gpu,
                self.literal_active_gpu,
                encoded_X,
                np.int32(e)
            )
            self.cuda_ctx.synchronize()
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        try:
            self._ensure_context()
            self._profiler.profile(
                cuda.memcpy_htod,
                self.clause_active_gpu,
                np.ascontiguousarray(clause_active)
            )
            self._profiler.profile(
                cuda.memcpy_htod,
                self.literal_active_gpu,
                literal_active
            )
            self.type_ii_feedback_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
                self.rng_gen.state,
                self.clause_bank_gpu,
                self.coordinator.host.number_of_clauses,
                self.coordinator.host.number_of_literals,
                self.coordinator.host.number_of_state_bits_ta,
                update_p,
                self.clause_active_gpu,
                self.literal_active_gpu,
                encoded_X,
                np.int32(e)
            )
            self.cuda_ctx.synchronize()
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def calculate_literal_clause_frequency(self, clause_active, host_literal_clause_count):
        try:
            self._ensure_context()
            self._profiler.profile(
                cuda.memcpy_htod,
                self.clause_active_gpu,
                np.ascontiguousarray(clause_active)
            )
            self.calculate_literal_frequency_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
                self.clause_bank_gpu,
                self.coordinator.host.number_of_clauses,
                self.coordinator.host.number_of_literals,
                self.coordinator.host.number_of_state_bits_ta,
                self.clause_active_gpu,
                self.literal_clause_count_gpu
            )
            self.cuda_ctx.synchronize()
            self._profiler.profile(cuda.memcpy_dtoh, host_literal_clause_count, self.literal_clause_count_gpu)
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def prepare_X(self, X):
        self._ensure_context()
        encoded_X = tmu.tools.encode(
            X,
            X.shape[0],
            self.coordinator.number_of_patches,
            self.coordinator.host.number_of_ta_chunks,
            self.coordinator.dim,
            self.coordinator.patch_dim,
            0
        )
        encoded_X_gpu = self._profiler.profile(cuda.mem_alloc, encoded_X.nbytes)
        self._profiler.profile(cuda.memcpy_htod, encoded_X_gpu, encoded_X)
        return encoded_X_gpu

    def prepare_X_autoencoder(self, X_csr, X_csc, active_output):
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
        X = np.ascontiguousarray(np.zeros(int(self.coordinator.host.number_of_ta_chunks), dtype=np.uint32))
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

    def produce_autoencoder_example(self, encoded_X, target, accumulation):
        target_value = np.random.choice(2) 
        self._ensure_context()
        try:
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
            self.produce_autoencoder_examples_gpu.prepared_call(
                self.coordinator.grid,
                self.coordinator.block,
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
             
            # self.produce_autoencoder_examples_gpu.prepared_call(
            #     self.coordinator.grid,
            #     self.coordinator.block,
            #     self.rng_gen.state,
            #     *encoded_X,
            #     int(target),
            #     int(target_value),
            #     int(accumulation)
            # )
            self.cuda_ctx.synchronize()
            X_gpu = encoded_X[-1]
            return X_gpu, target_value
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

    def cleanup(self):
        """Clean up all CUDA resources"""
        try:
            self._ensure_context()

            # Clean up context
            if hasattr(self, 'cuda_ctx') and self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
                self.cuda_ctx = None

        except Exception as e:
            _LOGGER.warning(f"Error during CUDA cleanup: {e}")


# Coordinator class that combines host and device branches.
# This class exposes the same public API as before but delegates CPU (host) and GPU (device) operations.
class ClauseBankCuda(BaseClauseBank):
    def __init__(
        self,
        seed: int,
        number_of_state_bits_ta: int,
        grid: tuple = (16 * 13, 1, 1),
        block: tuple = (128, 1, 1),
        **kwargs
    ):
        self.grid = grid
        self.block = block



        self.number_of_state_bits_ta = number_of_state_bits_ta
        # Initialize host part
        self.host = ClauseBankCudaHost(seed, number_of_state_bits_ta, **kwargs)
        # Copy additional attributes for easier access
        self.number_of_clauses = self.host.number_of_clauses
        self.number_of_literals = self.host.number_of_literals
        self.number_of_ta_chunks = self.host.number_of_ta_chunks
        self.number_of_patches = self.host.number_of_patches
        self.dim = getattr(self.host, "dim", None)
        self.patch_dim = getattr(self.host, "patch_dim", None)
        self.s = getattr(self.host, "s", None)
        self.boost_true_positive_feedback = getattr(self.host, "boost_true_positive_feedback", None)
        self.max_included_literals = getattr(self.host, "max_included_literals", None)

        if not cuda_installed:
            raise RuntimeError("CUDA support is required for ClauseBankCuda")

        # Create device with proper context management
        self.device = ClauseBankCudaDevice(self)
        self.device.synchronize_clause_bank()

    def calculate_clause_outputs_predict(self, encoded_X, e):
        if self.device:
            self.device.calculate_clause_outputs_predict(encoded_X, e, self.host.clause_output)
            return self.host.clause_output
        else:
            return None

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        if self.device:
            self.device.calculate_clause_outputs_update(literal_active, encoded_X, e,
                                                          self.host.clause_output)
            return self.host.clause_output
        else:
            return None

    def calculate_clause_outputs_patchwise(self, encoded_X, e):
        if self.device:
            self.device.calculate_clause_outputs_patchwise(encoded_X, e,
                                                            self.host.clause_output_patchwise)
            return self.host.clause_output_patchwise
        else:
            return None

    def type_i_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        if self.device:
            self.device.type_i_feedback(update_p, clause_active, literal_active, encoded_X, e)

    def type_ii_feedback(self, update_p, clause_active, literal_active, encoded_X, e):
        if self.device:
            self.device.type_ii_feedback(update_p, clause_active, literal_active, encoded_X, e)

    def calculate_literal_clause_frequency(self, clause_active):
        if self.device:
            self.device.calculate_literal_clause_frequency(clause_active,
                                                             self.host.literal_clause_count)
            return self.host.literal_clause_count
        else:
            return None

    def get_ta_action(self, clause, ta):
        if self.device:
            self.device.synchronize_clause_bank()
        return self.host.get_ta_action(clause, ta)

    def get_ta_state(self, clause, ta):
        if self.device:
            self.device.synchronize_clause_bank()
        return self.host.get_ta_state(clause, ta)

    def set_ta_state(self, clause, ta, state):
        self.host.set_ta_state(clause, ta, state)
        if self.device:
            self.device.upload_clause_bank(self.host.clause_bank)

    def prepare_X(self, X):
        if self.device:
            return self.device.prepare_X(X)
        else:
            return tmu.tools.encode(
                X,
                X.shape[0],
                self.number_of_patches,
                self.host.number_of_ta_chunks,
                self.dim,
                self.patch_dim,
                0
            )

    def prepare_X_autoencoder(self, X_csr, X_csc, active_output):
        if self.device:
            return self.device.prepare_X_autoencoder(X_csr, X_csc, active_output)
        else:
            return None

    def produce_autoencoder_example(self, encoded_X, target, accumulation, **kwargs):
        if self.device:
            return self.device.produce_autoencoder_example(encoded_X, target, accumulation)
        else:
            return None

    def synchronize_clause_bank(self):
        if self.device:
            self.device.synchronize_clause_bank()

    def __getstate__(self):
        if self.device is not None:
            # synchronize and cleanup
            self.device.synchronize_clause_bank()
            self.device.cleanup()
            self.device = None
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not cuda_installed:
            raise RuntimeError("CUDA support is required for ClauseBankCuda")
        # Recreate device with fresh context
        self.device = ClauseBankCudaDevice(self)

    def _cffi_init(self):
        # No extra CFFI initialization is needed because the host branch already handles it.
        pass

ClauseBankCUDA = ClauseBankCuda
