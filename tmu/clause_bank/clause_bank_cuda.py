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
import numpy as np

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
    
class ClauseBankCUDA(BaseClauseBank):
    clause_bank: np.ndarray
    co_p = None  # _cffi_backend._CDataBase
    cob_p = None  # _cffi_backend._CDataBase
    ptr_clause_and_target = None  # _cffi_backend._CDataBase
    cop_p = None  # _cffi_backend._CDataBase
    ptr_feedback_to_ta = None  # _cffi_backend._CDataBase
    ptr_output_one_patches = None  # _cffi_backend._CDataBase
    ptr_literal_clause_count = None  # _cffi_backend._CDataBase
    ptr_actions = None  # _cffi_backend._CDataBase

    def __init__(
            self,
            seed: int,
            number_of_state_bits_ta: int,
            **kwargs
    ):
        super().__init__(seed=seed, **kwargs)

        assert isinstance(number_of_state_bits_ta, int)
        self.number_of_state_bits_ta = number_of_state_bits_ta

        self.grid = (16 * 13, 1, 1)
        self.block = (128, 1, 1)

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

        mod = load_cuda_kernel(parameters, "cuda/calculate_clause_value_in_patch.cu")
        self.calculate_clause_value_in_patch_gpu = mod.get_function("calculate_clause_value_in_patch")
        self.calculate_clause_value_in_patch_gpu.prepare("iiiPPPPP")

        self.clause_output = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")
        self.clause_and_target = np.zeros(self.number_of_clauses * self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.clause_output_patchwise = np.empty(self.number_of_clauses * self.number_of_patches, dtype=np.uint32, order="c")
        self.feedback_to_ta = np.empty(self.number_of_ta_chunks, dtype=np.uint32, order="c")
        self.output_one_patches = np.empty(self.number_of_patches, dtype=np.uint32, order="c")
        self.literal_clause_count = np.empty(self.number_of_literals, dtype=np.uint32, order="c")
        self.type_ia_feedback_counter = np.zeros(self.number_of_clauses, dtype=np.uint32, order="c")
        
        self.current_clause_node_output_test = np.empty((self.number_of_clauses, self.number_of_patch_chunks), dtype=np.uint32, order="c")
        self.current_clause_node_output_test_gpu = cuda.mem_alloc(self.current_clause_node_output_test.nbytes)
        self.next_clause_node_output_test_gpu = cuda.mem_alloc(int(self.number_of_clauses * self.number_of_patch_chunks) * 4)
        
        if self.spatio_temporal:
            self.xi_hypervector = np.empty(self.number_of_patches * self.number_of_ta_chunks, dtype=np.uint32, order="c")

            self.clause_value_in_patch = np.empty(self.number_of_patches * self.number_of_clauses, dtype=np.uint32, order="c")
            self.clause_value_in_patch_tmp = np.empty(self.number_of_patches * self.number_of_clauses, dtype=np.uint32, order="c")

            self.clause_true_consecutive = np.empty(self.number_of_patches, dtype=np.uint32, order="c")
            self.clause_true_consecutive_before = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")
            self.clause_false_consecutive_before = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")
            
            self.clause_truth_value_transitions = np.empty(self.number_of_patches * self.number_of_clauses * 3, dtype=np.uint32, order="c")
            self.clause_truth_value_transitions_length = np.empty(self.number_of_patches, dtype=np.uint32, order="c")

            self.attention = np.empty(self.number_of_ta_chunks, dtype=np.uint32, order="c")
            self.attention_gpu = cuda.mem_alloc(self.attention.nbytes)

            self.hypervectors = np.empty((self.number_of_clauses, self.hypervector_bits), dtype=np.uint32, order="c")
            indexes = np.arange(self.hypervector_size, dtype=np.uint32)
            for i in range(self.number_of_clauses):
                self.hypervectors[i,:] = np.random.choice(indexes, size=(self.hypervector_bits), replace=False)
            self.hypervectors = self.hypervectors.reshape(self.number_of_clauses*self.hypervector_bits)

        # Incremental Clause Evaluation
        self.literal_clause_map = np.empty(
            (int(self.number_of_literals * self.number_of_clauses)),
            dtype=np.uint32,
            order="c"
        )
        self.literal_clause_map_pos = np.empty(
            (int(self.number_of_literals)),
            dtype=np.uint32,
            order="c"
        )
        self.false_literals_per_clause = np.empty(
            int(self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )
        self.previous_xi = np.empty(
            int(self.number_of_ta_chunks) * int(self.number_of_patches),
            dtype=np.uint32,
            order="c"
        )

        self.initialize_clauses()

        # Finally, map numpy arrays to CFFI compatible pointers.
        self._cffi_init()

        # Set pcg32 seed
        if self.seed is not None:
            assert isinstance(self.seed, int), "Seed must be a integer"

            lib.pcg32_seed(self.seed)
            lib.xorshift128p_seed(self.seed)

    def _cffi_init(self):
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)
        self.ptr_clause_and_target = ffi.cast("unsigned int *", self.clause_and_target.ctypes.data)
        self.cop_p = ffi.cast("unsigned int *", self.clause_output_patchwise.ctypes.data)
        self.ptr_feedback_to_ta = ffi.cast("unsigned int *", self.feedback_to_ta.ctypes.data)
        self.ptr_output_one_patches = ffi.cast("unsigned int *", self.output_one_patches.ctypes.data)
        self.ptr_literal_clause_count = ffi.cast("unsigned int *", self.literal_clause_count.ctypes.data)
        self.tiafc_p = ffi.cast("unsigned int *", self.type_ia_feedback_counter.ctypes.data)
        self.xih_p = ffi.cast("unsigned int *", self.xi_hypervector.ctypes.data)

        if self.spatio_temporal:
            self.cvip_p = ffi.cast("unsigned int *", self.clause_value_in_patch.ctypes.data)
            self.cvipt_p = ffi.cast("unsigned int *", self.clause_value_in_patch_tmp.ctypes.data)

            self.ctc_p = ffi.cast("unsigned int *", self.clause_true_consecutive.ctypes.data)
            self.ctcb_p = ffi.cast("unsigned int *", self.clause_true_consecutive_before.ctypes.data)
            self.cfcb_p = ffi.cast("unsigned int *", self.clause_false_consecutive_before.ctypes.data)

            self.ctvt_p = ffi.cast("unsigned int *", self.clause_truth_value_transitions.ctypes.data)
            self.ctvtl_p = ffi.cast("unsigned int *", self.clause_truth_value_transitions_length.ctypes.data)

            self.a_p = ffi.cast("unsigned int *", self.attention.ctypes.data)
            self.hv_p = ffi.cast("unsigned int *", self.hypervectors.ctypes.data)

        # Clause Initialization
        self.ptr_ta_state = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)

        # Action Initialization
        self.ptr_actions = ffi.cast("unsigned int *", self.actions.ctypes.data)

        # Incremental Clause Evaluation Initialization
        self.lcm_p = ffi.cast("unsigned int *", self.literal_clause_map.ctypes.data)
        self.lcmp_p = ffi.cast("unsigned int *", self.literal_clause_map_pos.ctypes.data)
        self.flpc_p = ffi.cast("unsigned int *", self.false_literals_per_clause.ctypes.data)
        self.previous_xi_p = ffi.cast("unsigned int *", self.previous_xi.ctypes.data)

    def initialize_clauses(self):
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="c"
        )

        self.clause_bank[:, :, 0: self.number_of_state_bits_ta - 1] = np.uint32(~0)
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta)))

        self.ta_state_gpu = cuda.mem_alloc(clause_bank.nbytes)

        self.actions = np.ascontiguousarray(np.zeros(self.number_of_ta_chunks, dtype=np.uint32))

    def calculate_clause_outputs_predict(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if self.spatio_temporal:
            lib.cb_prepare_hypervector(
                self.number_of_input_features,
                self.number_of_patches,
                self.hypervector_size,
                self.depth,
                xi_p,
                self.xih_p
            )

            current_clause_node_output = self.current_clause_node_output_test_gpu
            next_clause_node_output = self.current_clause_node_output_test_gpu

            clause_bank = self.clause_bank.reshape(-1)
            cuda.memcpy_htod(self.ta_state_gpu, clause_bank)

            encoded_X_gpu = cuda.mem_alloc(encoded_X[e, :].nbytes)
            cuda.memcpy_htod(encoded_X_gpu, encoded_X[e, :])

            self.attention[:] = 0
            for k in range(self.hypervector_size*self.depth, self.number_of_features):
                chunk_nr = k // 32
                chunk_pos = k % 32
                self.attention[chunk_nr] |= (1 << chunk_pos)

                chunk_nr = (k + self.number_of_features) // 32
                chunk_pos = (k + self.number_of_features) % 32

                self.attention[chunk_nr] |= (1 << chunk_pos);
            cuda.memcpy_htod(self.attention_gpu, self.attention)

            self.calculate_clause_value_in_patch_gpu.prepared_call(
                self.grid,
                self.block,
                self.number_of_clauses,
                self.number_of_features,
                self.number_of_state_bits_ta,
                self.ta_state_gpu,
                current_clause_node_output,
                next_clause_node_output,
                self.attention_gpu,
                encoded_X_gpu
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(self.current_clause_node_output_test, self.current_clause_node_output_test_gpu)
            ccnot_p = ffi.cast("unsigned int *", self.current_clause_node_output_test.ctypes.data)

            lib.cb_calculate_spatio_temporal_features(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_features,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.depth,
                self.hypervector_size,
                self.hypervector_bits,
                self.cvip_p,
                self.cvipt_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.a_p,
                self.hv_p,
                self.xih_p,
                ccnot_p
            )

            lib.cb_calculate_clause_outputs_predict_spatio_temporal(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.co_p,
                self.cvip_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.xih_p
            )
        else:
            lib.cb_calculate_clause_outputs_predict(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.co_p,
                self.xih_p
            )
        
        return self.clause_output

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        la_p = ffi.cast("unsigned int *", literal_active.ctypes.data)

        if self.spatio_temporal:
            lib.cb_prepare_hypervector(
                self.number_of_input_features,
                self.number_of_patches,
                self.hypervector_size,
                self.depth,
                xi_p,
                self.xih_p
            )

            current_clause_node_output = self.current_clause_node_output_test_gpu
            next_clause_node_output = self.current_clause_node_output_test_gpu

            clause_bank = self.clause_bank.reshape(-1)
            self.ta_state_gpu = cuda.mem_alloc(clause_bank.nbytes)
            cuda.memcpy_htod(self.ta_state_gpu, clause_bank)

            encoded_X_gpu = cuda.mem_alloc(encoded_X[e, :].nbytes)
            cuda.memcpy_htod(encoded_X_gpu, encoded_X[e, :])

            for k in range(self.hypervector_size*self.depth, self.number_of_features):
                chunk_nr = k // 32
                chunk_pos = k % 32
                self.attention[chunk_nr] |= (1 << chunk_pos)

                chunk_nr = (k + self.number_of_features) // 32
                chunk_pos = (k + self.number_of_features) % 32

                self.attention[chunk_nr] |= (1 << chunk_pos);
            cuda.memcpy_htod(self.attention_gpu, self.attention)

            self.calculate_clause_value_in_patch_gpu.prepared_call(
                self.grid,
                self.block,
                self.number_of_clauses,
                self.number_of_features,
                self.number_of_state_bits_ta,
                self.ta_state_gpu,
                current_clause_node_output,
                next_clause_node_output,
                self.attention_gpu,
                encoded_X_gpu
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(self.current_clause_node_output_test, self.current_clause_node_output_test_gpu)
            ccnot_p = ffi.cast("unsigned int *", self.current_clause_node_output_test.ctypes.data)

            lib.cb_calculate_spatio_temporal_features(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_features,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.depth,
                self.hypervector_size,
                self.hypervector_bits,
                self.cvip_p,
                self.cvipt_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.a_p,
                self.hv_p,
                self.xih_p,
                ccnot_p
            )

            lib.cb_calculate_clause_outputs_update_spatio_temporal(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.co_p,
                la_p,
                self.cvip_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.xih_p
            )
        else:
            lib.cb_calculate_clause_outputs_update(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.co_p,
                la_p,
                xi_p
            )

        return self.clause_output

    def calculate_clause_outputs_patchwise(self, encoded_X, e):
        xi_p = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)

        if self.spatio_temporal:
            lib.cb_prepare_hypervector(
                self.number_of_input_features,
                self.number_of_patches,
                self.hypervector_size,
                self.depth,
                xi_p,
                self.xih_p
            )

            lib.cb_calculate_spatio_temporal_features(
                self.ptr_ta_state,
                self.number_of_clauses,
                self.number_of_features,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                self.depth,
                self.hypervector_size,
                self.hypervector_bits,
                self.cvip_p,
                self.cvipt_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.a_p,
                self.hv_p,
                self.xih_p
            )

        lib.cb_calculate_clause_outputs_patchwise(
            self.ptr_ta_state,
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
        e
    ):
        ptr_xi = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_literal_active = ffi.cast("unsigned int *", literal_active.ctypes.data)

        if self.spatio_temporal:
            lib.cb_type_i_feedback_spatio_temporal(
                self.ptr_ta_state,
                self.ptr_feedback_to_ta,
                self.ptr_output_one_patches,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                update_p,
                self.s,
                self.boost_true_positive_feedback,
                self.reuse_random_feedback,
                self.max_included_literals,
                ptr_clause_active,
                ptr_literal_active,
                self.cvip_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.xih_p
            )
        else:
            lib.cb_type_i_feedback(
                self.ptr_ta_state,
                self.ptr_feedback_to_ta,
                self.ptr_output_one_patches,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                update_p,
                self.s,
                self.boost_true_positive_feedback,
                self.reuse_random_feedback,
                self.max_included_literals,
                ptr_clause_active,
                ptr_literal_active,
                ptr_xi
            )

    def type_ii_feedback(
        self,
        update_p,
        clause_active,
        literal_active,
        encoded_X,
        e
    ):
        ptr_xi = ffi.cast("unsigned int *", encoded_X[e, :].ctypes.data)
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        ptr_literal_active = ffi.cast("unsigned int *", literal_active.ctypes.data)

        if self.spatio_temporal:
            lib.cb_type_ii_feedback_spatio_temporal(
                self.ptr_ta_state,
                self.ptr_output_one_patches,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                update_p,
                ptr_clause_active,
                ptr_literal_active,
                self.cvip_p,
                self.ctc_p,
                self.ctcb_p,
                self.cfcb_p,
                self.ctvt_p,
                self.ctvtl_p,
                self.xih_p
            )
        else:
            lib.cb_type_ii_feedback(
                self.ptr_ta_state,
                self.ptr_output_one_patches,
                self.number_of_clauses,
                self.number_of_literals,
                self.number_of_state_bits_ta,
                self.number_of_patches,
                update_p,
                ptr_clause_active,
                ptr_literal_active,
                ptr_xi
            )

    def calculate_literal_clause_frequency(
            self,
            clause_active
    ):
        ptr_clause_active = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cb_calculate_literal_frequency(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ptr_clause_active,
            self.ptr_literal_clause_count
        )
        return self.literal_clause_count

    def included_literals(self):
        lib.cb_included_literals(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.ptr_actions
        )
        return self.actions

    def get_literals(self, independent=False):

        result = np.zeros((self.number_of_clauses, self.number_of_literals), dtype=np.uint32, order="c")
        result_p = ffi.cast("unsigned int *", result.ctypes.data)
        lib.cb_get_literals(
            self.ptr_ta_state_ind if independent else self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            result_p
        )
        return result

    def calculate_independent_literal_clause_frequency(self, clause_active):
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.cb_calculate_literal_frequency(
            self.ptr_ta_state_ind,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            ca_p,
            self.ptr_literal_clause_count
        )
        return self.literal_clause_count

    def number_of_include_actions(
            self,
            clause
    ):
        return lib.cb_number_of_include_actions(
            self.ptr_ta_state,
            clause,
            self.number_of_literals,
            self.number_of_state_bits_ta
        )

    def prepare_X(
            self,
            X
    ):
        return tmu.tools.encode(
            X,
            X.shape[0],
            self.number_of_patches,
            self.number_of_input_ta_chunks,
            self.dim,
            self.patch_dim,
            0
        )

    def prepare_X_autoencoder(
            self,
            X_csr,
            X_csc,
            active_output
    ):
        X = np.ascontiguousarray(np.empty(int(self.number_of_input_ta_chunks), dtype=np.uint32))
        return X_csr, X_csc, active_output, X

    def produce_autoencoder_example(
            self,
            encoded_X,
            target,
            target_true_p,
            accumulation
    ):
        (X_csr, X_csc, active_output, X) = encoded_X

        target_value = self.rng.random() <= target_true_p

        lib.tmu_produce_autoencoder_example(ffi.cast("unsigned int *", active_output.ctypes.data), active_output.shape[0],
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indptr).ctypes.data),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indices).ctypes.data),
                                             int(X_csr.shape[0]),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indptr).ctypes.data),
                                             ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indices).ctypes.data),
                                             int(X_csc.shape[1]),
                                             ffi.cast("unsigned int *", X.ctypes.data),
                                             int(target),
                                             int(target_value),
                                             int(accumulation))

        return X.reshape((1, -1)), target_value
