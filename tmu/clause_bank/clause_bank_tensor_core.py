# Tensor-core clause bank.
# The include test uses packed binary Tensor Cores on SM80-SM90 and an INT8
# Tensor Core fallback elsewhere. Training keeps the mask fixed for a minibatch
# and applies the same Type I / Type II rule on CUDA cores. predict() uses the
# selected matcher over the encoded batch and rejects empty clauses.
# The match grid is one block per row tile, class, and 8-clause panel, so the
# streaming multiprocessors share the work. The model batch_size controls how
# long the include mask stays fixed.
#
# This bank stands on BaseClauseBank. The device state and the kernels live in
# this module and cuda/tensor_core.cu.

import atexit
import os
from dataclasses import dataclass

import numpy as np

import tmu.tools
from tmu.clause_bank.base_clause_bank import BaseClauseBank
from tmu.clause_bank.clause_bank_cuda import load_cuda_kernel

try:
    import pycuda.driver as cuda
    _CUDA = True
except Exception:
    cuda = None
    _CUDA = False

_CTX = None
_OWNS_CONTEXT = False
_TC_MODULE = None
_CLEANED = False
# Partial panel scores stay under this so a long test set does not allocate
# one buffer for every clause panel at once.
_PARTIAL_BYTES = 64 * 1024 * 1024
_MAX_FEEDBACK_BATCH = (48 * 1024) // 4
# A longer freeze scores later examples against a stale include mask.
_MASK_BATCH = 64


@dataclass(frozen=True)
class FeedbackLaunch:
    threads: int
    shared_bytes: int
    batch_limit: int

    def grid(self, clauses, slots):
        return int(clauses), int(slots), 1


@dataclass(frozen=True)
class TensorCoreRuntime:
    compute_capability: tuple
    matcher: str
    max_threads_per_block: int
    shared_memory_per_block: int
    multiprocessor_count: int
    max_grid_z: int
    total_memory: int
    partial_bytes: int

    def row_launch(self, rows):
        rows = max(1, int(rows))
        max_warps = max(1, min(8, self.max_threads_per_block // 32))
        wanted_warps = 1
        for candidate in (2, 4, 8):
            if rows >= candidate * 16 and candidate <= max_warps:
                wanted_warps = candidate
        threads = wanted_warps * 32
        return wanted_warps * 16, threads

    def match_launch(self, rows, panels, slots):
        """Size row tiles while keeping enough independent blocks for the GPU."""
        rows = max(1, int(rows))
        panels = max(1, int(panels))
        slots = max(1, int(slots))
        _tile_rows, threads = self.row_launch(rows)
        warps = threads // 32
        target_blocks = max(1, self.multiprocessor_count * 2)
        column_blocks = panels * slots
        while warps > 1:
            row_tiles = (rows + warps * 16 - 1) // (warps * 16)
            if row_tiles * column_blocks >= target_blocks:
                break
            warps //= 2
        return warps * 16, warps * 32

    def feedback_launch(self, samples, chunks):
        batch_limit = ((self.shared_memory_per_block - 4) // 4) * 4
        samples = min(max(1, int(samples)), batch_limit)
        chunk_warps = max(1, (max(1, int(chunks)) + 31) // 32)
        max_warps = max(1, min(8, self.max_threads_per_block // 32))
        warps = min(chunk_warps, max_warps)
        shared_bytes = ((samples + 3) // 4) * 4 + 4
        return FeedbackLaunch(
            threads=warps * 32,
            shared_bytes=shared_bytes,
            batch_limit=batch_limit,
        )

    def prediction_batch_rows(self, rows, chunks, slots):
        rows = max(1, int(rows))
        bytes_per_row = max(4, int(chunks) * 4 + int(slots) * 4)
        budget = max(16 * 1024**2, self.total_memory // 16)
        return min(rows, max(16, budget // bytes_per_row))


def _runtime_attribute(attributes, name, default):
    if name in attributes:
        return int(attributes[name])
    for key, value in attributes.items():
        key_name = getattr(key, "name", str(key))
        if key_name == name or key_name.endswith(f".{name}"):
            return int(value)
    return int(default)


def build_tensor_core_runtime(
        compute_capability,
        attributes,
        total_memory,
        matcher=None,
):
    capability = tuple(int(part) for part in compute_capability)
    validate_compute_capability(capability)
    selected = (matcher or os.getenv("TMU_TENSOR_MATCHER", "auto")).strip().lower()
    if selected not in {"auto", "bmma", "imma"}:
        raise ValueError("Tensor matcher must be auto, bmma, or imma")
    binary_supported = (8, 0) <= capability <= (9, 0)
    if selected == "bmma" and not binary_supported:
        raise RuntimeError(
            f"BMMA is supported by this backend on SM80-SM90; found {capability[0]}.{capability[1]}"
        )
    if selected == "auto":
        selected = "bmma" if binary_supported else "imma"
    default_shared = _runtime_attribute(
        attributes, "MAX_SHARED_MEMORY_PER_BLOCK", 48 * 1024)
    optin_shared = _runtime_attribute(
        attributes, "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", default_shared)
    max_threads = _runtime_attribute(attributes, "MAX_THREADS_PER_BLOCK", 1024)
    multiprocessors = _runtime_attribute(attributes, "MULTIPROCESSOR_COUNT", 1)
    max_grid_z = _runtime_attribute(attributes, "MAX_GRID_DIM_Z", 65535)
    shared_memory = max(default_shared, optin_shared)
    if max_threads < 32:
        raise RuntimeError("TensorCore requires a device block limit of at least 32 threads")
    if shared_memory < 8:
        raise RuntimeError("TensorCore feedback requires at least 8 bytes of shared memory")
    if multiprocessors < 1:
        raise RuntimeError("TensorCore requires at least one GPU multiprocessor")
    if max_grid_z < 1:
        raise RuntimeError("TensorCore requires a positive grid Z limit")
    total_memory = max(1, int(total_memory))
    partial_bytes = min(256 * 1024**2, max(16 * 1024**2, total_memory // 64))
    return TensorCoreRuntime(
        compute_capability=capability,
        matcher=selected,
        max_threads_per_block=max_threads,
        shared_memory_per_block=shared_memory,
        multiprocessor_count=multiprocessors,
        max_grid_z=max_grid_z,
        total_memory=total_memory,
        partial_bytes=partial_bytes,
    )


def _runtime_for_device(device):
    attributes = {}
    for name in (
            "MAX_THREADS_PER_BLOCK",
            "MAX_SHARED_MEMORY_PER_BLOCK",
            "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
            "MULTIPROCESSOR_COUNT",
            "MAX_GRID_DIM_Z",
    ):
        attribute = getattr(cuda.device_attribute, name, None)
        if attribute is not None:
            try:
                attributes[name] = device.get_attribute(attribute)
            except Exception:
                pass
    return build_tensor_core_runtime(
        compute_capability=device.compute_capability(),
        attributes=attributes,
        total_memory=device.total_memory(),
    )


def _commit_device_epoch(class_ids, banks, weights, packed_weights):
    for row, class_id in enumerate(class_ids):
        weights[row][:] = packed_weights[int(class_id)]
        banks[row]._host_valid = False


def validate_tensor_core_configuration(
        number_of_patches,
        number_of_ta_chunks,
        number_of_state_bits_ta,
        batch_size,
        reuse_random_feedback,
        type_ia_ii_feedback_ratio,
):
    """Reject modes that the batched Tensor Core implementation cannot preserve."""
    if int(number_of_patches) != 1:
        raise ValueError("TensorCore currently supports exactly one patch")
    if not 1 <= int(number_of_state_bits_ta) <= 16:
        raise ValueError("number_of_state_bits_ta must be between 1 and 16 for TensorCore")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1 for TensorCore")
    if int(reuse_random_feedback):
        raise ValueError("reuse_random_feedback is not supported by TensorCore")
    if float(type_ia_ii_feedback_ratio) != 0.0:
        raise ValueError("type_ia_ii_feedback_ratio is not supported by TensorCore")


def validate_compute_capability(compute_capability):
    """Require the first NVIDIA generation that implements INT8 m16n8k32."""
    capability = tuple(int(part) for part in compute_capability)
    if capability < (8, 0):
        raise RuntimeError(
            "TensorCore requires an NVIDIA GPU with compute capability 8.0 or newer; "
            f"found {capability[0]}.{capability[1]}"
        )


def _ensure_context():
    global _CTX, _OWNS_CONTEXT
    if not _CUDA:
        raise RuntimeError("PyCUDA is required for the TensorCore clause bank")
    try:
        cuda.init()
        if _CTX is None:
            current = cuda.Context.get_current()
            if current is None:
                _CTX = cuda.Device(0).make_context()
                _OWNS_CONTEXT = True
            else:
                _CTX = current
                _OWNS_CONTEXT = False
        validate_compute_capability(_CTX.get_device().compute_capability())
    except Exception as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"Unable to initialize the TensorCore CUDA context: {error}") from error
    return _CTX


def _cleanup_context():
    global _CTX, _CLEANED
    if _CLEANED or _CTX is None:
        return
    _CLEANED = True
    if _OWNS_CONTEXT:
        try:
            if cuda.Context.get_current() == _CTX:
                _CTX.pop()
            _CTX.detach()
        except Exception:
            pass
    _CTX = None


atexit.register(_cleanup_context)


def _tile_for(n, runtime=None):
    """Row tile and block size. One warp owns 16 rows of the m16n8 tile."""
    if runtime is not None:
        return runtime.row_launch(n)
    n = int(n)
    if n >= 128:
        return 128, 256
    if n >= 64:
        return 64, 128
    if n >= 32:
        return 32, 64
    return 16, 32


def _tc_module():
    global _TC_MODULE
    _ensure_context()
    if _TC_MODULE is None:
        _TC_MODULE = load_cuda_kernel([], "cuda/tensor_core.cu")
    return _TC_MODULE


class ClauseBankTensorCore(BaseClauseBank):
    """TMU clause bank whose clause match runs on Tensor Cores."""

    def __init__(
            self,
            seed: int,
            d: float,
            number_of_state_bits_ind: int,
            number_of_state_bits_ta: int,
            batch_size: int,
            incremental: bool,
            early_stop: bool = True,
            **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.d = float(d)
        self.number_of_state_bits_ta = int(number_of_state_bits_ta)
        self.number_of_state_bits_ind = int(number_of_state_bits_ind)
        self.batch_size = int(batch_size)
        self.incremental = bool(incremental)
        self.early_stop = bool(early_stop)
        validate_tensor_core_configuration(
            number_of_patches=self.number_of_patches,
            number_of_ta_chunks=self.number_of_ta_chunks,
            number_of_state_bits_ta=self.number_of_state_bits_ta,
            batch_size=self.batch_size,
            reuse_random_feedback=self.reuse_random_feedback,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio,
        )
        remainder = self.number_of_literals % 32
        self._filter = (1 << remainder) - 1 if remainder else 0xFFFFFFFF
        self._init_host_state()
        self._initialize_device_state()

    def _initialize_device_state(self):
        _ensure_context()
        self._runtime = _runtime_for_device(_CTX.get_device())
        self._profile_enabled = os.getenv("TMU_TENSOR_PROFILE", "").lower() not in {
            "", "0", "false", "no"
        }
        self._timings = {}
        self._ta_gpu = cuda.mem_alloc(self.clause_bank.nbytes)
        cuda.memcpy_htod(self._ta_gpu, self.clause_bank)
        self._host_newer = False
        self._host_valid = True
        self._x_gpu = None
        self._x_bytes = 0
        self._tc_scratch = {}
        mod = _tc_module()
        self._ex_fn = mod.get_function("tc_extract")
        self._ex_fn.prepare("PPPPPiiiiIi")
        self._gather_fn = mod.get_function("tc_gather")
        self._gather_fn.prepare("PPPii")
        self._imma_fn = mod.get_function("tc_imma")
        self._imma_fn.prepare("PPPPPPiiiiiii")
        self._bmma_fn = mod.get_function("tc_bmma")
        self._bmma_fn.prepare("PPPPPPiiiiiii")
        self._reduce_fn = mod.get_function("tc_reduce")
        self._reduce_fn.prepare("PPiiii")
        self._fb_fn = mod.get_function("tc_feedback")
        self._fb_fn.prepare("PPPPPPPPPPiiiiiifiiiffiiiII")

    def _cffi_init(self):
        return None

    def __getstate__(self):
        if hasattr(self, "_host_valid") and not self._host_valid:
            self._pull()
        transient = {
            "_ta_gpu", "_x_gpu", "_x_bytes", "_tc_scratch",
            "_ex_fn", "_gather_fn", "_imma_fn", "_bmma_fn", "_reduce_fn", "_fb_fn",
            "_runtime", "_profile_enabled", "_timings",
        }
        return {key: value for key, value in self.__dict__.items() if key not in transient}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize_device_state()

    def _init_host_state(self):
        bank = np.empty(
            (self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
        )
        bank[:, :, : self.number_of_state_bits_ta - 1] = np.uint32(0xFFFFFFFF)
        bank[:, :, self.number_of_state_bits_ta - 1] = np.uint32(0)
        self.clause_bank = np.ascontiguousarray(bank.reshape(-1))

    def _require_imma(self):
        if not self._tc_ok():
            raise RuntimeError("TensorCore matching currently requires exactly one patch")

    def _push(self):
        if self._host_newer or not self._host_valid:
            if not self._host_valid:
                return
            cuda.memcpy_htod(self._ta_gpu, self.clause_bank)
            self._host_newer = False

    def _pull(self):
        cuda.memcpy_dtoh(self.clause_bank, self._ta_gpu)
        self._host_valid = True
        self._host_newer = False

    def _upload_x(self, encoded_X):
        encoded = np.ascontiguousarray(encoded_X, dtype=np.uint32)
        if self._x_gpu is None or self._x_bytes < encoded.nbytes:
            self._x_gpu = cuda.mem_alloc(encoded.nbytes)
            self._x_bytes = encoded.nbytes
        cuda.memcpy_htod(self._x_gpu, encoded)
        return self._x_gpu

    def set_ta_state(self, clause, ta, state):
        if not self._host_valid:
            self._pull()
        super().set_ta_state(clause, ta, state)
        self._host_newer = True
        self._host_valid = True

    def get_ta_state(self, clause, ta):
        if not self._host_valid:
            self._pull()
        return super().get_ta_state(clause, ta)

    def get_ta_action(self, clause, ta):
        if not self._host_valid:
            self._pull()
        return super().get_ta_action(clause, ta)

    def number_of_include_actions(self, clause):
        if not self._host_valid:
            self._pull()
        bits = int(self.number_of_state_bits_ta)
        chunks = int(self.number_of_ta_chunks)
        start = int(clause) * chunks * bits + bits - 1
        words = self.clause_bank[start:start + chunks * bits:bits]
        total = 0
        last = chunks - 1
        for index, word in enumerate(words):
            value = int(word)
            if index == last:
                value &= int(self._filter)
            total += value.bit_count()
        return total

    def calculate_literal_clause_frequency(self, clause_active):
        if not self._host_valid:
            self._pull()
        active = np.asarray(clause_active, dtype=np.uint32).reshape(-1)
        if active.size != self.number_of_clauses:
            raise ValueError("clause_active must contain one entry per clause")
        actions = self.clause_bank.reshape(
            self.number_of_clauses,
            self.number_of_ta_chunks,
            self.number_of_state_bits_ta,
        )[:, :, self.number_of_state_bits_ta - 1]
        frequency = np.zeros(self.number_of_literals, dtype=np.uint32)
        for literal in range(self.number_of_literals):
            word = literal // 32
            mask = np.uint32(1 << (literal % 32))
            frequency[literal] = np.count_nonzero(
                (active != 0) & ((actions[:, word] & mask) != 0)
            )
        return frequency

    def calculate_clause_outputs_patchwise(self, encoded_X, e):
        return self.calculate_clause_outputs_predict(encoded_X, e)

    def prepare_X(self, X):
        return tmu.tools.encode(
            np.ascontiguousarray(X),
            int(X.shape[0]),
            int(self.number_of_patches),
            int(self.number_of_ta_chunks),
            self.dim,
            self.patch_dim,
            0,
        )

    def _tc_buf(self, name, nbytes):
        current = self._tc_scratch.get(name)
        if current is None or current[1] < nbytes:
            current = (cuda.mem_alloc(max(int(nbytes), 4)), int(nbytes))
            self._tc_scratch[name] = current
        return current[0]

    def _time_call(self, name, operation, *args, **kwargs):
        if not getattr(self, "_profile_enabled", False):
            return operation(*args, **kwargs)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        result = operation(*args, **kwargs)
        end.record()
        end.synchronize()
        entry = self._timings.setdefault(name, {"seconds": 0.0, "calls": 0})
        entry["seconds"] += float(start.time_till(end)) / 1000.0
        entry["calls"] += 1
        return result

    def get_timings(self, reset=False):
        snapshot = {
            name: {"seconds": float(values["seconds"]), "calls": int(values["calls"])}
            for name, values in getattr(self, "_timings", {}).items()
        }
        if reset:
            self._timings = {}
        return snapshot

    def benchmark_snapshot(self, reset=False):
        """Return the selected device policy and optional CUDA stage timings."""
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            return {"timings": self.get_timings(reset=reset)}
        return {
            "matcher": runtime.matcher,
            "matcher_k_chunks": 8 if runtime.matcher == "bmma" else 1,
            "compute_capability": runtime.compute_capability,
            "max_threads_per_block": runtime.max_threads_per_block,
            "shared_memory_per_block": runtime.shared_memory_per_block,
            "multiprocessor_count": runtime.multiprocessor_count,
            "max_grid_z": runtime.max_grid_z,
            "total_memory": runtime.total_memory,
            "partial_bytes": runtime.partial_bytes,
            "timings": self.get_timings(reset=reset),
        }

    def _tc_ok(self):
        return int(self.number_of_patches) == 1

    def _pack_slots(self, class_ids, banks, weights, clause_active):
        n_slots = int(max(class_ids)) + 1
        n_clauses = int(self.number_of_clauses)
        ta_ptrs = np.zeros(n_slots, dtype=np.uint64)
        packed_w = np.zeros((n_slots, n_clauses), dtype=np.int32)
        packed_active = np.ones((n_slots, n_clauses), dtype=np.int32)
        active = np.asarray(clause_active)
        for row, class_id in enumerate(class_ids):
            ta_ptrs[int(class_id)] = np.uint64(int(banks[row]._ta_gpu))
            packed_w[int(class_id)] = np.asarray(weights[row], dtype=np.int32)
            if active.ndim == 2:
                packed_active[int(class_id)] = np.asarray(active[row], dtype=np.int32)
            elif active.size == n_clauses:
                packed_active[int(class_id)] = np.asarray(active, dtype=np.int32).reshape(-1)
        return n_slots, ta_ptrs, packed_w, packed_active

    def _mask_bytes(self, n_slots):
        # One bit per literal; both BMMA and the INT8 fallback consume this form.
        return int(n_slots) * int(self.number_of_clauses) * int(self.number_of_ta_chunks) * 4

    def _extract(
            self,
            ptrs_gpu,
            lit_gpu,
            masks_gpu,
            counts_gpu,
            included_counts_gpu,
            n_slots,
            apply_literal,
    ):
        total = int(n_slots) * int(self.number_of_clauses)
        threads = 128
        warps = threads // 32
        blocks = max(1, (total + warps - 1) // warps)
        self._time_call(
            "extract",
            self._ex_fn.prepared_call,
            (blocks, 1, 1),
            (threads, 1, 1),
            ptrs_gpu,
            lit_gpu,
            masks_gpu,
            counts_gpu,
            included_counts_gpu,
            int(self.number_of_clauses),
            int(n_slots),
            int(self.number_of_ta_chunks),
            int(self.number_of_state_bits_ta),
            int(self._filter) & 0xFFFFFFFF,
            int(apply_literal),
        )

    def _imma(self, x_gpu, masks_gpu, counts_gpu, w_gpu, fired_gpu, sums_gpu, n, n_slots, reject_empty, write_fired):
        n = int(n)
        n_slots = int(n_slots)
        n_clauses = int(self.number_of_clauses)
        chunks = int(self.number_of_ta_chunks)
        panels = (n_clauses + 7) // 8
        if n <= 0 or n_slots <= 0 or panels <= 0:
            return
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            tile_rows, threads = _tile_for(n)
        else:
            tile_rows, threads = runtime.match_launch(n, panels, n_slots)
        row_tiles = (n + tile_rows - 1) // tile_rows
        bytes_per_panel = n * n_slots * 4
        match_fn = self._bmma_fn if runtime is not None and runtime.matcher == "bmma" else self._imma_fn
        timing_name = "match_bmma" if runtime is not None and runtime.matcher == "bmma" else "match_imma"
        partial_budget = runtime.partial_bytes if runtime is not None else _PARTIAL_BYTES
        max_grid_z = runtime.max_grid_z if runtime is not None else 65535
        group_cap = min(max_grid_z, max(1, partial_budget // max(bytes_per_panel, 1)))
        partial = self._tc_buf("partial", min(panels, group_cap) * bytes_per_panel)
        done = 0
        while done < panels:
            group = min(group_cap, panels - done)
            self._time_call(
                timing_name,
                match_fn.prepared_call,
                (row_tiles, n_slots, group),
                (threads, 1, 1),
                x_gpu,
                masks_gpu,
                counts_gpu,
                w_gpu,
                fired_gpu,
                partial,
                n,
                n_clauses,
                n_slots,
                chunks,
                int(reject_empty),
                int(write_fired),
                int(done),
            )
            total = n * n_slots
            self._time_call(
                "reduce",
                self._reduce_fn.prepared_call,
                ((total + 255) // 256, 1, 1),
                (256, 1, 1),
                partial,
                sums_gpu,
                n,
                n_slots,
                int(group),
                int(done > 0),
            )
            done += group

    def imma_match(self, encoded, weights, reject_empty=True, literal_active=None):
        """Clause outputs and weighted sums using the runtime-selected matcher."""
        self._require_imma()
        _ensure_context()
        self._push()
        encoded = np.ascontiguousarray(encoded, dtype=np.uint32)
        n = int(encoded.shape[0])
        n_clauses = int(self.number_of_clauses)
        chunks = int(self.number_of_ta_chunks)
        weights = np.ascontiguousarray(weights, dtype=np.int32).reshape(-1)
        if n == 0:
            return (
                np.empty((0, n_clauses), dtype=np.uint8),
                np.empty(0, dtype=np.int32),
            )
        ta_ptrs = np.zeros(1, dtype=np.uint64)
        ta_ptrs[0] = np.uint64(int(self._ta_gpu))
        if literal_active is None:
            literal = np.full(chunks, np.uint32(0xFFFFFFFF))
            apply_literal = 0
        else:
            literal = np.ascontiguousarray(literal_active, dtype=np.uint32).reshape(-1)
            apply_literal = 1
        ptrs_gpu = self._tc_buf("ptrs1", ta_ptrs.nbytes)
        cuda.memcpy_htod(ptrs_gpu, ta_ptrs)
        lit_gpu = self._tc_buf("lit", literal.nbytes)
        cuda.memcpy_htod(lit_gpu, np.ascontiguousarray(literal))
        masks_gpu = self._tc_buf("masks1", self._mask_bytes(1))
        counts_gpu = self._tc_buf("counts1", n_clauses * 4)
        included_counts_gpu = self._tc_buf("included_counts1", n_clauses * 4)
        self._extract(
            ptrs_gpu,
            lit_gpu,
            masks_gpu,
            counts_gpu,
            included_counts_gpu,
            1,
            apply_literal,
        )
        w_gpu = self._tc_buf("w1", weights.nbytes)
        cuda.memcpy_htod(w_gpu, weights)
        x_gpu = self._upload_x(encoded)
        fired = np.empty((n_clauses, n), dtype=np.uint8)
        sums = np.empty(n, dtype=np.int32)
        fired_gpu = self._tc_buf("fired1", fired.nbytes)
        sums_gpu = self._tc_buf("sums1", max(sums.nbytes, 4))
        self._imma(x_gpu, masks_gpu, counts_gpu, w_gpu, fired_gpu, sums_gpu, n, 1, int(reject_empty), 1)
        _CTX.synchronize()
        cuda.memcpy_dtoh(fired, fired_gpu)
        cuda.memcpy_dtoh(sums, sums_gpu)
        return np.ascontiguousarray(fired.T), sums

    def calculate_clause_outputs_predict(self, encoded_X, e):
        row = np.ascontiguousarray(encoded_X[e], dtype=np.uint32).reshape(1, -1)
        fired, _sums = self.imma_match(
            row, np.ones(self.number_of_clauses, dtype=np.int32), reject_empty=True)
        return np.ascontiguousarray(fired.reshape(-1), dtype=np.uint32)

    def calculate_clause_outputs_update(self, literal_active, encoded_X, e):
        row = np.ascontiguousarray(encoded_X[e], dtype=np.uint32).reshape(1, -1)
        fired, _sums = self.imma_match(
            row,
            np.ones(self.number_of_clauses, dtype=np.int32),
            reject_empty=False,
            literal_active=literal_active,
        )
        return np.ascontiguousarray(fired.reshape(-1), dtype=np.uint32)

    def run_epoch(
            self,
            class_ids,
            banks,
            weights,
            encoded,
            sample_index,
            targets,
            not_targets,
            clause_active,
            literal_active,
            T,
            type_i_p,
            type_ii_p,
            weighted,
            do_type_i,
            do_type_ii,
            confidence_driven=False,
            collect_update_p=False,
    ):
        self._require_imma()
        _ensure_context()
        if int(T) <= 0:
            raise ValueError("T must be greater than zero for TensorCore training")
        for bank in banks:
            bank._push()
        n_clauses = int(self.number_of_clauses)
        chunks = int(self.number_of_ta_chunks)
        n_slots, ta_ptrs, packed_w, packed_active = self._pack_slots(class_ids, banks, weights, clause_active)
        literal = np.ascontiguousarray(literal_active, dtype=np.uint32).reshape(-1)
        ptrs_gpu = self._tc_buf("ptrs", ta_ptrs.nbytes)
        cuda.memcpy_htod(ptrs_gpu, ta_ptrs)
        w_gpu = self._tc_buf("weights", packed_w.nbytes)
        cuda.memcpy_htod(w_gpu, packed_w)
        act_gpu = self._tc_buf("active", packed_active.nbytes)
        cuda.memcpy_htod(act_gpu, np.ascontiguousarray(packed_active))
        lit_gpu = self._tc_buf("lit", literal.nbytes)
        cuda.memcpy_htod(lit_gpu, literal)
        x_gpu = self._upload_x(np.ascontiguousarray(encoded, dtype=np.uint32))
        masks_gpu = self._tc_buf("masks", self._mask_bytes(n_slots))
        counts_gpu = self._tc_buf("counts", n_slots * n_clauses * 4)
        included_counts_gpu = self._tc_buf("included_counts", n_slots * n_clauses * 4)
        sample_index = np.ascontiguousarray(sample_index, dtype=np.int32)
        targets = np.ascontiguousarray(targets, dtype=np.int32)
        not_targets = np.ascontiguousarray(not_targets, dtype=np.int32)
        target_update_p = np.empty(len(sample_index), dtype=np.float32) if collect_update_p else None
        not_target_update_p = np.full(len(sample_index), np.nan, dtype=np.float32) if collect_update_p else None
        inv_s = 1.0 if float(self.s) <= 0 else 1.0 / float(self.s)
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            feedback_batch_size = min(self.batch_size, _MAX_FEEDBACK_BATCH, _MASK_BATCH)
        else:
            feedback_batch_size = min(
                self.batch_size,
                runtime.feedback_launch(1, chunks).batch_limit,
                _MASK_BATCH,
            )
        for start in range(0, len(sample_index), feedback_batch_size):
            stop = min(start + feedback_batch_size, len(sample_index))
            n = stop - start
            idx = np.ascontiguousarray(sample_index[start:stop])
            tgt = np.ascontiguousarray(targets[start:stop])
            neg = np.ascontiguousarray(not_targets[start:stop])
            idx_gpu = self._tc_buf("idx", idx.nbytes)
            tgt_gpu = self._tc_buf("tgt", tgt.nbytes)
            neg_gpu = self._tc_buf("neg", neg.nbytes)
            cuda.memcpy_htod(idx_gpu, idx)
            cuda.memcpy_htod(tgt_gpu, tgt)
            cuda.memcpy_htod(neg_gpu, neg)
            batch_x = self._tc_buf("batch_x", n * chunks * 4)
            self._time_call(
                "gather",
                self._gather_fn.prepared_call,
                ((n * chunks + 255) // 256, 1, 1),
                (256, 1, 1),
                x_gpu, idx_gpu, batch_x, int(n), chunks,
            )
            self._extract(
                ptrs_gpu,
                lit_gpu,
                masks_gpu,
                counts_gpu,
                included_counts_gpu,
                n_slots,
                1,
            )
            fired_gpu = self._tc_buf("fired", n_slots * n_clauses * n)
            sums = np.empty((n, n_slots), dtype=np.int32)
            sums_gpu = self._tc_buf("sums", sums.nbytes)
            self._imma(batch_x, masks_gpu, counts_gpu, w_gpu, fired_gpu, sums_gpu, n, n_slots, 0, 1)
            if collect_update_p:
                _CTX.synchronize()
                cuda.memcpy_dtoh(sums, sums_gpu)
                clipped = np.clip(sums, -int(T), int(T))
                rows = np.arange(n)
                target_sums = clipped[rows, tgt]
                if confidence_driven:
                    target_update_p[start:stop] = (int(T) - np.abs(target_sums)) / float(T)
                else:
                    target_update_p[start:stop] = (int(T) - target_sums) / float(2 * int(T))
                valid_not_target = neg >= 0
                if np.any(valid_not_target):
                    negative_sums = clipped[rows[valid_not_target], neg[valid_not_target]]
                    if confidence_driven:
                        probabilities = (int(T) - np.abs(negative_sums)) / float(T)
                    else:
                        probabilities = (int(T) + negative_sums) / float(2 * int(T))
                    not_target_update_p[start + rows[valid_not_target]] = probabilities
            if runtime is None:
                launch = FeedbackLaunch(
                    threads=min(256, max(32, ((chunks + 31) // 32) * 32)),
                    shared_bytes=((int(n) + 3) // 4) * 4 + 4,
                    batch_limit=_MAX_FEEDBACK_BATCH,
                )
            else:
                launch = runtime.feedback_launch(n, chunks)
            if launch.shared_bytes > 48 * 1024:
                attribute = getattr(
                    getattr(cuda, "function_attribute", object()),
                    "MAX_DYNAMIC_SHARED_SIZE_BYTES",
                    None,
                )
                if attribute is not None:
                    self._fb_fn.set_attribute(attribute, launch.shared_bytes)
            self._time_call(
                "feedback",
                self._fb_fn.prepared_call,
                launch.grid(n_clauses, n_slots),
                (launch.threads, 1, 1),
                ptrs_gpu,
                w_gpu,
                batch_x,
                fired_gpu,
                sums_gpu,
                tgt_gpu,
                neg_gpu,
                act_gpu,
                lit_gpu,
                included_counts_gpu,
                int(n),
                n_clauses,
                int(n_slots),
                chunks,
                int(self.number_of_state_bits_ta),
                int(T),
                float(inv_s),
                int(self.boost_true_positive_feedback),
                int(self.max_included_literals),
                int(bool(weighted)),
                float(type_i_p),
                float(type_ii_p),
                int(bool(do_type_i)),
                int(bool(do_type_ii)),
                int(bool(confidence_driven)),
                int(self._filter) & 0xFFFFFFFF,
                int(self.rng.randint(1, 2**31 - 1)),
                shared_size=launch.shared_bytes,
            )
        _CTX.synchronize()
        cuda.memcpy_dtoh(packed_w, w_gpu)
        _commit_device_epoch(class_ids, banks, weights, packed_w)
        return {
            "update_p_target": target_update_p,
            "update_p_not_target": not_target_update_p,
        }

    def predict_class_sums(self, class_ids, banks, weights, encoded):
        self._require_imma()
        _ensure_context()
        for bank in banks:
            bank._push()
        encoded = np.ascontiguousarray(encoded, dtype=np.uint32)
        n = int(encoded.shape[0])
        n_clauses = int(self.number_of_clauses)
        chunks = int(self.number_of_ta_chunks)
        n_slots, ta_ptrs, packed_w, _packed_active = self._pack_slots(
            class_ids, banks, weights, np.ones(n_clauses, dtype=np.int32))
        literal = np.full(chunks, np.uint32(0xFFFFFFFF))
        ptrs_gpu = self._tc_buf("ptrs", ta_ptrs.nbytes)
        cuda.memcpy_htod(ptrs_gpu, ta_ptrs)
        w_gpu = self._tc_buf("weights", packed_w.nbytes)
        cuda.memcpy_htod(w_gpu, packed_w)
        lit_gpu = self._tc_buf("lit", literal.nbytes)
        cuda.memcpy_htod(lit_gpu, literal)
        masks_gpu = self._tc_buf("masks", self._mask_bytes(n_slots))
        counts_gpu = self._tc_buf("counts", max(n_slots * n_clauses * 4, 4))
        included_counts_gpu = self._tc_buf("included_counts", max(n_slots * n_clauses * 4, 4))
        self._extract(
            ptrs_gpu,
            lit_gpu,
            masks_gpu,
            counts_gpu,
            included_counts_gpu,
            n_slots,
            0,
        )
        fired_gpu = self._tc_buf("fired_dummy", 4)
        sums = np.empty((n, n_slots), dtype=np.int32)
        if n == 0:
            return sums
        runtime = getattr(self, "_runtime", None)
        row_limit = n if runtime is None else runtime.prediction_batch_rows(
            n, chunks, n_slots)
        for start in range(0, n, row_limit):
            stop = min(start + row_limit, n)
            batch = np.ascontiguousarray(encoded[start:stop])
            x_gpu = self._upload_x(batch)
            batch_sums = np.empty((stop - start, n_slots), dtype=np.int32)
            sums_gpu = self._tc_buf("sums", batch_sums.nbytes)
            self._imma(
                x_gpu,
                masks_gpu,
                counts_gpu,
                w_gpu,
                fired_gpu,
                sums_gpu,
                stop - start,
                n_slots,
                1,
                0,
            )
            _CTX.synchronize()
            cuda.memcpy_dtoh(batch_sums, sums_gpu)
            sums[start:stop] = batch_sums
        return sums
