import sys
import pathlib
import subprocess
import typing

from tmu.clause_bank.clause_bank import ClauseBank
from tmu.weight_bank import WeightBank

current_dir = pathlib.Path(__file__).parent
cmake_dir = current_dir / ".." / ".." / "cmake-build-release"
root_dir = current_dir / ".." / ".." / ".." / ".."
sys.path.append(str(cmake_dir.absolute()))

import numpy as np
import tmulibpy

# Attempt to import tmu
try:
    import tmu
    from tmu.models.classification.vanilla_classifier import TMClassifier
except ImportError as e:
    print(e)
    print("Attempting to install the package in editable mode from the project's root directory...")
    # Execute pip install command
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(root_dir.absolute()), "-vvv"])
    except subprocess.CalledProcessError as e:
        print("Failed to install the package. Please run 'pip install -e .' manually.")


class TMClassifierCPP(TMClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tm_memory = tmulibpy.TMMemory()
        self.tmpp = tmulibpy.TMVanillaClassifier(
            T=self.T,
            s=self.s,
            max_included_literals=self.max_included_literals,
            d=self.d,
            number_of_clauses=self.number_of_clauses,
            confidence_driven_updating=self.confidence_driven_updating,
            weighted_clauses=self.weighted_clauses,
            type_i_feedback=self.type_i_feedback,
            type_ii_feedback=self.type_ii_feedback,
            type_iii_feedback=self.type_iii_feedback,
            type_i_ii_ratio=self.type_i_ii_ratio,
            seed=self.seed,
        )
        self.clause_banks = self.tmpp.clause_banks
        self.weight_banks = self.tmpp.weight_banks

    def predict(
            self,
            X: np.ndarray,
            clip_class_sum: bool = False,
            return_class_sums: bool = False,
            **kwargs
    ):

        encoded_X_test = self.test_encoder_cache.get_encoded_data(
            data=X,
            encoder_func=lambda x: self.clause_banks[0].prepare_X(x)
        )

        max_classes, class_sums = self.tmpp.predict(
            encoded_X_test,
            clip_class_sum,
            return_class_sums
        )

        max_classes = np.asarray(max_classes)

        if return_class_sums:
            class_sums = np.asarray(class_sums).reshape(-1, self.number_of_classes)
            return max_classes, class_sums
        else:
            return max_classes

    def fit(
            self,
            X: np.ndarray[np.uint32],
            Y: np.ndarray[np.uint32],
            shuffle: bool = True,
            metrics: typing.Optional[list] = None,
            *args,
            **kwargs
    ):
        metrics = metrics or []
        assert X.shape[0] == len(Y), "X and Y must have the same number of samples"
        assert len(X.shape) >= 2, "X must be a 2D array"
        assert len(Y.shape) == 1, "Y must be a 1D array"
        assert X.dtype == np.uint32, "X must be of type uint32"
        assert Y.dtype == np.uint32, "Y must be of type uint32"

        self.init(X, Y)
        self.metrics.clear()

        encoded_X_train: np.ndarray = self.train_encoder_cache.get_encoded_data(
            data=X,
            encoder_func=lambda x: self.clause_banks[0].prepare_X(x)
        )

        self.tmpp.fit(
            y=Y,
            encoded_X_train=encoded_X_train
        )

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        assert self.clause_banks is not None
        assert self.weight_banks is not None
        assert self.tm_memory is not None

        clause_bank_sample = self.clause_banks[0]
        weight_bank_sample = self.weight_banks[0]

        mem_size = 0
        for cls in range(self.number_of_classes):
            try:
                mem_size += clause_bank_sample.get_required_memory_size()
            except AttributeError:
                pass
            try:
                mem_size += weight_bank_sample.get_required_memory_size(self.number_of_clauses)
            except AttributeError:
                pass
        mem_size += self.tmpp.get_required_memory_size()

        self.tm_memory.reserve(mem_size)
        for cls in range(self.number_of_classes):
            try:
                self.weight_banks[cls].initialize(self.tm_memory, self.number_of_clauses)
            except AttributeError:
                pass
            try:
                self.clause_banks[cls].initialize(self.tm_memory)
            except AttributeError:
                pass
        self.tmpp.initialize(self.tm_memory)

        # super().init_after(X, Y)
        self.tmpp.init_after(X, Y)
        self.positive_clauses = self.tmpp.positive_clauses
        self.negative_clauses = self.tmpp.negative_clauses

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type_orig, clause_bank_args_orig = self.build_clause_bank(X=X)

        ##self.clause_banks.set_clause_init(clause_bank_type_orig, clause_bank_args_orig)
        ##self.clause_banks.populate(list(range(self.number_of_classes)))
        # return

        clause_bank_type = tmulibpy.TMClauseBankDense
        clause_bank_args = dict(
            d=clause_bank_args_orig["d"],
            s=clause_bank_args_orig["s"],
            boost_true_positive_feedback=bool(clause_bank_args_orig["boost_true_positive_feedback"]),
            reuse_random_feedback=bool(clause_bank_args_orig["reuse_random_feedback"]),
            X_shape=clause_bank_args_orig["X_shape"],
            patch_dim=clause_bank_args_orig["patch_dim"],
            max_included_literals=clause_bank_args_orig["max_included_literals"],
            number_of_clauses=clause_bank_args_orig["number_of_clauses"],
            number_of_state_bits=clause_bank_args_orig["number_of_state_bits_ta"],
            number_of_state_bits_ind=clause_bank_args_orig["number_of_state_bits_ind"],
            batch_size=clause_bank_args_orig["batch_size"],
            incremental=clause_bank_args_orig["incremental"],
            seed=clause_bank_args_orig["seed"],

        )
        self.clause_banks.set_clause_init(clause_bank_type, clause_bank_args)
        self.clause_banks.populate(list(range(self.number_of_classes)))

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.weight_banks.set_clause_init(tmulibpy.TMWeightBank, dict())
        self.weight_banks.populate(list(range(self.number_of_classes)))


import logging
import argparse
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=5000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="CPU", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()

    data = MNIST().get()

    tm = TMClassifierCPP(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        seed=42,
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    data["x_train"].astype(np.uint32),
                    data["y_train"].astype(np.uint32),
                    metrics=["update_p"],
                )

            # print(res)
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.device == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)
