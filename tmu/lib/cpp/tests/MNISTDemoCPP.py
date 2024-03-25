import sys
import pathlib
import subprocess
import numpy as np

try:
    import tmulibpy
except ImportError as e:
    print("Failed to import tmulibpy. Attempting to import from manually built extension...")
    current_dir = pathlib.Path(__file__).parent
    cmake_dir = current_dir / ".." / ".." / "cmake-build-release"
    root_dir = current_dir / ".." / ".." / ".." / ".."
    sys.path.append(str(cmake_dir.absolute()))
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

    tm = tmulibpy.TMVanillaClassifier(
        T=args.T,
        s=args.s,
        d=200.0,
        number_of_clauses=args.num_clauses,
        confidence_driven_updating=False,
        weighted_clauses=True,
        type_i_feedback=True,
        type_ii_feedback=True,
        type_iii_feedback=False,
        type_i_ii_ratio=1.0,
        max_included_literals=0,
        boost_true_positive_feedback=True,
        reuse_random_feedback=True,
        patch_dim=None,
        number_of_state_bits=8,
        number_of_state_bits_ind=8,
        batch_size=100,
        incremental=True,
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
