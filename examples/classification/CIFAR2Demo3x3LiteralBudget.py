import argparse
import json
import logging
from multiprocessing import get_context

from keras.datasets import cifar10
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
import ssl
from tmu.tools import BenchmarkTimer

ssl._create_default_https_context = ssl._create_unverified_context

_LOGGER = logging.getLogger(__name__)


def preprocess_cifar10_data(resolution, animals):
    """
    Preprocess CIFAR-10 images and labels.

    Parameters:
    - resolution: The number of bins to quantize the pixel values into.
    - animals: An array of CIFAR-10 label indices to be considered as positive samples (1), with all others as negative samples (0).

    Returns:
    - X_train, Y_train: Processed training images and labels.
    - X_test, Y_test: Processed testing images and labels.
    """
    # Load CIFAR-10 data
    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

    # Flatten Y arrays
    Y_train, Y_test = Y_train.reshape(-1), Y_test.reshape(-1)

    # Initialize empty arrays for quantized images
    X_train = np.empty(
        (X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
        dtype=np.uint8)
    X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution),
                      dtype=np.uint8)

    # Quantize pixel values
    for z in range(resolution):
        threshold = (z + 1) * 255 / (resolution + 1)
        X_train[:, :, :, :, z] = X_train_org >= threshold
        X_test[:, :, :, :, z] = X_test_org >= threshold

    # Reshape quantized images and convert to uint32
    X_train = X_train.reshape(
        (X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3 * resolution)).astype(np.uint32)
    X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3 * resolution)).astype(
        np.uint32)

    # Process labels: 1 for animals, 0 for others
    Y_train = np.where(np.isin(Y_train, animals), 1, 0).astype(np.uint32)
    Y_test = np.where(np.isin(Y_test, animals), 1, 0).astype(np.uint32)

    return X_train, Y_train, X_test, Y_test


def metrics(args):
    return dict(
        accuracy=[],
        accuracy_train=[],
        number_of_includes=[],
        train_time=[],
        test_time=[],
        str_result=[],
        args=vars(args)
    )

def run_ensemble(ensemble_params):
    # Unpack parameters
    args, X_train, Y_train, X_test, Y_test, ensemble = ensemble_params

    T = int(args.clauses * 0.75)
    tm = TMClassifier(
        args.clauses,
        T,
        args.s,
        platform=args.platform,
        patch_dim=(args.patch_size, args.patch_size),
        number_of_state_bits_ta=args.number_of_state_bits_ta,
        weighted_clauses=args.weighted_clauses,
        literal_drop_p=args.literal_drop_p,
        max_included_literals=args.max_included_literals
    )

    ensemble_results = metrics(args)

    for epoch in range(args.epochs):
        # Benchmark training time
        benchmark1 = BenchmarkTimer()
        with benchmark1:
            tm.fit(X_train, Y_train)
        ensemble_results["train_time"].append(benchmark1.elapsed())

        # Benchmark testing time and calculate accuracy
        benchmark2 = BenchmarkTimer()
        with benchmark2:
            accuracy_test = 100 * (tm.predict(X_test) == Y_test).mean()
        ensemble_results["test_time"].append(benchmark2.elapsed())
        ensemble_results["accuracy"].append(accuracy_test)

        # Calculate additional metrics
        accuracy_train = 100 * (tm.predict(X_train) == Y_train).mean()
        ensemble_results["accuracy_train"].append(accuracy_train)

        number_of_includes = 0
        for i in range(2):
            for j in range(args.clauses):
                number_of_includes += tm.number_of_include_actions(i, j)
        number_of_includes /= 2 * args.clauses
        ensemble_results["number_of_includes"].append(number_of_includes)

        out_str = "%d %d %.2f %.2f %.2f %.2f %.2f" % (
            ensemble, epoch, number_of_includes, accuracy_train, accuracy_test, benchmark1.elapsed(),
            benchmark2.elapsed())
        ensemble_results["str_result"].append(out_str)
        _LOGGER.info(out_str)
    return ensemble, ensemble_results

def run_ensemble_no_mp(ensemble_params):
    """A wrapper to call `run_ensemble` without multiprocessing."""
    # Just directly call run_ensemble function without any multiprocessing overhead
    return run_ensemble(ensemble_params)


def main(args):
    animals = np.array([2, 3, 4, 5, 6, 7])
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_data(args.resolution, animals)

    ensemble_params = [(args, X_train, Y_train, X_test, Y_test, ensemble) for ensemble in range(args.ensembles)]

    # Check if multiprocessing should be used
    if args.use_multiprocessing:
        with get_context("spawn").Pool(processes=2) as pool:
            results = pool.map(run_ensemble, ensemble_params)
    else:
        # Run ensembles sequentially without multiprocessing
        results = list(map(run_ensemble_no_mp, ensemble_params))

    experiment_results = {}
    for ensemble, ensemble_results in results:
        experiment_results[f'ensemble_{ensemble}'] = ensemble_results

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_included_literals", type=int, default=32)
    parser.add_argument("--clauses", type=int, default=8000)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--platform", type=str, default="GPU")
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--number_of_state_bits_ta", type=int, default=8)
    parser.add_argument("--literal_drop_p", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--ensembles", type=int, default=5)
    parser.add_argument("--weighted-clauses", type=bool, default=True)
    parser.add_argument("--use_multiprocessing", action='store_true', help="Use multiprocessing to run ensembles in parallel")
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
