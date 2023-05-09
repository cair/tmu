import itertools
import numpy as np
import argparse
import logging
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    args = parser.parse_args()

    data = MNIST().get()
    X_train = data["x_train"]
    y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

    tm = TMClassifier(2000, 5000, 10.0, platform='CPU_sparse', weighted_clauses=True, literal_sampling=0.1,
                      max_included_literals=32, feedback_rate_excluded_literals=1, absorbing=100,
                      literal_insertion_state=-1)

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs} epochs")
    for e in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, y_train)

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result_test = 100 * (tm.predict(X_test) == y_test).mean()

        result_train = 100 * (tm.predict(X_train) == y_train).mean()

        absorbed = 0.0
        unallocated = 0
        num_classes = 10
        num_clauses = 200
        for i, j in itertools.product(range(num_classes), range(num_clauses)):
            absorbed += 1.0 - (tm.number_of_include_actions(i, j) + tm.number_of_exclude_actions(i, j)) / (
                    X_train.shape[1] * 2)
            unallocated += tm.number_of_unallocated_literals(i, j)

        absorbed = 100 * absorbed / (num_classes * num_clauses)

        _LOGGER.info(f"Epoch: {e + 1}, Test Accuracy: {result_test:.2f}%, Train Accuracy: {result_train:.2f}%, "
                     f"Absorbed: {absorbed:.2f}%, Unallocated: {unallocated}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")
