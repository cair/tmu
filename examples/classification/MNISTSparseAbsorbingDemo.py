import itertools
import logging
import argparse
import numpy as np
from tqdm import tqdm
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=5000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="CPU_sparse", type=str)
    parser.add_argument("--absorbing-include", default=150, type=int)
    parser.add_argument("--absorbing-exclude", default=90, type=int)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()

    data = MNIST().get()
    X_train = data["x_train"]
    y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]
    num_classes = len(np.unique(y_test))

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        absorbing_include=args.absorbing_include,
        absorbing_exclude=args.absorbing_exclude
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in tqdm(range(args.epochs)):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, y_train)

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(X_test) == y_test).mean()

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")

        absorbed_exclude = 0.0
        absorbed_include = 0.0
        include_counter = 0

        for i, j in itertools.product(range(num_classes), range(args.num_clauses)):
            absorbed_exclude += 1.0 - (tm.number_of_exclude_actions(i, j)) / (
                    X_train.shape[1] * 2 - tm.number_of_absorbed_include_actions(i, j) - tm.number_of_include_actions(i,
                                                                                                                      j))
        absorbed_exclude = 100 * absorbed_exclude / (num_classes * args.num_clauses)

        for i, j in itertools.product(range(num_classes), range(args.num_clauses)):
            if tm.number_of_absorbed_include_actions(i, j) + tm.number_of_include_actions(i, j) > 0:
                absorbed_include += 1.0 * tm.number_of_absorbed_include_actions(i, j) / (
                        tm.number_of_absorbed_include_actions(i, j) + tm.number_of_include_actions(i, j))
                include_counter += 1
            absorbed_include = 100 * absorbed_include / include_counter

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Absorbed Exclude: {absorbed_exclude}, Absorbed "
                     f"Include: {absorbed_include}, Training Time: {benchmark1.elapsed():.2f}s,"
                     f"Testing Time: {benchmark2.elapsed():.2f}s")

