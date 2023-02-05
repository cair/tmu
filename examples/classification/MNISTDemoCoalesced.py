import logging
import argparse
import numpy as np
from data import MNIST
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("--num_clauses", default=20000, type=int)
        parser.add_argument("--T", default=5000, type=int)
        parser.add_argument("--s", default=10.0, type=float)
        parser.add_argument("--weighted_clauses", default=True, type=bool)
        parser.add_argument("--epochs", default=60, type=int)
        args = parser.parse_args()

        data = MNIST().get()

        tm = TMCoalescedClassifier(
                number_of_clauses=args.num_clauses,
                T=args.T,
                s=args.s,
                weighted_clauses=args.weighted_clauses
        )

        _LOGGER.info(f"Running {TMCoalescedClassifier} for {args.epochs}")
        for epoch in range(args.epochs):
                benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
                with benchmark1:
                        tm.fit(data["x_train"], data["y_train"])

                benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
                with benchmark2:
                        result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()

                _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                             f"Testing Time: {benchmark2.elapsed():.2f}s")
