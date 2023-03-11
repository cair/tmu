import logging
import argparse

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
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()

    data = MNIST().get()

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in tqdm(range(args.epochs)):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(data["x_train"], data["y_train"])

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")
