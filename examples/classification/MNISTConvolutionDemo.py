import logging
import argparse
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=8000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()

X_train = np.where(X_train  > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)

tm = TMClassifier(8000, 10000, 5.0, patch_dim=(10, 10), max_included_literals=32, platform='CUDA', weighted_clauses=True)

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer()
        with benchmark1:
            tm.fit(data["x_train"], data["y_train"])

        benchmark2 = BenchmarkTimer()
        with benchmark2:
            result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")
