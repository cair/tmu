import logging
import argparse
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    data = MNIST().get()

    # Reshape MNist from 768 to 28x28
    data["x_train"] = data["x_train"].reshape(-1, 28, 28)
    data["x_test"] = data["x_test"].reshape(-1, 28, 28)

    tm = TMClassifier(
        args.num_clauses,
        args.T,
        args.s,
        patch_dim=tuple(args.patch_dim),
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(data["x_train"], data["y_train"])
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
        experiment_results["train_time"].append(benchmark2.elapsed())
        experiment_results["accuracy"].append(result)

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=8000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--platform", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--patch_dim", nargs="+", default=(10, 10), type=int)
    args = parser.parse_args()

    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)

    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
