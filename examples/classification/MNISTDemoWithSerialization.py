import logging
import argparse
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
import pickle

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        serialize_load_time=[],
        serialize_dump_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    data = MNIST().get()

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(data["x_train"], data["y_train"])
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Saving model to binary")
        with benchmark2:
            dumped = pickle.dumps(tm)
        experiment_results["serialize_dump_time"].append(benchmark2.elapsed())

        benchmark3 = BenchmarkTimer(logger=_LOGGER, text="Loading model from binary")
        with benchmark3:
            tm2 = pickle.loads(dumped)
        experiment_results["serialize_load_time"].append(benchmark3.elapsed())

        benchmark4 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark4:
            result = 100 * (tm2.predict(data["x_test"]) == data["y_test"]).mean()
            experiment_results["accuracy"].append(result)
        experiment_results["test_time"].append(benchmark4.elapsed())

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=50, type=int)
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--platform", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
