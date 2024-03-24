import itertools
import argparse
import logging
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        accuracy_train=[],
        absorbed=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    data = MNIST().get()
    X_train = data["x_train"]
    y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]

    tm = TMClassifier(
        args.num_clauses,
        args.T,
        args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        literal_sampling=args.literal_sampling,
        max_included_literals=args.max_included_literals,
        feedback_rate_excluded_literals=args.feedback_rate_excluded_literals,
        absorbing=args.absorbing,
        literal_insertion_state=args.literal_insertion_state
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs} epochs")
    for e in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, y_train)
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result_test = 100 * (tm.predict(X_test) == y_test).mean()
            experiment_results["accuracy"].append(result_test)
        experiment_results["test_time"].append(benchmark2.elapsed())

        result_train = 100 * (tm.predict(X_train) == y_train).mean()
        experiment_results["accuracy_train"].append(result_train)

        absorbed = 0.0
        unallocated = 0
        num_classes = 10
        num_clauses = 200
        for i, j in itertools.product(range(num_classes), range(num_clauses)):
            absorbed += 1.0 - (tm.number_of_include_actions(i, j) + tm.number_of_exclude_actions(i, j)) / (
                    X_train.shape[1] * 2)
            unallocated += tm.number_of_unallocated_literals(i, j)

        absorbed = 100 * absorbed / (num_classes * num_clauses)
        experiment_results["absorbed"].append(absorbed)

        _LOGGER.info(f"Epoch: {e + 1}, Test Accuracy: {result_test:.2f}%, Train Accuracy: {result_train:.2f}%, "
                     f"Absorbed: {absorbed:.2f}%, Unallocated: {unallocated}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")

        return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser(description="Configure and run the TMClassifier.")
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=5000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--platform", default='CPU_sparse', type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--literal_sampling", default=0.1, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--feedback_rate_excluded_literals", default=1, type=int)
    parser.add_argument("--absorbing", default=100, type=int)
    parser.add_argument("--literal_insertion_state", default=-1, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    result = main(default_args())
    _LOGGER.info(result)
