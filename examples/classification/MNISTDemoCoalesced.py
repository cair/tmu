import logging
import argparse
from tmu.data import MNIST
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        number_of_positive_clauses=[],
        accuracy=[],
        number_of_includes=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    data = MNIST().get()

    tm = TMCoalescedClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        focused_negative_sampling=args.focused_negative_sampling
    )

    _LOGGER.info(f"Running {TMCoalescedClassifier} for {args.epochs}")
    for epoch in range(args.epochs):

        benchmark1 = BenchmarkTimer()
        with benchmark1:
            tm.fit(data["x_train"], data["y_train"])
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer()
        with benchmark2:
            result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
            experiment_results["accuracy"].append(result)
        experiment_results["test_time"].append(benchmark2.elapsed())

        number_of_positive_clauses = 0
        for i in range(tm.number_of_classes):
            number_of_positive_clauses += (tm.weight_banks[i].get_weights() > 0).sum()
        number_of_positive_clauses /= tm.number_of_classes
        experiment_results["number_of_positive_clauses"].append(number_of_positive_clauses)

        number_of_includes = 0
        for j in range(args.num_clauses):
            number_of_includes += tm.number_of_include_actions(j)
        number_of_includes /= 2 * args.num_clauses
        experiment_results["number_of_includes"].append(number_of_includes)

        _LOGGER.info(
            f"Epoch: {epoch + 1}, "
            f"Accuracy: {result:.2f}, "
            f"Positive clauses: {number_of_positive_clauses}, "
            f"Literals: {number_of_includes}, "
            f"Training Time: {benchmark1.elapsed():.2f}s, "
            f"Testing Time: {benchmark2.elapsed():.2f}s"
        )

    return experiment_results


def default_args(**kwargs):
    default_clauses = 20000 // 5
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clauses", default=default_clauses, type=int)
    parser.add_argument("--T", default=default_clauses // 4, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--weighted-clauses", default=True, type=bool)
    parser.add_argument("--platform", default='CPU', type=str)
    parser.add_argument("--focused-negative-sampling", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
