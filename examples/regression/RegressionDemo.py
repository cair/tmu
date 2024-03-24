from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import logging
import argparse
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        rmsd=[],
        train_time=[],
        test_time=[]
    )

def main(args):
    experiment_results = metrics(args)

    california_housing = datasets.fetch_california_housing()
    X = california_housing.data
    Y = california_housing.target

    b = StandardBinarizer(max_bits_per_feature=10)
    X_transformed = b.fit_transform(X)

    tm = TMRegressor(
        args.num_clauses,
        args.T,
        args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses
    )

    tm_results = np.empty(0)

    _LOGGER.info(f"Running RMSD with {TMRegressor} for {args.num_runs}")
    for run in range(args.num_runs):
        X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y)

        for epoch in range(args.epochs):
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)
            experiment_results["train_time"].append(benchmark1.elapsed())

            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                tm_results = np.append(tm_results, np.sqrt(((tm.predict(X_test) - Y_test) ** 2).mean()))
            experiment_results["test_time"].append(benchmark1.elapsed())
            experiment_results["rmsd"].append(tm_results.mean())

            _LOGGER.info(
                f"#{run + 1} - Epoch: {epoch}: RMSD: {tm_results.mean()} +/- {1.96 * tm_results.std() / np.sqrt(run + 1)} ({benchmark1.elapsed()})")


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=1000, type=int)
    parser.add_argument("--T", default=500 * 10, type=int)
    parser.add_argument("--s", default=2.75, type=float)
    parser.add_argument("--platform", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--num-runs", default=25, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
