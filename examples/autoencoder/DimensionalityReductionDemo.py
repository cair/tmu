import argparse
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        cosine_similarity=[],
        train_time=[],
        recall=[],
        precision=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    print("Number of clauses:", args.num_clauses)

    output_active = np.array([0, 1], dtype=np.uint32)

    X_train = np.random.randint(0, 2, size=(5000, args.number_of_features), dtype=np.uint32)
    X_train[:, 0] = np.where(np.random.rand(5000) <= args.noise, 1 - X_train[:, 1], X_train[:, 1])

    tm = TMAutoEncoder(
        args.num_clauses,
        args.T,
        args.s,
        output_active,
        max_included_literals=args.max_included_literals,
        accumulation=args.accumulation,
        feature_negation=args.feature_negation,
        platform=args.platform,
        output_balancing=args.output_balancing
    )

    print(f"\nAccuracy Over {args.epochs} Epochs:")
    for e in range(args.epochs):

        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, number_of_examples=args.number_of_examples)
        experiment_results["train_time"].append(benchmark1.elapsed())

        print("\nEpoch #%d\n" % (e + 1))

        print("Calculating precision\n")
        precision = []
        for i in range(output_active.shape[0]):
            precision.append(tm.clause_precision(i, True, X_train, number_of_examples=500))
        experiment_results["precision"].append(precision)

        print("Calculating recall\n")
        recall = []
        for i in range(output_active.shape[0]):
            recall.append(tm.clause_recall(i, True, X_train, number_of_examples=500))
        experiment_results["recall"].append(precision)
        print("Clauses\n")

        for j in range(args.num_clauses):
            print("Clause #%d " % (j), end=' ')
            for i in range(output_active.shape[0]):
                print("x%d:W%d:P%.2f:R%.2f " % (i, tm.get_weight(i, j), precision[i][j], recall[i][j]), end=' ')

            l = []
            for k in range(tm.clause_bank.number_of_literals):
                if tm.get_ta_action(j, k) == 1:
                    if k < tm.clause_bank.number_of_features:
                        l.append("x%d(%d)" % (k, tm.clause_bank.get_ta_state(j, k)))
                    else:
                        l.append(
                            "¬x%d(%d)" % (k - tm.clause_bank.number_of_features, tm.clause_bank.get_ta_state(j, k)))
            print(" ∧ ".join(l))

        profile = np.empty((output_active.shape[0], args.num_clauses))
        for i in range(output_active.shape[0]):
            weights = tm.get_weights(i)
            profile[i, :] = np.where(weights >= args.clause_weight_threshold, weights, 0)

        similarity = cosine_similarity(profile)
        experiment_results["cosine_similarity"].append(list(similarity.flatten()))

        print("\nWord Similarity\n")

        for i in range(output_active.shape[0]):
            print("x%d" % (output_active[i]), end=': ')
            sorted_index = np.argsort(-1 * similarity[i, :])
            for j in range(1, output_active.shape[0]):
                print("x%d(%.2f) " % (output_active[sorted_index[j]], similarity[i, sorted_index[j]]), end=' ')
            print()

        print("\nTraining Time: %.2f" % (benchmark1.elapsed()))
        return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-features", default=12, type=int)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--clause-weight-threshold", default=0, type=int)
    parser.add_argument("--number-of-examples", default=2500, type=int)
    parser.add_argument("--accumulation", default=1, type=int)
    parser.add_argument("--num-clauses", default=10, type=bool)
    parser.add_argument("--T", default=8 * 10, type=int)
    parser.add_argument("--s", default=1.5, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--platform", default="CPU", type=str)
    parser.add_argument("--output-balancing", default=0.5, type=float)
    parser.add_argument("--max-included-literals", default=3, type=int)
    parser.add_argument("--feature-negation", default=True, type=bool)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    result = main(default_args())
    _LOGGER.info(result)
