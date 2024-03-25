import argparse
import logging

from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        class_0_precision_positive=[],
        class_0_recall_positive=[],
        class_0_recall_negative=[],
        class_0_precision_negative=[],
        class_1_precision_positive=[],
        class_1_recall_positive=[],
        class_1_recall_negative=[],
        class_1_precision_negative=[],
        literal_frequency=None,
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    X_train = np.random.randint(0, 2, size=(5000, args.number_of_features), dtype=np.uint32)
    Y_train = np.logical_xor(X_train[:, 0], X_train[:, 1]).astype(dtype=np.uint32)
    Y_train = np.where(np.random.rand(5000) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

    X_test = np.random.randint(0, 2, size=(5000, args.number_of_features), dtype=np.uint32)
    Y_test = np.logical_xor(X_test[:, 0], X_test[:, 1]).astype(dtype=np.uint32)

    tm = TMClassifier(args.number_of_clauses, args.T, args.s, platform=args.platform, boost_true_positive_feedback=0)

    for i in range(20):
        tm.fit(X_train, Y_train)
        accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
        experiment_results["accuracy"].append(accuracy)
        print("Accuracy:", accuracy)

    np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

    print("\nClass 0 Positive Clauses:\n")
    precision = tm.clause_precision(0, 0, X_test, Y_test)
    recall = tm.clause_recall(0, 0, X_test, Y_test)
    experiment_results["class_0_precision_positive"].append(list(np.asarray(precision)))
    experiment_results["class_0_recall_positive"].append(list(np.asarray(recall)))

    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(args.number_of_features * 2):
            if tm.get_ta_action(j, k, the_class=0, polarity=0):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=0, polarity=0)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=0, polarity=0)))
        print(" ∧ ".join(l))

    print("\nClass 0 Negative Clauses:\n")

    precision = tm.clause_precision(0, 1, X_test, Y_test)
    recall = tm.clause_recall(0, 1, X_test, Y_test)
    experiment_results["class_0_precision_negative"].append(list(np.asarray(precision)))
    experiment_results["class_0_recall_negative"].append(list(np.asarray(recall)))

    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(args.number_of_features * 2):
            if tm.get_ta_action(j, k, the_class=0, polarity=1):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=0, polarity=1)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=0, polarity=1)))
        print(" ∧ ".join(l))

    print("\nClass 1 Positive Clauses:\n")

    precision = tm.clause_precision(1, 0, X_test, Y_test)
    recall = tm.clause_recall(1, 0, X_test, Y_test)
    experiment_results["class_1_precision_positive"].append(list(np.asarray(precision)))
    experiment_results["class_1_recall_positive"].append(list(np.asarray(recall)))

    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(args.number_of_features * 2):
            if tm.get_ta_action(j, k, the_class=1, polarity=0):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=1, polarity=0)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=1, polarity=0)))
        print(" ∧ ".join(l))

    print("\nClass 1 Negative Clauses:\n")

    precision = tm.clause_precision(1, 1, X_test, Y_test)
    recall = tm.clause_recall(1, 1, X_test, Y_test)
    experiment_results["class_1_precision_negative"].append(list(np.asarray(precision)))
    experiment_results["class_1_recall_negative"].append(list(np.asarray(recall)))

    for j in range(args.number_of_clauses // 2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(args.number_of_features * 2):
            if tm.get_ta_action(j, k, the_class=1, polarity=1):
                if k < args.number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=1, polarity=1)))
                else:
                    l.append("¬x%d(%d)" % (k - args.number_of_features, tm.get_ta_state(j, k, the_class=1, polarity=1)))
        print(" ∧ ".join(l))

    print("\nClause Co-Occurence Matrix:\n")
    print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

    print("\nLiteral Frequency:\n")
    print(tm.literal_clause_frequency())
    experiment_results["literal_frequency"] = tm.literal_clause_frequency().tolist()

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--number-of-clauses", default=10, type=int)
    parser.add_argument("--platform", default='CPU', type=str)
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--s", default=3.0, type=float)
    parser.add_argument("--number-of-features", default=20, type=int)
    parser.add_argument("--noise", default=0.1, type=float, help="Noisy XOR")
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
