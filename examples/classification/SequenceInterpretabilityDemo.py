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

    X_train = np.zeros((args.examples, 1, args.sequence_length, 2), dtype=np.uint32)
    Y_train = np.random.randint(0, 3, size=(args.examples), dtype=np.uint32)
    for i in range(args.examples):
        position_1 = np.random.randint(0, args.sequence_length-2)
        position_2 = position_1+1
        position_3 = position_1+2

        #position_2 = np.random.randint(position_1+1, args.sequence_length-1)
        #position_3 = np.random.randint(position_2+1, args.sequence_length)
        
        if Y_train[i] == 0:
            X_train[i,0,position_1,0] = 1
            X_train[i,0,position_1,1] = 0

            X_train[i,0,position_2,0] = 0
            X_train[i,0,position_2,1] = 1

            X_train[i,0,position_3,0] = 0
            X_train[i,0,position_3,1] = 1
        elif Y_train[i] == 1:
            X_train[i,0,position_1,0] = 0
            X_train[i,0,position_1,1] = 1

            X_train[i,0,position_2,0] = 1
            X_train[i,0,position_2,1] = 0

            X_train[i,0,position_3,0] = 0
            X_train[i,0,position_3,1] = 1
        else:
            X_train[i,0,position_1,0] = 0
            X_train[i,0,position_1,1] = 1

            X_train[i,0,position_2,0] = 0
            X_train[i,0,position_2,1] = 1

            X_train[i,0,position_3,0] = 1
            X_train[i,0,position_3,1] = 0

        if np.random.rand() <= args.noise:
            Y_train[i] = np.random.choice(np.setdiff1d([0,1,2], [Y_train[i]]))

    X_test = np.zeros((args.examples//10, 1, args.sequence_length, 2), dtype=np.uint32)
    Y_test = np.random.randint(0, 3, size=(args.examples//10), dtype=np.uint32)
    for i in range(args.examples//10):
        position_1 = np.random.randint(0, args.sequence_length-2)
        position_2 = position_1+1
        position_3 = position_1+2

        #position_2 = np.random.randint(position_1+1, args.sequence_length-1)
        #position_3 = np.random.randint(position_2+1, args.sequence_length)
        
        if Y_test[i] == 0:
            X_test[i,0,position_1,0] = 1
            X_test[i,0,position_1,1] = 0

            X_test[i,0,position_2,0] = 0
            X_test[i,0,position_2,1] = 1

            X_test[i,0,position_3,0] = 0
            X_test[i,0,position_3,1] = 1
        elif Y_test[i] == 1:
            X_test[i,0,position_1,0] = 0
            X_test[i,0,position_1,1] = 1

            X_test[i,0,position_2,0] = 1
            X_test[i,0,position_2,1] = 0

            X_test[i,0,position_3,0] = 0
            X_test[i,0,position_3,1] = 1
        else:
            X_test[i,0,position_1,0] = 0
            X_test[i,0,position_1,1] = 1

            X_test[i,0,position_2,0] = 0
            X_test[i,0,position_2,1] = 1

            X_test[i,0,position_3,0] = 1
            X_test[i,0,position_3,1] = 0

    tm = TMClassifier(args.number_of_clauses, args.T, args.s, number_of_state_bits_ta=12, patch_dim=(1, 1), weighted_clauses=True, platform=args.platform, boost_true_positive_feedback=True, spatio_temporal=True, incremental=False, max_included_literals=16)

    for i in range(args.epochs):
        tm.fit(X_train, Y_train)

        accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
        experiment_results["accuracy"].append(accuracy)
        print("Accuracy:", accuracy)

        # np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

        # print("\nClass 0 Positive Clauses:\n")
        # precision = tm.clause_precision(0, 0, X_test, Y_test)
        # recall = tm.clause_recall(0, 0, X_test, Y_test)
        # experiment_results["class_0_precision_positive"].append(list(np.asarray(precision)))
        # experiment_results["class_0_recall_positive"].append(list(np.asarray(recall)))

        # for j in range(args.number_of_clauses // 2):
        #     #if recall[j] == 0:
        #     #    continue

        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features * 2):
        #         if tm.get_ta_action(j, k, the_class=0, polarity=0):
        #             if k < tm.clause_banks[0].number_of_features:
        #                 l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=0, polarity=0)))
        #             else:
        #                 l.append("¬x%d(%d)" % (k - tm.clause_banks[0].number_of_features, tm.get_ta_state(j, k, the_class=0, polarity=0)))
        #     print(" ∧ ".join(l))
        #     print()
        # print("\nClass 0 Negative Clauses:\n")

        # precision = tm.clause_precision(0, 1, X_test, Y_test)
        # recall = tm.clause_recall(0, 1, X_test, Y_test)
        # experiment_results["class_0_precision_negative"].append(list(np.asarray(precision)))
        # experiment_results["class_0_recall_negative"].append(list(np.asarray(recall)))

        # for j in range(args.number_of_clauses // 2):
        #     #if recall[j] == 0:
        #     #    continue
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j + args.number_of_clauses // 2, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features * 2):
        #         if tm.get_ta_action(j, k, the_class=0, polarity=1):
        #             if k < tm.clause_banks[0].number_of_features:
        #                 l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=0, polarity=1)))
        #             else:
        #                 l.append("¬x%d(%d)" % (k - tm.clause_banks[0].number_of_features, tm.get_ta_state(j, k, the_class=0, polarity=1)))
        #     print(" ∧ ".join(l))
        #     print()
        # print("\nClass 1 Positive Clauses:\n")

        # precision = tm.clause_precision(1, 0, X_test, Y_test)
        # recall = tm.clause_recall(1, 0, X_test, Y_test)
        # experiment_results["class_1_precision_positive"].append(list(np.asarray(precision)))
        # experiment_results["class_1_recall_positive"].append(list(np.asarray(recall)))

        # for j in range(args.number_of_clauses // 2):
        #     #if recall[j] == 0:
        #     #    continue

        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features * 2):
        #         if tm.get_ta_action(j, k, the_class=1, polarity=0):
        #             if k < tm.clause_banks[0].number_of_features:
        #                 l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=1, polarity=0)))
        #             else:
        #                 l.append("¬x%d(%d)" % (k - tm.clause_banks[0].number_of_features, tm.get_ta_state(j, k, the_class=1, polarity=0)))
        #     print(" ∧ ".join(l))
        #     print()
        # print("\nClass 1 Negative Clauses:\n")

        # precision = tm.clause_precision(1, 1, X_test, Y_test)
        # recall = tm.clause_recall(1, 1, X_test, Y_test)
        # experiment_results["class_1_precision_negative"].append(list(np.asarray(precision)))
        # experiment_results["class_1_recall_negative"].append(list(np.asarray(recall)))

        # for j in range(args.number_of_clauses // 2):
        #     #if recall[j] == 0:
        #     #    continue

        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j + args.number_of_clauses // 2, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features * 2):
        #         if tm.get_ta_action(j, k, the_class=1, polarity=1):
        #             if k < tm.clause_banks[0].number_of_features:
        #                 l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class=1, polarity=1)))
        #             else:
        #                 l.append("¬x%d(%d)" % (k - tm.clause_banks[0].number_of_features, tm.get_ta_state(j, k, the_class=1, polarity=1)))
        #     print(" ∧ ".join(l))
        #     print()

        # #print("\nClause Co-Occurence Matrix:\n")
        # #print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

        # print("\nLiteral Frequency:\n")
        # print(tm.literal_clause_frequency())
        # experiment_results["literal_frequency"] = tm.literal_clause_frequency().tolist()


        # print(tm.clause_banks[0].number_of_features)
    return experiment_results

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--number-of-clauses", default=10*2, type=int)
    parser.add_argument("--platform", default='CPU', type=str)
    parser.add_argument("--T", default=100*2, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--sequence-length", default=6, type=int)
    parser.add_argument("--noise", default=0.01, type=float, help="Noisy XOR")
    parser.add_argument("--examples", default=40000, type=int, help="Noisy XOR")

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
