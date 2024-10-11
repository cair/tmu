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
    Y_train = np.random.randint(0, 4, size=(args.examples), dtype=np.uint32)
    for i in range(args.examples):
        position_1 = np.random.randint(0, args.sequence_length-4)
        position_2 = position_1+1
        position_3 = position_1+2
        position_4 = position_1+3
        position_5 = position_1+4

        #position_1 = np.random.randint(0, args.sequence_length-4)
        #position_2 = position_1+2
        #position_3 = position_1+4
        
        if Y_train[i] == 0:
            X_train[i,0,position_1,0] = 1
            X_train[i,0,position_1,1] = 0

            X_train[i,0,position_2,0] = 1
            X_train[i,0,position_2,1] = 0

            X_train[i,0,position_3,0] = 1
            X_train[i,0,position_3,1] = 0

            X_train[i,0,position_4,0] = 1
            X_train[i,0,position_4,1] = 0

            X_train[i,0,position_5,0] = 0
            X_train[i,0,position_5,1] = 0
        elif Y_train[i] == 1:
            X_train[i,0,position_1,0] = 1
            X_train[i,0,position_1,1] = 0

            X_train[i,0,position_2,0] = 1
            X_train[i,0,position_2,1] = 0

            X_train[i,0,position_3,0] = 1
            X_train[i,0,position_3,1] = 0

            X_train[i,0,position_4,0] = 0
            X_train[i,0,position_4,1] = 0

            X_train[i,0,position_5,0] = 0
            X_train[i,0,position_5,1] = 0
        elif Y_train[i] == 2:
            X_train[i,0,position_1,0] = 1
            X_train[i,0,position_1,1] = 0

            X_train[i,0,position_2,0] = 1
            X_train[i,0,position_2,1] = 0

            X_train[i,0,position_3,0] = 0
            X_train[i,0,position_3,1] = 0

            X_train[i,0,position_4,0] = 0
            X_train[i,0,position_4,1] = 0

            X_train[i,0,position_5,0] = 0
            X_train[i,0,position_5,1] = 0
        else:
            X_train[i,0,position_1,0] = 1
            X_train[i,0,position_1,1] = 0

            X_train[i,0,position_2,0] = 0
            X_train[i,0,position_2,1] = 0

            X_train[i,0,position_3,0] = 0
            X_train[i,0,position_3,1] = 0

            X_train[i,0,position_4,0] = 0
            X_train[i,0,position_4,1] = 0

            X_train[i,0,position_5,0] = 0
            X_train[i,0,position_5,1] = 0

        if np.random.rand() <= args.noise:
            Y_train[i] = np.random.choice(np.setdiff1d([0,1,2,3], [Y_train[i]]))

    X_test = np.zeros((args.examples//10, 1, args.sequence_length, 2), dtype=np.uint32)
    Y_test = np.random.randint(0, 4, size=(args.examples//10), dtype=np.uint32)
    for i in range(args.examples//10):
        position_1 = np.random.randint(0, args.sequence_length-4)
        position_2 = position_1+1
        position_3 = position_1+2
        position_4 = position_1+3
        position_5 = position_1+4

        #position_1 = np.random.randint(0, args.sequence_length-4)
        #position_2 = position_1+2
        #position_3 = position_1+4
        
        if Y_test[i] == 0:
            X_test[i,0,position_1,0] = 1
            X_test[i,0,position_1,1] = 0

            X_test[i,0,position_2,0] = 1
            X_test[i,0,position_2,1] = 0

            X_test[i,0,position_3,0] = 1
            X_test[i,0,position_3,1] = 0

            X_test[i,0,position_4,0] = 1
            X_test[i,0,position_4,1] = 0

            X_test[i,0,position_5,0] = 0
            X_test[i,0,position_5,1] = 0
        elif Y_test[i] == 1:
            X_test[i,0,position_1,0] = 1
            X_test[i,0,position_1,1] = 0

            X_test[i,0,position_2,0] = 1
            X_test[i,0,position_2,1] = 0

            X_test[i,0,position_3,0] = 1
            X_test[i,0,position_3,1] = 0

            X_test[i,0,position_4,0] = 0
            X_test[i,0,position_4,1] = 0

            X_test[i,0,position_5,0] = 0
            X_test[i,0,position_5,1] = 0
        elif Y_test[i] == 2:
            X_test[i,0,position_1,0] = 1
            X_test[i,0,position_1,1] = 0

            X_test[i,0,position_2,0] = 1
            X_test[i,0,position_2,1] = 0

            X_test[i,0,position_3,0] = 0
            X_test[i,0,position_3,1] = 0

            X_test[i,0,position_4,0] = 0
            X_test[i,0,position_4,1] = 0

            X_test[i,0,position_5,0] = 0
            X_test[i,0,position_5,1] = 0
        else:
            X_test[i,0,position_1,0] = 1
            X_test[i,0,position_1,1] = 0

            X_test[i,0,position_2,0] = 0
            X_test[i,0,position_2,1] = 0

            X_test[i,0,position_3,0] = 0
            X_test[i,0,position_3,1] = 0

            X_test[i,0,position_4,0] = 0
            X_test[i,0,position_4,1] = 0

            X_test[i,0,position_5,0] = 0
            X_test[i,0,position_5,1] = 0

    tm = TMClassifier(args.number_of_clauses, args.T, args.s, number_of_state_bits_ta=args.number_of_state_bits_ta, depth=args.depth, patch_dim=(1, 1), weighted_clauses=True, platform=args.platform, boost_true_positive_feedback=True, spatio_temporal=True, incremental=False, max_included_literals=args.max_included_literals)

    for i in range(args.epochs):
        tm.fit(X_train, Y_train)

        temporal_features = tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches*2

        position_features = (tm.clause_banks[0].dim[0] - tm.clause_banks[0].patch_dim[0]) + (tm.clause_banks[0].dim[1] - tm.clause_banks[0].patch_dim[1])

        np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

        # print("\nClass 0 Positive Clauses:\n")

        # the_class = 0
        # polarity = 0

        # precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
        # recall = tm.clause_recall(the_class, polarity, X_test, Y_test)

        # for j in range(args.number_of_clauses // 2):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features):
        #         if k in range(tm.clause_banks[0].number_of_clauses):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JB%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JB%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*2):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*2, tm.clause_banks[0].number_of_clauses*3):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*3, tm.clause_banks[0].number_of_clauses*4):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*4, tm.clause_banks[0].number_of_clauses*5):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*5, tm.clause_banks[0].number_of_clauses*6):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses*6 + 2*tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TCB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TCB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features, temporal_features + position_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" POS%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬POS%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features + position_features, tm.clause_banks[0].number_of_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" x%d(%d)" % (k  - temporal_features - position_features, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬x%d(%d)" % (k - temporal_features - position_features, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         else:
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" U%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬U%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
           
        #     print(" ∧ ".join(l))
        # print()

        # print("\nClass 0 Negative Clauses:\n")

        # the_class = 0
        # polarity = 1

        # precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
        # recall = tm.clause_recall(the_class, polarity, X_test, Y_test)

        # for j in range(args.number_of_clauses // 2):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j + args.number_of_clauses // 2, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features):
        #         if k in range(tm.clause_banks[0].number_of_clauses):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JB%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JB%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*2):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*2, tm.clause_banks[0].number_of_clauses*3):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*3, tm.clause_banks[0].number_of_clauses*4):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*4, tm.clause_banks[0].number_of_clauses*5):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*5, tm.clause_banks[0].number_of_clauses*6):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses*6 + 2*tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TCB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TCB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features, temporal_features + position_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" POS%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬POS%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features + position_features, tm.clause_banks[0].number_of_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" x%d(%d)" % (k  - temporal_features - position_features, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬x%d(%d)" % (k - temporal_features - position_features, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         else:
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" U%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬U%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
           
        #     print(" ∧ ".join(l))
        # print()

        # print("\nClass 1 Positive Clauses:\n")

        # the_class = 1
        # polarity = 0

        # precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
        # recall = tm.clause_recall(the_class, polarity, X_test, Y_test)

        # for j in range(args.number_of_clauses // 2):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features):
        #         if k in range(tm.clause_banks[0].number_of_clauses):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JB%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JB%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*2):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*2, tm.clause_banks[0].number_of_clauses*3):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*3, tm.clause_banks[0].number_of_clauses*4):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*4, tm.clause_banks[0].number_of_clauses*5):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*5, tm.clause_banks[0].number_of_clauses*6):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses*6 + 2*tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TCB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TCB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features, temporal_features + position_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" POS%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬POS%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features + position_features, tm.clause_banks[0].number_of_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" x%d(%d)" % (k  - temporal_features - position_features, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬x%d(%d)" % (k - temporal_features - position_features, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         else:
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" U%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬U%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
        #     print(" ∧ ".join(l))
        # print()

        # print("\nClass 1 Negative Clauses:\n")

        # the_class = 1
        # polarity = 1

        # precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
        # recall = tm.clause_recall(the_class, polarity, X_test, Y_test)

        # for j in range(args.number_of_clauses // 2):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j + args.number_of_clauses // 2, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features):
        #         if k in range(tm.clause_banks[0].number_of_clauses):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JB%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JB%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*2):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*2, tm.clause_banks[0].number_of_clauses*3):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*3, tm.clause_banks[0].number_of_clauses*4):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*4, tm.clause_banks[0].number_of_clauses*5):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*5, tm.clause_banks[0].number_of_clauses*6):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses*6 + 2*tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TCB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TCB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
              
        #         elif k in range(temporal_features, temporal_features + position_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" POS%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬POS%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features + position_features, tm.clause_banks[0].number_of_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" x%d(%d)" % (k  - temporal_features - position_features, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬x%d(%d)" % (k - temporal_features - position_features, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         else:
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" U%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬U%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
        #     print(" ∧ ".join(l))
        # print()

        # print("\nClass 2 Positive Clauses:\n")

        # the_class = 2
        # polarity = 0

        # precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
        # recall = tm.clause_recall(the_class, polarity, X_test, Y_test)

        # for j in range(args.number_of_clauses // 2):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(2, 0, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features):
        #         if k in range(tm.clause_banks[0].number_of_clauses):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JB%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JB%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*2):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*2, tm.clause_banks[0].number_of_clauses*3):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*3, tm.clause_banks[0].number_of_clauses*4):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*4, tm.clause_banks[0].number_of_clauses*5):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*5, tm.clause_banks[0].number_of_clauses*6):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses*6 + 2*tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TCB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TCB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
               
        #         elif k in range(temporal_features, temporal_features + position_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" POS%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬POS%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features + position_features, tm.clause_banks[0].number_of_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" x%d(%d)" % (k  - temporal_features - position_features, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬x%d(%d)" % (k - temporal_features - position_features, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         else:
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" U%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬U%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
        #     print(" ∧ ".join(l))
        # print()

        # print("\nClass 2 Negative Clauses:\n")

        # the_class = 2
        # polarity = 1

        # precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
        # recall = tm.clause_recall(the_class, polarity, X_test, Y_test)

        # for j in range(args.number_of_clauses // 2):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j + args.number_of_clauses // 2, tm.get_weight(2, 1, j), precision[j], recall[j]), end=' ')
        #     l = []
        #     for k in range(tm.clause_banks[0].number_of_features):
        #         if k in range(tm.clause_banks[0].number_of_clauses):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JB%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JB%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*2):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬JA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*2, tm.clause_banks[0].number_of_clauses*3):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬B%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*2, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*3, tm.clause_banks[0].number_of_clauses*4):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬A%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*3, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*4, tm.clause_banks[0].number_of_clauses*5):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*4, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*5, tm.clause_banks[0].number_of_clauses*6):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬OffA%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*5, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses*6 + 2*tm.clause_banks[0].number_of_patches):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" #TCB%d(%d)" % (k  - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬#TCB%d(%d)" % (k - tm.clause_banks[0].number_of_clauses*6 - tm.clause_banks[0].number_of_patches, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features, temporal_features + position_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" POS%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬POS%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         elif k in range(temporal_features + position_features, tm.clause_banks[0].number_of_features):
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" x%d(%d)" % (k  - temporal_features - position_features, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬x%d(%d)" % (k - temporal_features - position_features, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))

        #         else:
        #             if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
        #                 l.append(" U%d(%d)" % (k, tm.get_ta_state(j, k, the_class=the_class, polarity=polarity)))
        #             if tm.get_ta_action(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity):
        #                 l.append("¬U%d(%d)" % (k, tm.get_ta_state(j, k + tm.clause_banks[0].number_of_features, the_class=the_class, polarity=polarity)))
        #     print(" ∧ ".join(l))
        # print()

        #print("\nClause Co-Occurence Matrix:\n")
        #print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

        # print("\nLiteral Frequency:\n")
        # print(tm.literal_clause_frequency())
        # experiment_results["literal_frequency"] = tm.literal_clause_frequency().tolist()

        accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
        experiment_results["accuracy"].append(accuracy)
        print("Epoch: %d Accuracy: %.3f" % (i, accuracy))

        #print(tm.clause_banks[0].number_of_features, tm.clause_banks[0].number_of_patches, tm.clause_banks[0].number_of_clauses, tm.clause_banks[0].number_of_clauses*6 + tm.clause_banks[0].number_of_patches*2 + (tm.clause_banks[0].dim[0] - tm.clause_banks[0].patch_dim[0]) + (tm.clause_banks[0].dim[1] - tm.clause_banks[0].patch_dim[1]))
    return experiment_results

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--number-of-clauses", default=10*2, type=int)
    parser.add_argument("--max-included-literals", default=16, type=int)
    parser.add_argument("--platform", default='CPU', type=str)
    parser.add_argument("--T", default=100*2, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--sequence-length", default=10, type=int)
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--examples", default=40000, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--number-of-state-bits-ta", default=10, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
