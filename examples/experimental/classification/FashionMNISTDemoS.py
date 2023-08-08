import logging
import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
import numpy as np
from keras.datasets import fashion_mnist
import cv2

percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=5000//100, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=False, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--type_i_ii_ratio", default=1.0, type=float)

    args = parser.parse_args()

    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    Y_train = Y_train
    Y_test = Y_test

    for i in range(X_train.shape[0]):
        X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    for i in range(X_test.shape[0]):
        X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        type_i_ii_ratio=args.type_i_ii_ratio
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=_LOGGER, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)

            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(X_test) == Y_test).mean()

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        #X_train_transformed = tm.transform(X_train).astype(np.float64)
        X_test_transformed = tm.transform(X_test).astype(np.float64)

        #clause_precision = []
        #for i in range(10):
        #    clause_precision.append(tm.clause_precision(i, 0, X_train, Y_train))
        #    X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2] *= clause_precision[-1]
        #    X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses] = 0
        
        #Y_test_predicted = np.argmax(X_test_transformed, axis=1)//args.num_clauses
        #print("Max Accuracy", (Y_test_predicted == Y_test).mean())

        for i in range(10):
            Y_test_predicted = tm.predict(X_test)

            positive_examples = (Y_test==i).sum()
            positive_hits = (np.logical_and(Y_test==i, Y_test == Y_test_predicted)).sum()
            positive_misses = (np.logical_and(Y_test==i, Y_test != Y_test_predicted)).sum()

            negative_examples = (Y_test!=i).sum()
            negative_hits = (np.logical_and(Y_test!=i, Y_test == Y_test_predicted)).sum()
            negative_misses = (np.logical_and(Y_test!=i, Y_test != Y_test_predicted)).sum()

            positive_precision = tm.clause_precision(i, 0, X_test, Y_test)
            positive_recall = tm.clause_recall(i, 0, X_test, Y_test)
            positive_precision_sorted = np.argsort(positive_precision)            
            positive_recall_sorted = np.argsort(positive_recall)

            negative_precision = tm.clause_precision(i, 1, X_test, Y_test)
            negative_recall = tm.clause_recall(i, 1, X_test, Y_test)
            negative_precision_sorted = np.argsort(negative_precision)
            negative_recall_sorted = np.argsort(negative_recall)

            print("\nClass %d positive clauses\n" % (i))

            print("\tPrecision:", end=' ')
            for p in percentiles:
                print("(%.2f %.2f)" % (positive_precision[positive_precision_sorted[int(args.num_clauses//2*p)]], positive_recall[positive_precision_sorted[int(args.num_clauses//2*p)]]), end=' ')
            print()

            print("\tRecall:", end=' ')
            for p in percentiles:
                print("(%.2f %.2f)" % (positive_precision[positive_recall_sorted[int(args.num_clauses//2*p)]], positive_recall[positive_recall_sorted[int(args.num_clauses//2*p)]]), end=' ')
            print()
            
            correct_high_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][:,positive_precision_sorted[int(args.num_clauses//2*0.9):]][Y_test==i,:].sum(axis=1)
            print("\tCorrect high-precision activations per example:",  end=' ')
            for p in percentiles:
                print("%.0f" % (np.percentile(correct_high_precision_activations_per_example, p*100)), end=' ')
            print()

            correct_low_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][:,positive_precision_sorted[:int(args.num_clauses//2*0.1)]][Y_test==i,:].sum(axis=1)
            print("\tCorrect low-precision activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % (np.percentile(correct_low_precision_activations_per_example, p*100)), end=' ')
            print()
            
            correct_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][Y_test==i,:].sum(axis=1)
            print("\tCorrect activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_example, p*100), end= ' ')
            print()

            correct_activations_per_clause = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][Y_test==i,:].sum(axis=0)
            print("\tCorrect activations per clause:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_clause, p*100), end= ' ')
            print()

            correct_activations_per_hit = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][np.logical_and(Y_test==i, Y_test == Y_test_predicted),:].sum(axis=1)
            print("\tCorrect activations per correctly classified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_hit, p*100), end= ' ')
            print()

            correct_activations_per_miss = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][np.logical_and(Y_test==i, Y_test != Y_test_predicted),:].sum(axis=1)
            print("\tCorrect activations per misclassified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_miss, p*100), end= ' ')
            print()
            
            incorrect_high_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][:,negative_precision_sorted[int(args.num_clauses//2*0.9):]][Y_test==i,:].sum(axis=1)
            print("\tIncorrect high-precision activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_high_precision_activations_per_example, p*100), end= ' ')
            print()

            incorrect_low_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][:,negative_precision_sorted[:int(args.num_clauses//2*0.1)]][Y_test==i,:].sum(axis=1)
            print("\tIncorrect low-precision activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_low_precision_activations_per_example, p*100), end= ' ')
            print()

            incorrect_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][Y_test==i,:].sum(axis=1)
            print("\tIncorrect activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_example, p*100), end= ' ')
            print()

            incorrect_activations_per_clause = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][Y_test!=i,:].sum(axis=0)
            print("\tIncorrect activations per clause:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_clause, p*100), end= ' ')
            print()

            incorrect_activations_per_hit = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][np.logical_and(Y_test==i, Y_test==Y_test_predicted),:].sum(axis=1)
            print("\tIncorrect activations per correctly classified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_hit, p*100), end= ' ')
            print()

            incorrect_activations_per_miss = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][np.logical_and(Y_test==i, Y_test!= Y_test_predicted),:].sum(axis=1)
            print("\tIncorrect activations per misclassified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_miss, p*100), end= ' ')
            print()

            total_high_precision_activations_per_example = np.log((correct_high_precision_activations_per_example + 1) / (incorrect_high_precision_activations_per_example + 1))
            total_high_precision_activations_per_example_sorted = np.argsort(total_high_precision_activations_per_example)
            print("\tTotal high-precision activations per example:", end=' ')
            for p in percentiles:
                positive_examples_p = int(positive_examples*p)
                print("(%d vs %d)" % (
                    correct_high_precision_activations_per_example[total_high_precision_activations_per_example_sorted[positive_examples_p]],
                    incorrect_high_precision_activations_per_example[total_high_precision_activations_per_example_sorted[positive_examples_p]]
                    ), end= ' '
                )
            print()
              
            total_low_precision_activations_per_example = np.log((correct_low_precision_activations_per_example + 1) / (incorrect_low_precision_activations_per_example + 1))
            total_low_precision_activations_per_example_sorted = np.argsort(total_low_precision_activations_per_example)
            print("\tTotal low-precision activations per example:", end=' ')
            for p in percentiles:
                positive_examples_p = int(positive_examples*p)
                print("(%d vs %d)" % (
                    correct_low_precision_activations_per_example[total_low_precision_activations_per_example_sorted[positive_examples_p]],
                    incorrect_low_precision_activations_per_example[total_low_precision_activations_per_example_sorted[positive_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_example = np.log((correct_activations_per_example + 1) / (incorrect_activations_per_example + 1))
            total_activations_per_example_sorted = np.argsort(total_activations_per_example)
            print("\tTotal activations per example:", end=' ')
            for p in percentiles:
                positive_examples_p = int(positive_examples*p)
                print("(%d vs %d)" % (
                    correct_activations_per_example[total_activations_per_example_sorted[positive_examples_p]],
                    incorrect_activations_per_example[total_activations_per_example_sorted[positive_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_hit = np.log((correct_activations_per_hit + 1) / (incorrect_activations_per_hit + 1))
            total_activations_per_hit_sorted = np.argsort(total_activations_per_hit)
            print("\tTotal activations per correctly classified example:", end=' ')
            for p in percentiles:
                positive_examples_p = int(positive_hits*p)
                print("(%d vs %d)" % (
                    correct_activations_per_hit[total_activations_per_hit_sorted[positive_examples_p]],
                    incorrect_activations_per_hit[total_activations_per_hit_sorted[positive_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_miss = np.log((correct_activations_per_miss + 1) / (incorrect_activations_per_miss + 1))
            total_activations_per_miss_sorted = np.argsort(total_activations_per_miss)
            print("\tTotal activations per misclassified example:", end=' ')
            for p in percentiles:
                positive_examples_p = int(positive_misses*p)
                print("(%d vs %d)" % (
                    correct_activations_per_miss[total_activations_per_miss_sorted[positive_examples_p]],
                    incorrect_activations_per_miss[total_activations_per_miss_sorted[positive_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_clause = np.log((correct_activations_per_clause + 1) / (incorrect_activations_per_clause + 1))
            total_activations_per_clause_sorted = np.argsort(total_activations_per_clause)
            print("\tTotal activations per clause:", end=' ')
            for p in percentiles:
                index = total_activations_per_clause_sorted[int(args.num_clauses//2*p)]
                print("(%d vs %d %.2f/%.2f)" % (
                        correct_activations_per_clause[index],
                        incorrect_activations_per_clause[index],
                        positive_precision[index],
                        positive_recall[index]
                    ), end= ' '
                )
            print()

            #########

            print("\nClass %d Negative clauses\n" % (i))

            print("\tPrecision:", end=' ')
            for p in percentiles:
                print("(%.2f %.2f)" % (negative_precision[negative_precision_sorted[int(args.num_clauses//2*p)]], negative_recall[negative_precision_sorted[int(args.num_clauses//2*p)]]), end=' ')
            print()

            print("\tRecall:", end=' ')
            for p in percentiles:
                print("(%.2f %.2f)" % (negative_precision[negative_recall_sorted[int(args.num_clauses//2*p)]], negative_recall[negative_recall_sorted[int(args.num_clauses//2*p)]]), end=' ')
            print()
            
            correct_high_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][:,negative_precision_sorted[int(args.num_clauses//2*0.9):]][Y_test!=i,:].sum(axis=1)
            print("\tCorrect high-precision activations per example:",  end=' ')
            for p in percentiles:
                print("%.0f" % (np.percentile(correct_high_precision_activations_per_example, p*100)), end=' ')
            print()

            correct_low_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][:,negative_precision_sorted[:int(args.num_clauses//2*0.1)]][Y_test!=i,:].sum(axis=1)
            print("\tCorrect low-precision activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % (np.percentile(correct_low_precision_activations_per_example, p*100)), end=' ')
            print()
            
            correct_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][Y_test!=i,:].sum(axis=1)
            print("\tCorrect activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_example, p*100), end= ' ')
            print()

            correct_activations_per_clause = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][Y_test!=i,:].sum(axis=0)
            print("\tCorrect activations per clause:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_clause, p*100), end= ' ')
            print()

            correct_activations_per_hit = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][np.logical_and(Y_test!=i, Y_test == Y_test_predicted),:].sum(axis=1)
            print("\tCorrect activations per correctly classified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_hit, p*100), end= ' ')
            print()

            correct_activations_per_miss = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][np.logical_and(Y_test!=i, Y_test != Y_test_predicted),:].sum(axis=1)
            print("\tCorrect activations per misclassified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(correct_activations_per_miss, p*100), end= ' ')
            print()
            
            incorrect_high_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][:,positive_precision_sorted[int(args.num_clauses//2*0.9):]][Y_test!=i,:].sum(axis=1)
            print("\tIncorrect high-precision activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_high_precision_activations_per_example, p*100), end= ' ')
            print()

            incorrect_low_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][:,positive_precision_sorted[:int(args.num_clauses//2*0.1)]][Y_test!=i,:].sum(axis=1)
            print("\tIncorrect low-precision activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_low_precision_activations_per_example, p*100), end= ' ')
            print()

            incorrect_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][Y_test!=i,:].sum(axis=1)
            print("\tIncorrect activations per example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_example, p*100), end= ' ')
            print()

            incorrect_activations_per_clause = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][Y_test==i,:].sum(axis=0)
            print("\tIncorrect activations per clause:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_clause, p*100), end= ' ')
            print()

            incorrect_activations_per_hit = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][np.logical_and(Y_test!=i, Y_test==Y_test_predicted),:].sum(axis=1)
            print("\tIncorrect activations per correctly classified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_hit, p*100), end= ' ')
            print()

            incorrect_activations_per_miss = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][np.logical_and(Y_test!=i, Y_test!= Y_test_predicted),:].sum(axis=1)
            print("\tIncorrect activations per misclassified example:", end=' ')
            for p in percentiles:
                print("%.0f" % np.percentile(incorrect_activations_per_miss, p*100), end= ' ')
            print()

            total_high_precision_activations_per_example = np.log((correct_high_precision_activations_per_example + 1) / (incorrect_high_precision_activations_per_example + 1))
            total_high_precision_activations_per_example_sorted = np.argsort(total_high_precision_activations_per_example)
            print("\tTotal high-precision activations per example:", end=' ')
            for p in percentiles:
                negative_examples_p = int(negative_examples*p)
                print("(%d vs %d)" % (
                    correct_high_precision_activations_per_example[total_high_precision_activations_per_example_sorted[negative_examples_p]],
                    incorrect_high_precision_activations_per_example[total_high_precision_activations_per_example_sorted[negative_examples_p]]
                    ), end= ' '
                )
            print()
              
            total_low_precision_activations_per_example = np.log((correct_low_precision_activations_per_example + 1) / (incorrect_low_precision_activations_per_example + 1))
            total_low_precision_activations_per_example_sorted = np.argsort(total_low_precision_activations_per_example)
            print("\tTotal low-precision activations per example:", end=' ')
            for p in percentiles:
                negative_examples_p = int(negative_examples*p)
                print("(%d vs %d)" % (
                    correct_low_precision_activations_per_example[total_low_precision_activations_per_example_sorted[negative_examples_p]],
                    incorrect_low_precision_activations_per_example[total_low_precision_activations_per_example_sorted[negative_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_example = np.log((correct_activations_per_example + 1) / (incorrect_activations_per_example + 1))
            total_activations_per_example_sorted = np.argsort(total_activations_per_example)
            print("\tTotal activations per example:", end=' ')
            for p in percentiles:
                negative_examples_p = int(negative_examples*p)
                print("(%d vs %d)" % (
                    correct_activations_per_example[total_activations_per_example_sorted[negative_examples_p]],
                    incorrect_activations_per_example[total_activations_per_example_sorted[negative_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_hit = np.log((correct_activations_per_hit + 1) / (incorrect_activations_per_hit + 1))
            total_activations_per_hit_sorted = np.argsort(total_activations_per_hit)
            print("\tTotal activations per correctly classified example:", end=' ')
            for p in percentiles:
                negative_examples_p = int(negative_hits*p)
                print("(%d vs %d)" % (
                    correct_activations_per_hit[total_activations_per_hit_sorted[negative_examples_p]],
                    incorrect_activations_per_hit[total_activations_per_hit_sorted[negative_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_miss = np.log((correct_activations_per_miss + 1) / (incorrect_activations_per_miss + 1))
            total_activations_per_miss_sorted = np.argsort(total_activations_per_miss)
            print("\tTotal activations per misclassified example:", end=' ')
            for p in percentiles:
                negative_examples_p = int(negative_misses*p)
                print("(%d vs %d)" % (
                    correct_activations_per_miss[total_activations_per_miss_sorted[negative_examples_p]],
                    incorrect_activations_per_miss[total_activations_per_miss_sorted[negative_examples_p]]
                    ), end= ' '
                )
            print()

            total_activations_per_clause = np.log((correct_activations_per_clause + 1) / (incorrect_activations_per_clause + 1))
            total_activations_per_clause_sorted = np.argsort(total_activations_per_clause)
            print("\tTotal activations per clause:", end=' ')
            for p in percentiles:
                print("(%d vs %d %.2f/%.2f)" % (
                        correct_activations_per_clause[total_activations_per_clause_sorted[int(args.num_clauses//2*p)]],
                        incorrect_activations_per_clause[total_activations_per_clause_sorted[int(args.num_clauses//2*p)]],
                        negative_precision[total_activations_per_clause_sorted[int(args.num_clauses//2*p)]],
                        negative_recall[total_activations_per_clause_sorted[int(args.num_clauses//2*p)]]
                    ), end= ' '
                )
            print()
            
        activations_per_example = X_test_transformed.sum(axis=1)
        activations_per_clause = X_test_transformed.sum(axis=0)
        total_activations = X_test_transformed.sum()

        print()

        print("Total activations:", total_activations)
        print("Activations per example:", np.percentile(activations_per_example, 1), np.percentile(activations_per_example, 5), np.percentile(activations_per_example, 10), np.percentile(activations_per_example, 25), activations_per_example.mean(), np.percentile(activations_per_example, 75), np.percentile(activations_per_example, 90), np.percentile(activations_per_example, 95), np.percentile(activations_per_example, 99))
        print("Activations per clause:", np.percentile(activations_per_clause, 5), np.percentile(activations_per_clause, 5), np.percentile(activations_per_clause, 10), np.percentile(activations_per_clause, 25), activations_per_clause.mean(), np.percentile(activations_per_clause, 75), np.percentile(activations_per_clause, 90), np.percentile(activations_per_clause, 95), np.percentile(activations_per_clause, 99))

        print()

        if args.device == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)
