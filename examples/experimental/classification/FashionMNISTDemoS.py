import logging
import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
import numpy as np
from keras.datasets import fashion_mnist
import cv2

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

        activations_per_example = X_test_transformed.sum(axis=1)
        activations_per_clause = X_test_transformed.sum(axis=0)

        print()
        print("Activations per example:", np.percentile(activations_per_example, 1), np.percentile(activations_per_example, 5), np.percentile(activations_per_example, 10), np.percentile(activations_per_example, 25), activations_per_example.mean(), np.percentile(activations_per_example, 75), np.percentile(activations_per_example, 90), np.percentile(activations_per_example, 95), np.percentile(activations_per_example, 99))
        print("Activations per clause:", np.percentile(activations_per_clause, 5), np.percentile(activations_per_clause, 5), np.percentile(activations_per_clause, 10), np.percentile(activations_per_clause, 25), activations_per_clause.mean(), np.percentile(activations_per_clause, 75), np.percentile(activations_per_clause, 90), np.percentile(activations_per_clause, 95), np.percentile(activations_per_clause, 99))

        #clause_precision = []
        #for i in range(10):
        #    clause_precision.append(tm.clause_precision(i, 0, X_train, Y_train))
        #    X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2] *= clause_precision[-1]
        #    X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses] = 0
        
        #Y_test_predicted = np.argmax(X_test_transformed, axis=1)//args.num_clauses
        #print("Max Accuracy", (Y_test_predicted == Y_test).mean())

        p90 = int(args.num_clauses//2 * 0.9)
        p95 = int(args.num_clauses//2 * 0.95)
        p99 = int(args.num_clauses//2 * 0.99)
        for i in range(10):
            print("\nClass %d positive clauses\n" % (i))
            precision = tm.clause_precision(i, 0, X_test, Y_test)
            recall = tm.clause_recall(i, 0, X_test, Y_test)

            precision_sorted = np.argsort(precision)
            recall_sorted = np.argsort(recall)
            
            print("\tPrecision:", precision.mean(), "(", precision[precision_sorted[p90]], recall[precision_sorted[p90]], ")", "(", precision[precision_sorted[p95]], recall[precision_sorted[p95]], ")", "(", precision[precision_sorted[p99]], recall[precision_sorted[p99]], ")")
            print("\tRecall:", recall.mean(), "(", precision[recall_sorted[p90]], recall[recall_sorted[p90]], ")", "(", precision[recall_sorted[p95]], recall[recall_sorted[p95]], ")", "(", precision[recall_sorted[p99]], recall[recall_sorted[p99]], ")")
            
            high_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses:i*args.num_clauses+args.num_clauses//2][:,precision_sorted[p90:]][Y_test==i,:].sum(axis=1)
            print("\tHigh-precision activations per example:", np.percentile(high_precision_activations_per_example, 1), np.percentile(high_precision_activations_per_example, 5), np.percentile(high_precision_activations_per_example, 10), np.percentile(high_precision_activations_per_example, 25), high_precision_activations_per_example.mean(), np.percentile(high_precision_activations_per_example, 75), np.percentile(high_precision_activations_per_example, 90), np.percentile(high_precision_activations_per_example, 95), np.percentile(high_precision_activations_per_example, 99))

            print("\nClass %d Negative clauses\n" % (i))

            precision = tm.clause_precision(i, 1, X_test, Y_test)
            recall = tm.clause_recall(i, 1, X_test, Y_test)

            precision_sorted = np.argsort(precision)
            recall_sorted = np.argsort(recall)
            
            print("\tPrecision:", precision.mean(), "(", precision[precision_sorted[p90]], recall[precision_sorted[p90]], ")", "(", precision[precision_sorted[p95]], recall[precision_sorted[p95]], ")", "(", precision[precision_sorted[p99]], recall[precision_sorted[p99]], ")")
            print("\tRecall:", recall.mean(), "(", precision[recall_sorted[p90]], recall[recall_sorted[p90]], ")", "(", precision[recall_sorted[p95]], recall[recall_sorted[p95]], ")", "(", precision[recall_sorted[p99]], recall[recall_sorted[p99]], ")")
            
            high_precision_activations_per_example = X_test_transformed[:,i*args.num_clauses+args.num_clauses//2:(i+1)*args.num_clauses][:,precision_sorted[p90:]][Y_test!=i,:].sum(axis=1)
            print("\tHigh-precision activations per example:", np.percentile(high_precision_activations_per_example, 1), np.percentile(high_precision_activations_per_example, 5), np.percentile(high_precision_activations_per_example, 10), np.percentile(high_precision_activations_per_example, 25), high_precision_activations_per_example.mean(), np.percentile(high_precision_activations_per_example, 75), np.percentile(high_precision_activations_per_example, 90), np.percentile(high_precision_activations_per_example, 95), np.percentile(high_precision_activations_per_example, 99))

        print()

        if args.device == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)
