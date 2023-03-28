import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.vanilla_classifier import TMClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

Y_train = Y_train
Y_test = Y_test

tm = TMClassifier(2000, 5000, 10.0, platform='CPU_sparse', weighted_clauses=True, literal_sampling = 0.1, max_included_literals=32, feedback_rate_excluded_literals=1, absorbing=50, literal_insertion_state=75)

print("\nAccuracy over 60 epochs:\n")
for e in range(250):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100*(tm.predict(X_train) == Y_train).mean()

        absorbed = 0.0
        unallocated = 0
        for i in range(10):
                for j in range(200):
                        absorbed += 1.0 - (tm.number_of_include_actions(i, j) + tm.number_of_exclude_actions(i, j)) / (X_train.shape[1]*2)
                        unallocated += tm.number_of_unallocated_literals(i, j)

        absorbed = 100 * absorbed / (10*200)

        print("#%d Test Accuracy: %.2f%% Train Accuracy: %.2f%% Absorbed: %.2f%% Unallocated: %d Training: %.2fs Testing: %.2fs" % (e+1, result_test, result_train, absorbed, unallocated, stop_training-start_training, stop_testing-start_testing))