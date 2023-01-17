import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

tm = TMCoalescedClassifier(20000, 50*100, 10.0, max_included_literals=5, weighted_clauses=True)

print("\nAccuracy over 60 epochs:\n")
for i in range(60):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        number_of_includes = 0
        for j in range(20000):
                number_of_includes += tm.number_of_include_actions(j)
        number_of_includes /= 20000

        print("#%d Accuracy: %.2f%% Includes: %.1f Training: %.2fs Testing: %.2fs" % (i+1, result, number_of_includes, stop_training-start_training, stop_testing-start_testing))
