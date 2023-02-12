import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

clauses = 20000//5
T = clauses//4

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

tm = TMCoalescedClassifier(clauses, T, 10.0, focused_negative_sampling=True, max_included_literals=10, max_positive_clauses=200, weighted_clauses=True)

print("\nAccuracy over 60 epochs:\n")
for e in range(60):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        number_of_positive_clauses = 0
        for i in range(tm.number_of_classes):
                number_of_positive_clauses += (tm.weight_banks[i].get_weights() > 0).sum()
        number_of_positive_clauses /= tm.number_of_classes

        number_of_includes = 0
        for j in range(clauses):
                number_of_includes += tm.number_of_include_actions(j)
        number_of_includes /= 2*clauses


        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs Positive clauses: %.1f Literals: %.1f" % (e+1, result, stop_training-start_training, stop_testing-start_testing, number_of_positive_clauses, number_of_includes))