import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

number_of_random_labels = 250
number_of_clauses = 100

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

Y_train_random = np.random.randint(number_of_random_labels, size=X_train.shape[0])

tm = TMCoalescedClassifier(number_of_clauses, int(np.sqrt(number_of_clauses))+1, 2.0, platform='CPU', weighted_clauses=True, type_ia_ii_feedback_ratio=1)

print("\nAccuracy over 60 epochs:\n")
for i in range(60):
        start_training = time()
        tm.fit(X_train, Y_train_random)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_train) == Y_train_random).mean()
        stop_testing = time()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
