import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.vanilla_classifier import TMClassifier

clauses = 2000
T = 5000
s = 10.0
number_of_state_bits_ta = 8

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

tm = TMClassifier(clauses, T, s, type_iii_feedback=True, number_of_state_bits_ta=number_of_state_bits_ta, number_of_state_bits_ind=8, platform='CPU', weighted_clauses=True)

f = open("mnist_type_iii_feedback_%.1f_%d_%d_%d.txt" % (s, clauses, T, number_of_state_bits_ta), "w+")
print("\nAccuracy over 250 epochs:\n")
for i in range(250):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        start_testing = time()
        result_train = 100*(tm.predict(X_train) == Y_train).mean()
        stop_testing = time()

        print("#%d Training Accuracy: %.2f%% Testing Accuracy: %.2f%% Literals: %d Training: %.2fs Testing: %.2fs" % (i+1, result_train, result_test, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing))

        print("%d %.2f %.2f %d %.2f %.2f" % (i, result_train, result_test, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing), file=f)
        f.flush()
f.close()
