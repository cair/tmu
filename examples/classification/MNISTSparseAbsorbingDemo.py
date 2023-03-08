import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.vanilla_classifier import TMClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)#[0:1000]
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)#[0:1000]

Y_train = Y_train#[0:1000]
Y_test = Y_test#[0:1000]

tm = TMClassifier(2000, 5000, 10.0, max_included_literals=32, platform='CPU_sparse', weighted_clauses=True, absorbing=75)

print("\nAccuracy over 60 epochs:\n")
for e in range(60):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        absorbed = 0.0
        for i in range(10):
                for j in range(2000):
                        absorbed += 1.0 - (tm.number_of_include_actions(i, j) + tm.number_of_exclude_actions(i, j)) / (X_train.shape[1]*2)
        absorbed = 100 * absorbed / (10*2000)

        print("#%d Accuracy: %.2f%% Absorbed: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, absorbed, stop_training-start_training, stop_testing-start_testing))