import numpy as np
from time import time

from keras.datasets import mnist

from tmu.tsetlin_machine import TMClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = TMClassifier(2000, 50*100, 10.0, platform='CPU', max_included_literals=5, weighted_clauses=True)

print("\nAccuracy over 250 epochs:\n")
for e in range(250):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        number_of_includes = 0
        for i in range(10):
                for j in range(2000):
                        number_of_includes += tm.number_of_include_actions(i, j)
        number_of_includes /= 10*2000

        print("#%d Accuracy: %.2f%% Includes: %.1f Training: %.2fs Testing: %.2fs" % (e+1, result_test, number_of_includes, stop_training-start_training, stop_testing-start_testing))
