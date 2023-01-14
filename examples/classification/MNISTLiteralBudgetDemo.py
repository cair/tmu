import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.vanilla_classifier import TMClassifier

ensembles = 5
epochs = 250

clauses = 8000
T = 10000
s = 5.0
max_included_literals = 1

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, clauses, T, max_included_literals), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = TMClassifier(clauses, T, s, platform='CUDA', max_included_literals=max_included_literals, weighted_clauses=True)

        for epoch in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                number_of_includes = 0
                for i in range(10):
                        for j in range(clauses):
                                number_of_includes += tm.number_of_include_actions(i, j)
                number_of_includes /= 10*clauses

                print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
