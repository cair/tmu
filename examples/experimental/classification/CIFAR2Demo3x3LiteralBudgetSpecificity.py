from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar10

animals = np.array([2, 3, 4, 5, 6, 7])

max_included_literals = 32
clauses = 8000
T = int(clauses * 0.75)
s = 10.0
patch_size = 3
resolution = 8
number_of_state_bits_ta = 8
literal_drop_p = 0.0

epochs = 250
ensembles = 5

classes = 10

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
                   dtype=np.uint8)
for z in range(resolution):
    X_train[:, :, :, :, z] = X_train_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution),
                  dtype=np.uint8)
for z in range(resolution):
    X_test[:, :, :, :, z] = X_test_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3 * resolution))
X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3 * resolution))

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

f = open("cifar2_%.1f_%d_%d_%d_%.2f_%d_%d.txt" % (
s, clauses, T, patch_size, literal_drop_p, resolution, max_included_literals), "w+")
for ensemble in range(ensembles):
    tm = TMClassifier(clauses, T, s, platform='GPU', patch_dim=(patch_size, patch_size),
                      number_of_state_bits_ta=number_of_state_bits_ta, weighted_clauses=True,
                      literal_drop_p=literal_drop_p, max_included_literals=max_included_literals)
    for epoch in range(epochs):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(X_train) == Y_train).mean()

        number_of_includes = 0
        for i in range(2):
            for j in range(clauses):
                number_of_includes += tm.number_of_include_actions(i, j)
        number_of_includes /= 2 * clauses

        print("\nClass 0 Positive Clauses:\n")

        precision = tm.clause_precision(0, 0, X_test, Y_test)
        recall = tm.clause_recall(0, 0, X_test, Y_test)

        print(precision.mean(), recall.mean())

        print("\nClass 0 Negative Clauses:\n")

        precision = tm.clause_precision(0, 1, X_test, Y_test)
        recall = tm.clause_recall(0, 1, X_test, Y_test)

        print(precision.mean(), recall.mean())

        print("\nClass 1 Positive Clauses:\n")

        precision = tm.clause_precision(1, 0, X_test, Y_test)
        recall = tm.clause_recall(1, 0, X_test, Y_test)

        print(precision.mean(), recall.mean())

        print("\nClass 1 Negative Clauses:\n")

        precision = tm.clause_precision(1, 1, X_test, Y_test)
        recall = tm.clause_recall(1, 1, X_test, Y_test)

        print(precision.mean(), recall.mean())

        print("%d %d %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, number_of_includes, result_train, result_test, stop_training - start_training, stop_testing - start_testing))
        print("%d %d %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, number_of_includes, result_train, result_test, stop_training - start_training, stop_testing - start_testing),
              file=f)
        f.flush()
f.close()
