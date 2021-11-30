import numpy as np
from time import time

from keras.datasets import cifar10

import cv2

from tmu.tsetlin_machine import TMOneVsOneClassifier

factor = 20
clauses = int(4000*factor)
T = int(75*10*factor)
s = 20.0
patch_size = 8

epochs = 250
ensembles = 10

labels = [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
                X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
                X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

f = open("cifar10_%.1f_%d_%d_%d.txt" % (s, clauses, T,  patch_size), "w+")

for e in range(ensembles):
        tm = TMOneVsOneClassifier(clauses, T, s, (patch_size, patch_size))

        for i in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train, incremental=True)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()
                print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()