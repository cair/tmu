from tmu.tsetlin_machine import TMCoalescedClassifier
import numpy as np
from time import time

from keras.datasets import cifar100

import cv2

clauses = 80000
T = int(clauses*0.75)
s = 10.0 
patch_size = 3
resolution = 16
number_of_state_bits = 9
literal_drop_p = 0.25

epochs = 250
ensembles = 10

(X_train_org, Y_train), (X_test_org, Y_test) = cifar100.load_data()

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution), dtype=np.uint8) 
for z in range(resolution):
        X_train[:,:,:,:,z] = X_train_org[:,:,:,:] >= (z+1)*255/(resolution+1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution), dtype=np.uint8) 
for z in range(resolution):
        X_test[:,:,:,:,z] = X_test_org[:,:,:,:] >= (z+1)*255/(resolution+1)

X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3*resolution))
X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3*resolution))

f = open("cifar100_%.1f_%d_%d_%d_%.2f.txt" % (s, clauses, T,  patch_size, literal_drop_p), "w+")
for e in range(ensembles):
        tm = TMCoalescedClassifier(clauses, T, s, platform='CUDA', weighted_clauses=True, literal_drop_p=literal_drop_p)
        for i in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()
                print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
