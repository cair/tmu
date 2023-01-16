from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time

from keras.datasets import cifar10

import cv2

clauses = 128
T = int(clauses*0.75)
s = 10.0
patch_size = 3
resolution = 8
number_of_state_bits = 8
clause_drop_p = 0.0
literal_drop_p = 0.0

epochs = 250
ensembles = 10

labels = [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

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

f = open("cifar10_%.1f_%d_%d_%d_%d_%.2f_%.2f.txt" % (s, clauses, T,  patch_size, resolution, clause_drop_p, literal_drop_p), "w+")

for e in range(ensembles):
        tm = TMClassifier(clauses, T, s, weighted_clauses=True, platform='CPU', clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p, patch_dim=(patch_size, patch_size), number_of_state_bits_ta=number_of_state_bits)

        for i in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train, type_iii_feedback=True)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()
                print("%d %d %.2f %.2f %d %.2f %.2f" % (e, i, result_train, result_test, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %d %.2f %.2f" % (e, i, result_train, result_test, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
