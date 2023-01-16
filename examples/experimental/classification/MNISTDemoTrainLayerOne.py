import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

import copy

clauses = 64
T = int(clauses*0.75)
s = 5.0
patch_size = 3
resolution = 8
number_of_state_bits = 8

(X_train_org, Y_train), (X_test_org, Y_test) = mnist.load_data()

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], resolution), dtype=np.uint8)
for z in range(resolution):
        X_train[:,:,:,z] = X_train_org[:,:,:] >= (z+1)*255/(resolution+1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], resolution), dtype=np.uint8)
for z in range(resolution):
        X_test[:,:,:,z] = X_test_org[:,:,:] >= (z+1)*255/(resolution+1)

X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], resolution))
X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], resolution))

tm = TMCoalescedClassifier(clauses, T, s, platform='CUDA', patch_dim=(3, 3), weighted_clauses=True)

print("\nAccuracy over 10 epochs:\n")
for i in range(1):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()
	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

print("\nTransforming datasets")
start_transformation = time()
X_train_transformed = tm.transform_patchwise(X_train)
print(X_train_transformed.shape)
X_test_transformed = tm.transform_patchwise(X_test)
stop_transformation = time()
print("Transformation time: %.fs" % (stop_transformation - start_transformation))

print("Saving transformed datasets")
np.savez_compressed("X_train_transformed.npz", X_train_transformed)
np.savez_compressed("X_test_transformed.npz", X_test_transformed)
