from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
import numpy as np
from time import time
from sklearn.metrics import f1_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar10

max_included_literals=32
clauses = 80000
T = int(clauses*0.75)
s = 10.0
patch_size = 3
resolution = 8
number_of_state_bits_ta = 8
literal_drop_p = 0.0

epochs = 250
ensembles = 5

classes = 10

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

X_train = X_train[Y_train<classes]
Y_train = Y_train[Y_train<classes]

X_test = X_test[Y_test<classes]
Y_test = Y_test[Y_test<classes]

f = open("cifar10_coalesced_%.1f_%d_%d_%d_%.2f_%d_%d.txt" % (s, clauses, T,  patch_size, literal_drop_p, resolution, max_included_literals), "w+")
for ensemble in range(ensembles):
	tm = TMCoalescedClassifier(clauses, T, s, platform='CUDA', patch_dim=(patch_size, patch_size), number_of_state_bits_ta=number_of_state_bits_ta, focused_negative_sampling=True, weighted_clauses=True, literal_drop_p=literal_drop_p, max_included_literals=max_included_literals)
	for epoch in range(epochs):
		start_training = time()
		tm.fit(X_train, Y_train)
		stop_training = time()

		start_testing = time()
		result_test = 100*(tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		number_of_includes = 0
		for j in range(clauses):
			number_of_includes += tm.number_of_include_actions(j)
		number_of_includes /= clauses

		print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, stop_training-start_training, stop_testing-start_testing))
		print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
		f.flush()
f.close()
