import numpy as np
from time import time
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import lil_matrix
from skimage.util import view_as_windows
from sklearn.feature_extraction.text import CountVectorizer
import cv2
from skimage.transform import pyramid_gaussian, pyramid_laplacian, downscale_local_mean
import matplotlib.pyplot as plt

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

scaling = 1.0
unique_patches = 2**9
hypervector_size = int(512*scaling)
bits = np.maximum(5, int(5*scaling))
print(bits)

resolution = 8

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = np.zeros((unique_patches, hypervector_size), dtype=np.uint32)
for i in range(unique_patches):
        selection = np.random.choice(indexes, size=(bits))
        encoding[i][selection] = 1

animals = np.array([2, 3, 4, 5, 6, 7])

ensembles = 5
epochs = 250

max_included_literals = 32
clauses = 2000
T = 5000
s = 1.5
step = 1
visual_tokens = True

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org[0:5000]
X_test_org = X_test_org[0:5000]
Y_train = Y_train.reshape(Y_train.shape[0])[0:5000]
Y_test = Y_test.reshape(Y_test.shape[0])[0:5000]

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
                   dtype=np.uint8)
for z in range(resolution):
    X_train[:, :, :, :, z] = X_train_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution),
                  dtype=np.uint8)
for z in range(resolution):
    X_test[:, :, :, :, z] = X_test_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

if visual_tokens:
        X_train_tokenized = np.zeros((X_train.shape[0], 30//step, 30//step, hypervector_size), dtype=np.uint32)

        for i in range(X_train.shape[0]):
                for c in range(3):
                        for z in range(resolution):
                                windows = view_as_windows(X_train[i,:,:,c,z], (3, 3), step=step)
                                for u in range(windows.shape[0]):
                                        for v in range(windows.shape[1]):
                                                patch = windows[u,v].reshape(-1).astype(np.uint32)
                                                patch_id = patch.dot(1 << np.arange(patch.shape[-1] - 1, -1, -1))
                                                X_train_tokenized[i, u, v, :] |= np.roll(encoding[patch_id], c + z*10)
        print("Training data produced")

        X_test_tokenized = np.zeros((X_test.shape[0], 30//step, 30//step, hypervector_size), dtype=np.uint32)

        for i in range(X_test.shape[0]):
                for c in range(3):
                        for z in range(resolution):
                                windows = view_as_windows(X_test[i,:,:,c,z], (3, 3), step=step)
                                for u in range(windows.shape[0]):
                                        for v in range(windows.shape[1]):
                                                patch = windows[u,v].reshape(-1).astype(np.uint32)
                                                patch_id = patch.dot(1 << np.arange(patch.shape[-1] - 1, -1, -1))
                                                X_test_tokenized[i, u, v, :] |= np.roll(encoding[patch_id], c + z*10)

        print("Testing data produced")

        X_train = X_train_tokenized
        X_test = X_test_tokenized
else:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], -1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], -1))

print(X_train.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d_%d_%d.txt" % (s, clauses, T, step, visual_tokens, scaling), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        if visual_tokens:
                tm = TMClassifier(clauses, T, s, max_included_literals=max_included_literals, patch_dim=(1,1), platform='CPU', weighted_clauses=True)
        else:
                tm = TMClassifier(clauses, T, s, max_included_literals=max_included_literals, patch_dim=(3,3), platform='CPU', weighted_clauses=True)

        for epoch in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()

                number_of_includes = 0
                for i in range(2):
                        for j in range(clauses):
                                number_of_includes += tm.number_of_include_actions(i, j)
                number_of_includes /= 2*clauses

                print("%d %d %.2f %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, result_train, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
