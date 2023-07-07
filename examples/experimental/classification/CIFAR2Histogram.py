import numpy as np
from time import time
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import lil_matrix
from skimage.util import view_as_windows
from sklearn.feature_extraction.text import CountVectorizer
from skimage.transform import pyramid_gaussian, pyramid_laplacian, downscale_local_mean

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

animals = np.array([2, 3, 4, 5, 6, 7])

ensembles = 5
epochs = 250

max_included_literals = 32
clauses = 8000
T = int(clauses * 0.75)
s = 10.0
patch_size = 8
resolution = 8
number_of_state_bits_ta = 8
literal_drop_p = 0.0
step = 2

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org.astype(np.uint32)#[:10000]
X_test_org = X_test_org.astype(np.uint32)#[:10000]
Y_train = Y_train.reshape(Y_train.shape[0])#[:10000]
Y_test = Y_test.reshape(Y_test.shape[0])#[:10000]

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

X_train = np.zeros((X_train_org.shape[0], (X_train_org.shape[1] - patch_size)//step + 1, (X_train_org.shape[2] - patch_size)//step + 1, resolution**3), dtype=np.uint32)
for i in range(X_train.shape[0]):
        windows_r = view_as_windows(X_train_org[i,:,:,0], (patch_size, patch_size), step=step)
        windows_g = view_as_windows(X_train_org[i,:,:,1], (patch_size, patch_size), step=step)
        windows_b = view_as_windows(X_train_org[i,:,:,2], (patch_size, patch_size), step=step)

        for u in range(windows_r.shape[0]):
                for v in range(windows_r.shape[1]):
                        patch_r = windows_r[u,v].astype(np.uint32)
                        patch_g = windows_g[u,v].astype(np.uint32)
                        patch_b = windows_b[u,v].astype(np.uint32)
                        for x in range(patch_size):
                                for y in range(patch_size):
                                        color_id = (patch_r[x, y]//(256//resolution)) * (resolution**2) + (patch_g[x, y]//(256//resolution)) * resolution + (patch_b[x, y]//(256//resolution))
                                        X_train[i, u, v, color_id] = 1

print("Training data produced")

X_test = np.zeros((X_test_org.shape[0], (X_train_org.shape[1] - patch_size)//step + 1, (X_train_org.shape[2] - patch_size)//step + 1, resolution**3), dtype=np.uint32)
for i in range(X_test.shape[0]):
        windows_r = view_as_windows(X_test_org[i,:,:,0], (patch_size, patch_size), step=step)
        windows_g = view_as_windows(X_test_org[i,:,:,1], (patch_size, patch_size), step=step)
        windows_b = view_as_windows(X_test_org[i,:,:,2], (patch_size, patch_size), step=step)

        for u in range(windows_r.shape[0]):
                for v in range(windows_r.shape[1]):
                        patch_r = windows_r[u,v].astype(np.uint32)
                        patch_g = windows_g[u,v].astype(np.uint32)
                        patch_b = windows_b[u,v].astype(np.uint32)
                        for x in range(patch_size):
                                for y in range(patch_size):
                                        color_id = (patch_r[x, y]//(256//resolution)) * (resolution**2) + (patch_g[x, y]//(256//resolution)) * resolution + (patch_b[x, y]//(256//resolution))
                                        X_test[i, u, v, color_id] = 1
print("Testing data produced")

print(X_train.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d.txt" % (s, clauses, T, step), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = TMClassifier(clauses, T, s, max_included_literals=max_included_literals, patch_dim=(1,1), platform='GPU', weighted_clauses=True)
      
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
