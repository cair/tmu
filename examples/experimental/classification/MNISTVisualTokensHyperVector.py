import numpy as np
from time import time
from keras.datasets import mnist
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import lil_matrix
from skimage.util import view_as_windows
from sklearn.feature_extraction.text import CountVectorizer

ensembles = 5
epochs = 250

max_included_literals = 32
clauses = 8000 // 4
T = 10000 // 4
s = 1.5

# Produces hypervector codes

scaling = 0.5
unique_patches = 2**9 # Total number of unique patches
hypervector_size = int(512*scaling)
bits = np.maximum(5, int(5*scaling))

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = np.zeros((unique_patches, hypervector_size), dtype=np.uint32)
for i in range(unique_patches):
        selection = np.random.choice(indexes, size=(bits))
        encoding[i][selection] = 1

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0)[:10000]
X_test = np.where(X_test > 75, 1, 0)[:10000]
Y_train = Y_train[0:10000]
Y_test = Y_test[0:10000]

# Tokenizes images into hypervectors

X_train_tokenized = np.zeros((X_train.shape[0], 26, 26, hypervector_size), dtype=np.uint32)
for i in range(X_train.shape[0]):
        windows = view_as_windows(X_train[i,:,:], (3, 3))
        for q in range(windows.shape[0]):
                for r in range(windows.shape[1]):
                        patch = windows[q,r].reshape(-1).astype(np.uint32)
                        patch_id = patch.dot(1 << np.arange(patch.shape[-1] - 1, -1, -1))
                        X_train_tokenized[i, q, r, :] = encoding[patch_id]

print("Training data produced")

X_test_tokenized = np.zeros((X_test.shape[0], 26, 26, hypervector_size), dtype=np.uint32)
for i in range(X_test.shape[0]):
        windows = view_as_windows(X_test[i,:,:], (3, 3))
        for q in range(windows.shape[0]):
                for r in range(windows.shape[1]):
                        patch = windows[q,r].reshape(-1).astype(np.uint32)
                        patch_id = patch.dot(1 << np.arange(patch.shape[-1] - 1, -1, -1))
                        X_test_tokenized[i, q, r, :] = encoding[patch_id]

print("Testing data produced")

# Starts training on the visual tokens encoded as hypervectors

X_train = X_train_tokenized
X_test = X_test_tokenized

f = open("mnist_%.1f_%d_%d_%d_%d.txt" % (s, clauses, T, bits, hypervector_size), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = TMClassifier(clauses, T, s, max_included_literals = max_included_literals, patch_dim=(1,1), platform='CPU', weighted_clauses=True)

        for epoch in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()

                number_of_includes = 0
                for i in range(10):
                        for j in range(clauses):
                                number_of_includes += tm.number_of_include_actions(i, j)
                number_of_includes /= 10*clauses

                print("%d %d %.2f %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, result_train, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
