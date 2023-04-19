import numpy as np
from time import time
from keras.datasets import mnist
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import lil_matrix
from skimage.util import view_as_windows
from sklearn.feature_extraction.text import CountVectorizer

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    data = np.zeros(m , dtype=np.uint32)
    data[num] = 1
    return data

ensembles = 5
epochs = 250

clauses = 8000 // 4
T = 10000 // 4
s = 5.0

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0)[:10000]
X_test = np.where(X_test > 75, 1, 0)[:10000]
Y_train = Y_train[0:10000]
Y_test = Y_test[0:10000]

X_train_tokenized = np.zeros((X_train.shape[0], 26, 26, 512), dtype=np.uint32)

for i in range(X_train.shape[0]):
        windows = view_as_windows(X_train[i,:,:], (3, 3))
        for q in range(windows.shape[0]):
                for r in range(windows.shape[1]):
                        patch = windows[q,r].reshape(-1).astype(np.uint32)
                        patch_id = patch.dot(1 << np.arange(patch.shape[-1] - 1, -1, -1))
                        X_train_tokenized[i, q, r, :] = bin_array(patch_id, 512)

print("Training data produced")

X_test_tokenized = np.zeros((X_test.shape[0], 26, 26, 512), dtype=np.uint32)

for i in range(X_test.shape[0]):
        windows = view_as_windows(X_test[i,:,:], (3, 3))
        for q in range(windows.shape[0]):
                for r in range(windows.shape[1]):
                        patch = windows[q,r].reshape(-1).astype(np.uint32)
                        patch_id = patch.dot(1 << np.arange(patch.shape[-1] - 1, -1, -1))
                        X_test_tokenized[i, q, r, :] = bin_array(patch_id, 512)

print("Testing data produced")

X_train = X_train_tokenized
X_test = X_test_tokenized

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, clauses, T, max_included_literals), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = TMClassifier(clauses, T, s, patch_dim=(1,1), platform='CPU', weighted_clauses=True)

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
