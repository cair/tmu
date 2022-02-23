import numpy as np
from time import time

from keras.datasets import mnist

from tmu.tsetlin_machine import TMCoalescedClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = TMCoalescedClassifier(20000, 50*100, 10.0, weighted_clauses=True, type_iii_feedback=True, focused_negative_sampling=True)

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        start_testing = time()
        result_train = 100*(tm.predict(X_train) == Y_train).mean()
        stop_testing = time()

        print("#%d Training Accuracy: %.2f%% Testing Accuracy: %.2f%% Literals: %d Training: %.2fs Testing: %.2fs" % (i+1, result_train, result_test, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing))

        print("%d %.2f %.2f %d %.2f %.2f" % (i, result_train, result_test, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing), file=f)
        f.flush()
f.close()