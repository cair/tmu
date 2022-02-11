import numpy as np
from time import time

from keras.datasets import mnist

from tmu.tsetlin_machine import TMClassifier

number_of_clauses = 2000

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = TMClassifier(number_of_clauses, 50*100, 10.0, 200.0, number_of_state_bits_ta=8, number_of_state_bits_ind=8, platform='CPU', weighted_clauses=True)

number_of_features = 28*28

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
        start_training = time()
        tm.fit(X_train, Y_train, type_iii=True)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        print("#%d Accuracy: %.2f%% Literals: %d Training: %.2fs Testing: %.2fs" % (i+1, result, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing))