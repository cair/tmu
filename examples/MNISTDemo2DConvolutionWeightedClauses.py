import numpy as np
from time import time

from keras.datasets import mnist

from tmu.tsetlin_machine import TMOneVsOneClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

tm = TMOneVsOneClassifier(2500, int(250*0.75), 5.0, platform='CPU', patch_dim=(10, 10), weighted_clauses=True)

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
	start = time()
	tm.fit(X_train, Y_train)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))