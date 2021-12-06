import numpy as np
from time import time

from keras.datasets import mnist

from tmu.tsetlin_machine import TMCoalescedClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

tm = TMCoalescedClassifier(20000//4, 50*100//4, 5.0, platform='CUDA', patch_dim=(10, 10), weighted_clauses=True)

print("\nAccuracy over 30 epochs:\n")
for i in range(30):
	start = time()
	tm.fit(X_train, Y_train)
	stop = time()
	
	result_gpu = 100*(tm.predict_gpu(X_test) == Y_test).mean()
	print("#%d Accuracy GPU: %.2f%% (%.2fs)" % (i+1, result_gpu, stop-start))
	result_cpu = 100*(tm.predict_cpu(X_test) == Y_test).mean()
	print("#%d Accuracy CPU: %.2f%% (%.2fs)" % (i+1, result_gpu, stop-start))
