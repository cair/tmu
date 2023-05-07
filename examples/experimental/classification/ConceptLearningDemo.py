from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time

number_of_features = 10
noise = 0.0

s = 1.1
clauses = 1
T = 3

examples = 5000

X_train = np.random.randint(0, 2, size=(examples, number_of_features), dtype=np.uint32)
Y_train = np.logical_and(X_train[:,2], np.logical_and(X_train[:,0], X_train[:,1])).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(examples) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(examples, number_of_features), dtype=np.uint32)
Y_test = np.logical_and(X_test[:,2], np.logical_and(X_test[:,0], X_test[:,1])).astype(dtype=np.uint32)

tm = TMClassifier(clauses*2, T, s, type_i_ii_ratio=1.0, platform='CPU', boost_true_positive_feedback=0)

for i in range(20):
	tm.fit(X_train, Y_train)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

precision = tm.clause_precision(1, 0, X_test, Y_test)
recall = tm.clause_recall(1, 0, X_test, Y_test)

for j in range(clauses):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]))
	l = []
	for k in range(number_of_features*2):
		if tm.get_ta_action(j, k, the_class = 1, polarity = 0):
			if k < number_of_features:
				print("\tINCLUDE: x%d (TA State %d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
			else:
				print("\tINCLUDE ¬x%d (TA State %d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
		else:
			if k < number_of_features:
				print("\tEXCLUDE: x%d (TA State %d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
			else:
				print("\tEXCLUDE ¬x%d (TA State %d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))