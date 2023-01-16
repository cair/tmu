from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
import numpy as np
from time import time

clauses = 4
number_of_features = 20
T = 15
s = 3.9
noise = 0.1

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

tm = TMCoalescedClassifier(clauses, T, s, clause_drop_p = 0.5, platform='CUDA', boost_true_positive_feedback=0)

for i in range(20):
	tm.fit(X_train, Y_train)

precision = []
recall = []
for i in range(2):
	precision.append((tm.clause_precision(i, 0, X_test, Y_test), tm.clause_precision(i, 1, X_test, Y_test)))
	recall.append((tm.clause_recall(i, 0, X_test, Y_test), tm.clause_recall(i, 1, X_test, Y_test)))

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("\nClauses:\n")

for j in range(clauses):
	print("Clause #%d " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.get_ta_action(j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))

	for i in range(2):
		print("\tC:%d W:%+2d P:%.2f R:%.2f" % (i, tm.get_weight(i, j), precision[i][int(tm.get_weight(i, j) < 0)][j], recall[i][int(tm.get_weight(i, j) < 0)][j]))

print("\nClause Co-Occurence Matrix:\n")
print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

print("\nLiteral Frequency:\n")
print(tm.literal_clause_frequency())
