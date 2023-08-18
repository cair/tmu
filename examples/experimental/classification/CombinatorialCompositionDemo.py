import numpy as np
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import BernoulliNB

from scipy.sparse import csr_matrix

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.models.classification.vanilla_classifier import TMClassifier

number_of_code_chunks = 2

number_of_classes = 2**number_of_code_chunks

noise = 0.2

number_of_code_bits = 4

number_of_features = number_of_code_chunks * number_of_code_bits

clause_weight_threshold = 0

number_of_examples = 2500
accumulation = 1

factor = 20
number_of_clauses = 10*factor
T = 8*10
s = 20.0

print("Number of classes", number_of_classes)
print("Number of clauses:", number_of_clauses)
print("Number of features:", number_of_features)

output_active = np.arange(number_of_features, dtype=np.uint32)

#zero = np.random.randint(2, size=number_of_code_bits, dtype=np.uint32)

zero = np.zeros(number_of_code_bits, dtype=np.uint32)
one = np.logical_not(zero).astype(np.uint32)

X_train = np.random.randint(0, 2, size=(number_of_examples, number_of_features), dtype=np.uint32)
Y_train = np.empty(number_of_examples, dtype=np.uint32)
for i in range(number_of_examples):
	class_bits = np.random.randint(2, size=number_of_code_chunks, dtype=np.uint32)
	Y_train[i] = class_bits.dot(1 << np.arange(class_bits.shape[-1] - 1, -1, -1))
	for j in range(number_of_code_chunks):
		if class_bits[j] == 0:
			X_train[i,number_of_code_bits*j:number_of_code_bits*(j+1)] = zero
		else:
			X_train[i,number_of_code_bits*j:number_of_code_bits*(j+1)] = one
X_train = np.where(np.random.rand(number_of_examples, number_of_features) <= noise, 1-X_train, X_train) # Adds noise

X_test = np.random.randint(0, 2, size=(number_of_examples, number_of_features), dtype=np.uint32)
Y_test = np.empty(number_of_examples, dtype=np.uint32)
for i in range(number_of_examples):
	class_bits = np.random.randint(2, size=number_of_code_chunks, dtype=np.uint32)
	Y_test[i] = class_bits.dot(1 << np.arange(class_bits.shape[-1] - 1, -1, -1))
	for j in range(number_of_code_chunks):
		if class_bits[j] == 0:
			X_test[i,number_of_code_bits*j:number_of_code_bits*(j+1)] = zero
		else:
			X_test[i,number_of_code_bits*j:number_of_code_bits*(j+1)] = one
X_test = np.where(np.random.rand(number_of_examples, number_of_features) <= noise, 1-X_test, X_test) # Adds noise

tm = TMClassifier(number_of_clauses, T, s, weighted_clauses=False, platform='CPU')

for i in range(200):
	tm.fit(X_train, Y_train)

np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

print("Zero", zero)
print("One", one)
 
for i in range(number_of_classes):
	print("\nClass %d Positive Clauses:\n" % (i))

	precision = tm.clause_precision(i, 0, X_test, Y_test)
	recall = tm.clause_recall(i, 0, X_test, Y_test)

	for j in range(number_of_clauses//2):
		print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(i, 0, j), precision[j], recall[j]), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k, the_class = i, polarity = 0):
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

	print("\nClass %d Negative Clauses:\n" % (i))

	precision = tm.clause_precision(i, 1, X_test, Y_test)
	recall = tm.clause_recall(i, 1, X_test, Y_test)

	for j in range(number_of_clauses//2):
		print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(i, 1, j), precision[j], recall[j]), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k, the_class = i, polarity = 1):
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

nb = BernoulliNB()
nb.fit(X_train, Y_train)
print("NB Accuracy:", 100*(nb.predict(X_test) == Y_test).mean())
