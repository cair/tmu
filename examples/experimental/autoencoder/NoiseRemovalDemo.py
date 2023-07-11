import numpy as np
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix

from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

number_of_code_chunks = 4

number_of_classes = 2**number_of_code_chunks

noise = 0.2

number_of_code_bits = 32

number_of_features = number_of_code_chunks * number_of_code_bits

number_of_examples = 2500

number_of_clauses = 512
T = int(number_of_clauses*0.75*100)
s = 10.0#10.0
max_included_literals = number_of_features*2
accumulation = 1
clause_weight_threshold = 0

print("Number of classes", number_of_classes)
print("Number of clauses:", number_of_clauses)
print("T:", T)
print("Number of features:", number_of_features)

output_active = np.arange(number_of_features, dtype=np.uint32)

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
X_test_noisy = np.where(np.random.rand(number_of_examples, number_of_features) <= noise, 1-X_test, X_test) # Adds noise

tm = TMAutoEncoder(number_of_clauses, T, s, output_active, max_included_literals=max_included_literals, feature_negation=False, accumulation=accumulation, platform='CPU', output_balancing=True)

print("\nAccuracy Over 40 Epochs:")
for e in range(100):
	start_training = time()
	tm.fit(X_train, number_of_examples=number_of_examples)
	stop_training = time()

	print("\nEpoch #%d\n" % (e+1))

	print(zero, zero)
	print(one, one)

	print("Calculating precision\n")
	precision = []
	for i in range(output_active.shape[0]):
		precision.append(tm.clause_precision(i, True, X_test, number_of_examples=500))

	print("Calculating recall\n")
	recall = []
	for i in range(output_active.shape[0]):
		recall.append(tm.clause_recall(i, True, X_test, number_of_examples=500))

	print("Clauses\n")

	for j in range(number_of_clauses):
		print("Clause #%d " % (j), end=' ')
		for i in range(output_active.shape[0]):
			print("x%d:W%d:P%.2f:R%.2f " % (i, tm.get_weight(i, j), precision[i][j], recall[i][j]), end=' ')

		l = []
		for k in range(tm.clause_bank.number_of_literals):
			if tm.get_ta_action(j, k) == 1:
				if k < tm.clause_bank.number_of_features:
					l.append("x%d(%d)" % (k, tm.clause_bank.get_ta_state(j, k)))
				else:
					l.append("¬x%d(%d)" % (k-tm.clause_bank.number_of_features, tm.clause_bank.get_ta_state(j, k)))
		print(" ∧ ".join(l))

	profile = np.empty((output_active.shape[0], number_of_clauses))
	for i in range(output_active.shape[0]):
		weights = tm.get_weights(i)
		profile[i,:] = np.where(weights >= clause_weight_threshold, weights, 0)

	similarity = cosine_similarity(profile)

	print("\nSimilarity\n")

	for i in range(output_active.shape[0]):
		print("x%d" % (output_active[i]), end=': ')
		sorted_index = np.argsort(-1*similarity[i,:])
		for j in range(1, output_active.shape[0]):
			print("x%d(%.2f) " % (output_active[sorted_index[j]], similarity[i,sorted_index[j]]), end=' ')
		print()

	print("Number of features", tm.clause_bank.number_of_features)
	print("Number of classes", tm.number_of_classes)

	X_predicted = tm.predict(X_test)

	print("Accuracy", 100*(tm.predict(X_test_noisy) == X_test).mean())

	print("#1 Input", X_test_noisy[0,:])
	print("#1 Predicted", X_predicted[0,:])
	print("#1 Correct", X_test[0,:])

	print("#2 Input", X_test_noisy[1,:])
	print("#2 Predicted", X_predicted[1,:])
	print("#2 Correct", X_test[1,:])

	print("\nTraining Time: %.2f" % (stop_training - start_training))
