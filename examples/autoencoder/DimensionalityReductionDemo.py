import numpy as np
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix

from tmu.models.autoencoder.autoencoder import TMAutoEncoder

number_of_features = 12

noise = 0.1

clause_weight_threshold = 0

number_of_examples = 2500
accumulation = 1

clauses = 10
T = 8*10
s = 1.5

print("Number of clauses:", clauses)

output_active = np.array([0, 1], dtype=np.uint32)

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
X_train[:,0] = np.where(np.random.rand(5000) <= noise, 1- X_train[:,1],  X_train[:,1])

tm = TMAutoEncoder(clauses, T, s, output_active, max_included_literals=3, accumulation=accumulation, feature_negation=True, platform='CPU', output_balancing=True)

print("\nAccuracy Over 40 Epochs:")
for e in range(40):
	start_training = time()
	tm.fit(X_train, number_of_examples=number_of_examples)
	stop_training = time()

	print("\nEpoch #%d\n" % (e+1))

	print("Calculating precision\n")
	precision = []
	for i in range(output_active.shape[0]):
		precision.append(tm.clause_precision(i, True, X_train, number_of_examples=500))

	print("Calculating recall\n")
	recall = []
	for i in range(output_active.shape[0]):
		recall.append(tm.clause_recall(i, True, X_train, number_of_examples=500))

	print("Clauses\n")

	for j in range(clauses):
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

	profile = np.empty((output_active.shape[0], clauses))
	for i in range(output_active.shape[0]):
		weights = tm.get_weights(i)
		profile[i,:] = np.where(weights >= clause_weight_threshold, weights, 0)

	similarity = cosine_similarity(profile)

	print("\nWord Similarity\n")

	for i in range(output_active.shape[0]):
		print("x%d" % (output_active[i]), end=': ')
		sorted_index = np.argsort(-1*similarity[i,:])
		for j in range(1, output_active.shape[0]):
			print("x%d(%.2f) " % (output_active[sorted_index[j]], similarity[i,sorted_index[j]]), end=' ')
		print()

	print("\nTraining Time: %.2f" % (stop_training - start_training))
