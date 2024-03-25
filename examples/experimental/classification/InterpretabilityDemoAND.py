from tmu.experimental.models.multichannel_classifier import TMMultiChannelClassifier
import numpy as np

number_of_and_components = 2
clauses = 4
T = 15*2 # Test
global_T = ((number_of_and_components*T,T), (T,number_of_and_components*T))
s = (3.1, 2.1)
number_of_features = 4
noise = 0.1

examples = 20000

epochs = 25

X_train = np.random.randint(0, 2, size=(number_of_and_components, examples, number_of_features), dtype=np.uint32)
Y_train = np.ones(examples)
for i in range(number_of_and_components):
	Yi_train = np.logical_xor(X_train[i,:,0], X_train[i,:,1]).astype(dtype=np.uint32)
	Y_train = np.logical_and(Y_train, Yi_train)
X_train_0 = X_train[:,Y_train == 0]
Y_train_0 = Y_train[Y_train == 0]
X_train_1 = X_train[:,Y_train == 1]
Y_train_1 = Y_train[Y_train == 1]
X_train = np.concatenate([X_train_0[:,:Y_train_1.shape[0]], X_train_1], axis=1)
Y_train = np.concatenate([Y_train_0[:Y_train_1.shape[0]], Y_train_1], axis=0)

indexes = np.arange(Y_train_1.shape[0]*2)
np.random.shuffle(indexes)
print(indexes)
X_train = X_train[:,indexes]
Y_train = Y_train[indexes]

print(X_train.shape)
print(Y_train.shape)

Y_train = np.where(np.random.rand(Y_train.shape[0]) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(number_of_and_components, examples, number_of_features), dtype=np.uint32)
Y_test = np.ones(examples)
for i in range(number_of_and_components):
	Yi_test = np.logical_xor(X_test[i,:,0], X_test[i,:,1]).astype(dtype=np.uint32)
	Y_test = np.logical_and(Y_test, Yi_test)

X_test_0 = X_test[:,Y_test == 0]
Y_test_0 = Y_test[Y_test == 0]
X_test_1 = X_test[:,Y_test == 1]
Y_test_1 = Y_test[Y_test == 1]
X_test = np.concatenate([X_test_0[:,:Y_test_1.shape[0]], X_test_1], axis=1)
Y_test = np.concatenate([Y_test_0[:Y_test_1.shape[0]], Y_test_1], axis=0)

print(Y_test.sum()/Y_test.shape[0])

print("TEST")

count_11 = 0
count_00 = 0
count_01 = 0
count_10 = 0
count = 0
for e in range(X_test.shape[1]):
	if Y_test[e] == 0:
		if X_test[0,e,0] == 1 and X_test[0,e,1] == 1:
			count_11 += 1

		if X_test[0,e,0] == 0 and X_test[0,e,1] == 0:
			count_00 += 1

		if X_test[0,e,0] == 0 and X_test[0,e,1] == 1:
			count_01 += 1

		if X_test[0,e,0] == 1 and X_test[0,e,1] == 0:
			count_10 += 1

		count += 1

print("Y=0")
print(count_11, count_11/count)
print(count_00, count_00/count)
print(count_01, count_01/count)
print(count_10, count_10/count)

count_11 = 0
count_00 = 0
count_01 = 0
count_10 = 0
count = 0
for e in range(X_test.shape[1]):
	if Y_test[e] == 1:
		if X_test[0,e,0] == 1 and X_test[0,e,1] == 1:
			count_11 += 1

		if X_test[0,e,0] == 0 and X_test[0,e,1] == 0:
			count_00 += 1

		if X_test[0,e,0] == 0 and X_test[0,e,1] == 1:
			count_01 += 1

		if X_test[0,e,0] == 1 and X_test[0,e,1] == 0:
			count_10 += 1

		count += 1

print("Y=1")
print(count_11, count_11/count)
print(count_00, count_00/count)
print(count_01, count_01/count)
print(count_10, count_10/count)

print("TRAINING")

count_11 = 0
count_00 = 0
count_01 = 0
count_10 = 0
count = 0
for e in range(X_train.shape[1]):
	if Y_train[e] == 0:
		if X_train[0,e,0] == 1 and X_train[0,e,1] == 1:
			count_11 += 1

		if X_train[0,e,0] == 0 and X_train[0,e,1] == 0:
			count_00 += 1

		if X_train[0,e,0] == 0 and X_train[0,e,1] == 1:
			count_01 += 1

		if X_train[0,e,0] == 1 and X_train[0,e,1] == 0:
			count_10 += 1

		count += 1

print("Y=0")
print(count_11, count_11/count)
print(count_00, count_00/count)
print(count_01, count_01/count)
print(count_10, count_10/count)

count_11 = 0
count_00 = 0
count_01 = 0
count_10 = 0
count = 0
for e in range(X_train.shape[1]):
	if Y_train[e] == 1:
		if X_train[0,e,0] == 1 and X_train[0,e,1] == 1:
			count_11 += 1

		if X_train[0,e,0] == 0 and X_train[0,e,1] == 0:
			count_00 += 1

		if X_train[0,e,0] == 0 and X_train[0,e,1] == 1:
			count_01 += 1

		if X_train[0,e,0] == 1 and X_train[0,e,1] == 0:
			count_10 += 1

		count += 1

print("Y=1")
print(count_11, count_11/count)
print(count_00, count_00/count)
print(count_01, count_01/count)
print(count_10, count_10/count)

tm = TMMultiChannelClassifier(clauses, global_T, T, s, platform='CPU', boost_true_positive_feedback=0)

for e in range(epochs):
	tm.fit(X_train, Y_train)

#precision = []
#recall = []
#for i in range(2):
#	precision.append((tm.clause_precision(i, 0, X_test, Y_test), tm.clause_precision(i, 1, X_test, Y_test)))
#	recall.append((tm.clause_recall(i, 0, X_test, Y_test), tm.clause_recall(i, 1, X_test, Y_test)))

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("\nClauses:\n")

for c in range(number_of_and_components):
	print("Channel #%d" % (c))

	for j in range(clauses):
		print("\tClause #%d " % (j), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k) == 1:
				if k < number_of_features:
					l.append(" x%d" % (k))
				else:
					l.append("¬x%d" % (k-number_of_features))
		print(" ∧ ".join(l))

		for i in range(2):
			#print("\tC:%d W:%+2d P:%.2f R:%.2f" % (i, tm.get_weight(i, j), precision[i][int(tm.get_weight(i, j) < 0)][j], recall[i][int(tm.get_weight(i, j) < 0)][j]))
			print("\t\tC:%d W:%+2d" % (i, tm.get_weight(i, j)))

print("\nClause Co-Occurence Matrix:\n")
print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

print("\nLiteral Frequency:\n")
print(tm.literal_clause_frequency())

