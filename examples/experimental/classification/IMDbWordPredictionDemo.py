import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer

from tmu.models.classification.vanilla_classifier import TMClassifier

target_word = 'car' #'frightening'#'comedy'#'romance'#"scary"
#target_word = 'brilliant'

examples = 20000
context_size = 1
profile_size = 50

max_included_literals = 1

type_i_ii_ratio = 1.0

clause_drop_p = 0.0

factor = 4
clauses = int(factor*20/(1.0 - clause_drop_p))
T = 3
s = 150.0

print("Number of clauses:", clauses)

NUM_WORDS=10000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

id_to_word = {value:key for key,value in word_to_id.items()}

training_documents = []
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id].lower())

	training_documents.append(terms)

testing_documents = []
for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id].lower())

	testing_documents.append(terms)

def tokenizer(s):
	return s

vectorizer_X = CountVectorizer(tokenizer=tokenizer, lowercase=False, max_features=NUM_WORDS, binary=True)

X_train = vectorizer_X.fit_transform(training_documents).toarray()
feature_names = vectorizer_X.get_feature_names_out()

number_of_features = vectorizer_X.get_feature_names_out().shape[0]
target_id = vectorizer_X.vocabulary_[target_word]
Y_train = np.copy(X_train[:,target_id])
X_train[:,target_id] = 0

X_train_0 = X_train[Y_train==0]
Y_train_0 = Y_train[Y_train==0]
X_train_1 = X_train[Y_train==1]
Y_train_1 = Y_train[Y_train==1]

print("Number of Target Words:", Y_train_1.shape[0])

X_train = np.zeros((examples, number_of_features), dtype=np.uint32)
Y_train = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	if np.random.rand() <= 0.5:
		for c in range(context_size):
			X_train[i] = np.logical_or(X_train[i], X_train_1[np.random.randint(X_train_1.shape[0])])
		Y_train[i] = 1
	else:
		for c in range(context_size):
			X_train[i] = np.logical_or(X_train[i], X_train_0[np.random.randint(X_train_0.shape[0])])
		Y_train[i] = 0

X_test = vectorizer_X.transform(testing_documents).toarray()
Y_test = np.copy(X_test[:,target_id])
X_test[:,target_id] = 0

X_test_0 = X_test[Y_test==0]
Y_test_0 = Y_test[Y_test==0]
X_test_1 = X_test[Y_test==1]
Y_test_1 = Y_test[Y_test==1]
X_test = np.zeros((examples, number_of_features), dtype=np.uint32)
Y_test = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	if np.random.rand() <= 0.5:
		for c in range(context_size):
			X_test[i] = np.logical_or(X_test[i], X_test_1[np.random.randint(X_test_1.shape[0])])
		Y_test[i] = 1
	else:
		for c in range(context_size):
			X_test[i] = np.logical_or(X_test[i], X_test_0[np.random.randint(X_test_0.shape[0])])
		Y_test[i] = 0

tm = TMClassifier(clauses, T, s,  max_included_literals = max_included_literals, feature_negation = False, type_i_ii_ratio = type_i_ii_ratio, clause_drop_p = clause_drop_p, platform='CPU', weighted_clauses=False)

print("\nAccuracy Over 100 Epochs:")
for i in range(100):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("\nClass 0 Positive Clauses:\n")

	precision = tm.clause_precision(0, 0, X_test, Y_test)
	recall = tm.clause_recall(0, 0, X_test, Y_test)

	for j in range(clauses//2):
		print("Clause #%d W:%d P:%.2f R:%.3f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k, the_class = 0, polarity = 0):
				if k < number_of_features:
					l.append(" %s(%d)" % (feature_names[k], tm.get_ta_state(j, k, the_class = 0, polarity = 0)))
				else:
					l.append("¬%s(%d)" % (feature_names[k-number_of_features], tm.get_ta_state(j, k, the_class = 0, polarity = 0) ))
		print(" ∧ ".join(l))

	print("\nClass 0 Negative Clauses:\n")

	precision = tm.clause_precision(0, 1, X_test, Y_test)
	recall = tm.clause_recall(0, 1, X_test, Y_test)

	for j in range(clauses//2):
		print("Clause #%d W:%d P:%.2f R:%.3f " % (j, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k, the_class = 0, polarity = 1):
				if k < number_of_features:
					l.append(" %s(%d)" % (feature_names[k], tm.get_ta_state(j, k, the_class = 0, polarity = 1)))
				else:
					l.append("¬%s(%d)" % (feature_names[k-number_of_features], tm.get_ta_state(j, k, the_class = 0, polarity = 1)))
		print(" ∧ ".join(l))

	print("\nClass 1 Positive Clauses:\n")

	precision = tm.clause_precision(1, 0, X_test, Y_test)
	recall = tm.clause_recall(1, 0, X_test, Y_test)

	for j in range(clauses//2):
		print("Clause #%d W:%d P:%.2f R:%.3f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k, the_class = 1, polarity = 0):
				if k < number_of_features:
					l.append(" %s(%d)" % (feature_names[k], tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
				else:
					l.append("¬%s(%d)" % (feature_names[k-number_of_features], tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
		print(" ∧ ".join(l))

	print("\nClass 1 Negative Clauses:\n")

	precision = tm.clause_precision(1, 1, X_test, Y_test)
	recall = tm.clause_recall(1, 1, X_test, Y_test)

	for j in range(clauses//2):
		print("Clause #%d W:%d P:%.2f R:%.3f " % (j, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k, the_class = 1, polarity = 1):
				if k < number_of_features:
					l.append(" %s(%d)" % (feature_names[k], tm.get_ta_state(j, k, the_class = 1, polarity = 1)))
				else:
					l.append("¬%s(%d)" % (feature_names[k-number_of_features], tm.get_ta_state(j, k, the_class = 1, polarity = 1)))
		print(" ∧ ".join(l))

	print("\nPositive Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break
		print(feature_names[k], end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break
		print("¬" + feature_names[k - number_of_features], end=' ')

	print()
	print("\nNegative Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break
		print(feature_names[k], end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break
		print("¬" + feature_names[k - number_of_features], end=' ')

	print()
	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
