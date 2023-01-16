import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from tmu.models.classification.multitask_classifier import TMMultiTaskClassifier

target_words = ['awful', 'terrible', 'lousy', 'abysmal', 'crap', 'outstanding', 'brilliant', 'excellent', 'superb', 'magnificent', 'marvellous', 'truck', 'plane', 'car', 'cars', 'motorcycle',  'scary', 'frightening', 'terrifying', 'horrifying', 'funny', 'comic', 'hilarious', 'witty']

number_of_examples = 2000
context_size = 25

clause_weight_threshold = 0

max_included_literals = 3

confidence_driven_updating = False

clause_drop_p = 0.75

clauses = int(20/(1.0 - clause_drop_p))
T = 40
s = 5.0

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

X_train_csr = vectorizer_X.fit_transform(training_documents)
X_train_csc = X_train_csr.tocsc()
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]

X_train = {}
Y_train = {}
for i in range(len(target_words)):
	X_train[i] = lil_matrix((number_of_examples, number_of_features), dtype=np.uint32)
	Y_train[i] = lil_matrix(number_of_examples, dtype=np.uint32)
	for e in range(number_of_examples):
		target_word = target_words[i]
		target_id = vectorizer_X.vocabulary_[target_word]
		target = np.random.choice(2)
		if target == 1:
			target_indices = X_train_csc[:,target_id].indices
		else:
			target_indices = np.setdiff1d(np.arange(X_train_csr.shape[0], dtype=np.uint32), X_train_csc[:,target_id].indices)
		examples = np.random.choice(target_indices, size=context_size, replace=True)
		X_train[i][e,:] = (X_train_csr[examples].toarray().sum(axis=0) > 0).astype(np.uint32)
		X_train[i][e,target_id] = 1
		Y_train[i][e] = target
	X_train[i] = X_train[i].tocsr()
	Y_train[i] = Y_train[i].tocsr()

X_test_csr = vectorizer_X.transform(testing_documents)
X_test_csc = X_test_csr.tocsc()

X_test = {}
Y_test = {}
for i in range(len(target_words)):
	X_test[i] = lil_matrix((number_of_examples, number_of_features), dtype=np.uint32)
	Y_test[i] = lil_matrix(number_of_examples, dtype=np.uint32)
	for e in range(number_of_examples):
		target_word = target_words[i]
		target_id = vectorizer_X.vocabulary_[target_word]
		target = np.random.choice(2)
		if target == 1:
			target_indices = X_test_csc[:,target_id].indices
		else:
			target_indices = np.setdiff1d(np.arange(X_test_csr.shape[0], dtype=np.uint32), X_test_csc[:,target_id].indices)
		examples = np.random.choice(target_indices, size=context_size, replace=True)
		X_test[i][e,:] = (X_test_csr[examples].toarray().sum(axis=0) > 0).astype(np.uint32)
		X_test[i][e,target_id] = 1
		Y_test[i][e] = target
	X_test[i] = X_test[i].tocsr()
	Y_test[i] = Y_test[i].tocsr()

tm = TMMultiTaskClassifier(clauses, T, s, confidence_driven_updating=confidence_driven_updating, max_included_literals=max_included_literals, feature_negation=False, clause_drop_p = clause_drop_p, platform='CPU', weighted_clauses=True)

print("\nAccuracy Over 40 Epochs:")
for e in range(40):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	prediction = tm.predict(X_test)
	result = []
	for i in range(len(target_words)):
		result.append(100*(prediction[i] == Y_test[i]).mean())
	stop_testing = time()

	for j in range(clauses):
		print("Clause #%d " % (j), end=' ')
		for i in range(len(target_words)):
			print("%s:%d " % (target_words[i], tm.get_weight(i, j)), end=' ')

		l = []
		for k in range(tm.clause_bank.number_of_literals):
			if tm.get_ta_action(j, k) == 1:
				if k < tm.clause_bank.number_of_features:
					l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
				else:
					l.append("¬%s(%d)" % (feature_names[k-tm.clause_bank.number_of_features], tm.clause_bank.get_ta_state(j, k)))
		print(" ∧ ".join(l))

	profile = np.empty((len(target_words), clauses))
	for i in range(len(target_words)):
		weights = tm.get_weights(i)
		profile[i,:] = np.where(weights >= clause_weight_threshold, weights, 0)

	similarity = cosine_similarity(profile)

	print("\nWord Similarity\n")

	for i in range(len(target_words)):
		print(target_words[i], end=': ')
		sorted_index = np.argsort(-1*similarity[i,:])
		for j in range(1, len(target_words)):
			print("%s(%.2f) " % (target_words[sorted_index[j]], similarity[i,sorted_index[j]]), end=' ')
		print()

	print("\n#%d Accuracy: %s Training: %.2fs Testing: %.2fs" % (e+1, str(result), stop_training-start_training, stop_testing-start_testing))
	print()
