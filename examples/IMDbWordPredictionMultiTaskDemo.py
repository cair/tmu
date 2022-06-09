import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer

from tmu.tsetlin_machine import TMMultiTaskClassifier

target_words = ['awful', 'terrible', 'lousy', 'brilliant', 'excellent', 'superb']

examples = 20000
context_size = 25
profile_size = 50

clause_drop_p = 0.0

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

X_train = vectorizer_X.fit_transform(training_documents).toarray()
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]

X_train_full = np.zeros((len(target_words), X_train.shape[0], X_train.shape[1]), dtype=np.uint32)
Y_train_full = np.zeros((len(target_words), X_train.shape[0]), dtype=np.uint32)
for i in range(len(target_words)):
	target_word = target_words[i]
	target_id = vectorizer_X.vocabulary_[target_word]
	Y_train_full[i] = np.copy(X_train[:,target_id])
	X_train_full[i] = np.copy(X_train)
	X_train_full[i,:,target_id] = 0

X_train_0 = {}
Y_train_0 = {}
X_train_1 = {}
Y_train_1 = {}
for i in range(len(target_words)):
	X_train_0[i] = X_train_full[i][Y_train_full[i]==0]
	Y_train_0[i]  = Y_train_full[i][Y_train_full[i]==0]
	X_train_1[i]  = X_train_full[i][Y_train_full[i]==1]
	Y_train_1[i]  = Y_train_full[i][Y_train_full[i]==1]

X_train = np.zeros((len(target_words), examples, number_of_features), dtype=np.uint32)
Y_train = np.zeros((len(target_words), examples), dtype=np.uint32)

for i in range(len(target_words)):
	for e in range(examples):
		if np.random.rand() <= 0.5:
			for c in range(context_size):
				X_train[i, e] = np.logical_or(X_train[i, e], X_train_1[i][np.random.randint(X_train_1[i].shape[0])])
			Y_train[i, e] = 1
		else:
			for c in range(context_size):
				X_train[i, e] = np.logical_or(X_train[i, e], X_train_0[i][np.random.randint(X_train_0[i].shape[0])])
			Y_train[i, e] = 0

X_test = vectorizer_X.transform(testing_documents).toarray()

X_test_full = np.zeros((len(target_words), X_test.shape[0], X_test.shape[1]), dtype=np.uint32)
Y_test_full = np.zeros((len(target_words), X_test.shape[0]), dtype=np.uint32)
for i in range(len(target_words)):
	target_word = target_words[i]
	target_id = vectorizer_X.vocabulary_[target_word]
	Y_test_full[i] = np.copy(X_test[:,target_id])
	X_test_full[i] = np.copy(X_test)
	X_test_full[i,:,target_id] = 0

X_test_0 = {}
Y_test_0 = {}
X_test_1 = {}
Y_test_1 = {}
for i in range(len(target_words)):
	X_test_0[i] = X_test_full[i][Y_test_full[i]==0]
	Y_test_0[i]  = Y_test_full[i][Y_test_full[i]==0]
	X_test_1[i]  = X_test_full[i][Y_test_full[i]==1]
	Y_test_1[i]  = Y_test_full[i][Y_test_full[i]==1]

X_test = np.zeros((len(target_words), examples, number_of_features), dtype=np.uint32)
Y_test = np.zeros((len(target_words), examples), dtype=np.uint32)

for i in range(len(target_words)):
	for e in range(examples):
		if np.random.rand() <= 0.5:
			for c in range(context_size):
				X_test[i, e] = np.logical_or(X_test[i, e], X_test_1[i][np.random.randint(X_test_1[i].shape[0])])
			Y_test[i, e] = 1
		else:
			for c in range(context_size):
				X_test[i, e] = np.logical_or(X_test[i, e], X_test_0[i][np.random.randint(X_test_0[i].shape[0])])
			Y_test[i, e] = 0

tm = TMMultiTaskClassifier(clauses, T, s, feature_negation=False, clause_drop_p = clause_drop_p, platform='CPU', weighted_clauses=True)

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
					l.append("%s" % (feature_names[k]))
				else:
					l.append("¬%s" % (feature_names[k-tm.clause_bank.number_of_features]))
		print(" ∧ ".join(l))

	print()
	print("\n#%d Accuracy: %s Training: %.2fs Testing: %.2fs" % (i+1, str(result), stop_training-start_training, stop_testing-start_testing))