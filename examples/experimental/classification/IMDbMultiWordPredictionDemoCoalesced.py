import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

#target_words = ['masterpiece', 'brilliant', 'comedy', 'scary', 'funny', 'hate', 'love', 'awful', 'terrible']

target_words = ['awful', 'scary', 'brilliant']

#target_words = ['awful', 'terrible', 'brilliant']

examples = 20000
context_size = 25
profile_size = 50

clause_drop_p = 0.0

clauses = int(20*len(target_words)/(1.0 - clause_drop_p))
T = 40
s = 5.0

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

X_train_full = vectorizer_X.fit_transform(training_documents).toarray()
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]

target_ids_list = []
for target_word in target_words:
	target_ids_list.append(vectorizer_X.vocabulary_[target_word])
target_ids = np.array(target_ids_list)

Y_train_multi = np.copy(X_train_full[:,target_ids])
X_train_full[:,target_ids] = 0

X_train = np.zeros((examples, number_of_features), dtype=np.uint32)
Y_train = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	target_class = np.random.choice(np.arange(target_ids.shape[0]))
	target_rows = np.where(Y_train_multi[:,target_class] == 1)[0]
	for c in range(context_size):
		X_train[i] = np.logical_or(X_train[i], X_train_full[np.random.choice(target_rows)])
	Y_train[i] = target_class

X_test_full = vectorizer_X.transform(testing_documents).toarray()
Y_test_multi = np.copy(X_test_full[:,target_ids])
X_test_full[:,target_ids] = 0

X_test = np.zeros((examples, number_of_features), dtype=np.uint32)
Y_test = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	target_class = np.random.choice(np.arange(target_ids.shape[0]))
	target_rows = np.where(Y_test_multi[:,target_class] == 1)[0]
	for c in range(context_size):
		X_test[i] = np.logical_or(X_test[i], X_test_full[np.random.choice(target_rows)])
	Y_test[i] = target_class

tm = TMCoalescedClassifier(clauses, T, s, feature_negation=False, clause_drop_p=clause_drop_p, platform='CPU', weighted_clauses=True)

print("\nAccuracy Over 40 Epochs:\n")
for i in range(40):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("\n********** Epoch %d **********" % (i+1))

	for j in range(clauses):
		print("\n\tClause #%d:" % (j), end=' ')
		l = []
		for k in range(number_of_features*2):
			if tm.get_ta_action(j, k) == 1:
				if k < number_of_features:
					l.append("%s" % (feature_names[k]))
				else:
					l.append("¬%s" % (feature_names[k-number_of_features]))
		print(" ∧ ".join(l))

		for target_class in range(target_ids.shape[0]):
			print("\t\t%s: %+2d" % (target_words[target_class], tm.get_weight(target_class, j)))

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))


