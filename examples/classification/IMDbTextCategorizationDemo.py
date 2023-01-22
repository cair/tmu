import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer

from tmu.models.classification.vanilla_classifier import TMClassifier

MAX_NGRAM = 2
FEATURES = 5000

NUM_WORDS=5000
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

vectorizer_X = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,MAX_NGRAM), lowercase=False, binary=True)

X_train = vectorizer_X.fit_transform(training_documents)
Y_train = train_y.astype(np.uint32)

X_test = vectorizer_X.transform(testing_documents)
Y_test = test_y.astype(np.uint32)

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train).toarray()
X_test = SKB.transform(X_test).toarray()

tm = TMClassifier(10000, 8000, 2.0, platform='CUDA', weighted_clauses=True, clause_drop_p=0.75)

print("\nAccuracy over 40 epochs:\n")
for i in range(40):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
