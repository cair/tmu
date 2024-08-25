import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from tmu.models.classification.vanilla_classifier import TMClassifier

maxlen = 1000

epochs = 25

batches = 10

hypervector_size = 2048
bits = 1024

clauses = 10000*2
T = 8000
s = 40.0

NUM_WORDS=10000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, maxlen=maxlen, index_from=INDEX_FROM)

train_x, train_y = train
test_x, test_y = test

#train_x = train_x[0:1000]
#train_y = train_y[0:1000]

#test_x = test_x[0:1000]
#test_y = test_y[0:1000]

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

# Read from file instead, otherwise the same

print("Retrieving embeddings...")

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = {}
for i in range(NUM_WORDS+INDEX_FROM):
	encoding[i] = np.random.choice(indexes, size=(bits), replace=False)


# encoding = {}
# f = open("/data/near-lossless-binarization/binary_vectors_1024.vec", "r")
# line = f.readline()
# line = f.readline().strip()
# while line:
# 	entries = line.split(" ")
# 	if entries[0] in word_to_id:
# 		values = np.unpackbits(np.fromstring(" ".join(entries[1:]), dtype=np.int64, sep=' ').view(np.uint8))
# 		encoding[word_to_id[entries[0]]] = np.unpackbits(np.fromstring(" ".join(entries[1:]), dtype=np.int64, sep=' ').view(np.uint8)).nonzero()
# 	line = f.readline().strip()
# f.close()
	
print("Producing bit representation...")

print(train_y.shape[0])
X_train = np.zeros((train_y.shape[0], maxlen, 1, hypervector_size), dtype=np.uint32)
for e in range(train_y.shape[0]):
	position = 0
	for word_id in train_x[e]:
		if word_id in encoding:
			X_train[e, position, 0][encoding[word_id]] = 1
			position += 1

Y_train = train_y.astype(np.uint32)

print(test_y.shape[0])
X_test = np.zeros((test_y.shape[0], maxlen, 1, hypervector_size), dtype=np.uint32)
for e in range(test_y.shape[0]):
	position = 0
	for word_id in test_x[e]:
		if word_id in encoding:
			X_test[e, position, 0][encoding[word_id]] = 1
			position += 1

Y_test = test_y.astype(np.uint32)

batch_size_train = Y_train.shape[0] // batches
batch_size_test = Y_test.shape[0] // batches

tm = TMClassifier(
        clauses,
        T,
        s,
        weighted_clauses=True,
        patch_dim=(1, 1),
        spatio_temporal=False,
        depth=1
    )

for i in range(epochs):
	start_training = time()
	for batch in range(batches):
		tm.fit(X_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train[batch*batch_size_train:(batch+1)*batch_size_train])
	stop_training = time()

	start_testing = time()
	Y_test_predicted = np.zeros(0, dtype=np.uint32)
	for batch in range(batches):
		Y_test_predicted = np.concatenate((Y_test_predicted, tm.predict(X_test[batch*batch_size_test:(batch+1)*batch_size_test])))
	result_test = 100*(Y_test_predicted == Y_test[:batch_size_test*batches]).mean()
	f1_test = f1_score(Y_test[:batch_size_test*batches], Y_test_predicted, average='macro')
	stop_testing = time()

	Y_train_predicted = np.zeros(0, dtype=np.uint32)
	for batch in range(batches):
		Y_train_predicted = np.concatenate((Y_train_predicted, tm.predict(X_train[batch*batch_size_train:(batch+1)*batch_size_train])))
	result_train = 100*(Y_train_predicted == Y_train[:batch_size_train*batches]).mean()

	f1_train = f1_score(Y_train[:batch_size_train*batches], Y_train_predicted, average='macro')

	print("#%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))