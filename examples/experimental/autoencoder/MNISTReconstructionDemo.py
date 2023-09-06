import numpy as np
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from keras.datasets import mnist
from skimage import io
from tmu.models.classification.vanilla_classifier import TMClassifier

noise = 0.0

number_of_features = 28*28

number_of_examples = 1000

number_of_clauses = 1024
T = int(number_of_clauses*0.75)
s = 25.0
max_included_literals = 2*number_of_features
accumulation = 1
clause_weight_threshold = 0
upsampling=1

print("Number of clauses:", number_of_clauses)
print("T:", T)
print("Number of features:", number_of_features)

output_active = np.arange(number_of_features, dtype=np.uint32)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)
Y_train = Y_train
Y_test = Y_test
#X_train = np.where(np.random.rand(X_train.shape[0], number_of_features) <= noise, 1-X_train, X_train) # Adds noise

X_test_noisy = np.where(np.random.rand(X_test.shape[0], number_of_features) <= noise, 1-X_test, X_test) # Adds noise

tm = TMAutoEncoder(number_of_clauses, T, s, output_active, max_included_literals=max_included_literals, feature_negation=True, accumulation=accumulation, platform='CPU', output_balancing=0, upsampling=upsampling)

print("\nAccuracy Over 40 Epochs:")
for e in range(150):
	start_training = time()
	tm.fit(X_train, number_of_examples=number_of_examples)
	stop_training = time()

	print("\nEpoch #%d\n" % (e+1))

	X_test_predicted = tm.predict(X_test)
	X_train_predicted = tm.predict(X_train)

	print("Test accuracy", 100*(X_test_predicted == X_test).mean())

	print("Train accuracy", 100*(X_train_predicted == X_train).mean())

	print("\nTraining Time: %.2f" % (stop_training - start_training))

	io.imshow(X_test[0,:].reshape((28,28)))
	io.show()

	io.imshow(X_test_predicted[0,:].reshape((28,28)))
	io.show()

	io.imshow(X_test[1,:].reshape((28,28)))
	io.show()

	io.imshow(X_test_predicted[1,:].reshape((28,28)))
	io.show()

	io.imshow(X_test[2,:].reshape((28,28)))
	io.show()

	io.imshow(X_test_predicted[2,:].reshape((28,28)))
	io.show()

	io.imshow(X_test[3,:].reshape((28,28)))
	io.show()

	io.imshow(X_test_predicted[3,:].reshape((28,28)))
	io.show()
