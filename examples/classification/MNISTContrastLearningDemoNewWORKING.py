import numpy as np
from time import time
from keras.datasets import mnist
from skimage.transform import rotate
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

examples = 500#2500#1000#5000#1000#2500#5000
number_of_random_labels = examples//5

clauses = int(250)
max_positive_clauses = clauses

T = int((np.sqrt(clauses)/2 + 2)*10)
s = 2.0

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

shuffled_index = np.arange(X_train.shape[0])
np.random.shuffle(shuffled_index)
X_contrast = X_train[shuffled_index][:examples]
Y_contrast = np.arange(examples, dtype=np.uint32)
#Y_contrast = np.random.randint(number_of_random_labels, size=examples).astype(np.uint32)

#tm = TMCoalescedClassifier(clauses, T, s, platform='CPU', max_positive_clauses=max_positive_clauses, max_included_literals=8, weighted_clauses=True)
tm = TMCoalescedClassifier(clauses, T, s, platform='CPU', focused_negative_sampling=True, max_included_literals=16, weighted_clauses=True)

print("Starting contrast learning")

for e in range(50):
        start_training = time()
        tm.fit(X_contrast, Y_contrast)
        stop_training = time()

        print("#%d Training: %.2fs" % (e+1, stop_training-start_training))

# start_testing = time()
# result = 100*(tm.predict(X_contrast) == Y_contrast).mean()
# stop_testing = time()

# number_of_includes = 0
# for j in range(clauses):
#         number_of_includes += tm.number_of_include_actions(j)
# number_of_includes /= 2*clauses

# print("#%d Accuracy: %.2f%% Includes: %.1f Training: %.2fs Testing: %.2fs" % (e+1, result, number_of_includes, stop_training-start_training, stop_testing-start_testing))

X_train_transformed = tm.transform(X_train)
X_test_transformed = tm.transform(X_test)

print("\nFINAL\n")

tm_final = TMClassifier(int(2000), int(50*100), 10.0, platform='CPU', weighted_clauses=True)
for e in range(500):
        start_training = time()
        tm_final.fit(X_train_transformed, Y_train)
        stop_training = time()

        start_testing = time()
        result = 100*(tm_final.predict(X_test_transformed) == Y_test).mean()
        stop_testing = time()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))



