from tmu.composite.tuner import TMCompositeTuner
from tmu.data.cifar10 import CIFAR10
import numpy as np


def create_subset(X, Y, samples_per_class):
    """
    Create a balanced subset of the data with an equal number of samples per class.

    Parameters:
    - X: numpy array, feature matrix with shape (n_samples, n_features)
    - Y: numpy array, labels with shape (n_samples,) where each element is an integer class label
    - samples_per_class: int, the number of samples to include from each class

    Returns:
    - A tuple of two numpy arrays: (subset_X, subset_Y) shuffled and balanced subset.
    """
    unq = len(np.unique(Y))

    subset_X, subset_Y = [], []

    for i in range(unq):  # Assuming 10 classes in CIFAR-10
        mask = (Y == i).reshape(-1)
        subset_X.append(X[mask][:samples_per_class])
        subset_Y.append(Y[mask][:samples_per_class])

    # Using np.concatenate for clarity
    subset_X, subset_Y = np.concatenate(subset_X), np.concatenate(subset_Y)

    # Shuffling the dataset
    indices = np.arange(subset_X.shape[0])
    np.random.shuffle(indices)

    return subset_X[indices], subset_Y[indices]


if __name__ == "__main__":
    data = CIFAR10().get()
    X_train_org = data["x_train"]
    Y_train = data["y_train"]
    X_test_org = data["x_test"]
    Y_test = data["y_test"]
    percentage = 0.1
    num_train_samples = int(5000 * percentage)
    num_test_samples = int(1000 * percentage)

    X_train_subset, Y_train_subset = create_subset(X_train_org, Y_train, num_train_samples)
    X_test_subset, Y_test_subset = create_subset(X_test_org, Y_test, num_test_samples)
    Y_test_subset = Y_test_subset.reshape(Y_test_subset.shape[0])
    Y_train_subset = Y_train_subset.reshape(Y_train_subset.shape[0])
    data_train = dict(X=X_train_subset, Y=Y_train_subset)
    data_test = dict(X=X_test_subset, Y=Y_test_subset)

    # Instantiate tuner
    tuner = TMCompositeTuner(
        data_train=data_train,
        data_test=data_test,
        n_jobs=1  # for parallelization; set to 1 for no parallelization
    )

    # Specify number of trials (iterations of the tuning process)
    n_trials = 100

    # Run the tuner
    best_params, best_value = tuner.tune(n_trials=n_trials)

    # Print out the results
    print("Best Parameters:", best_params)
    print("Best Value:", best_value)
