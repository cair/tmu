import typing

import numpy as np


class StandardBinarizer:
    """
    The standard TM binarizer is detailed in https://arxiv.org/pdf/1905.04199.pdf, Section 3.3
    Hyperparameters:
        max_bits_per_feature: how many threshold values each feature should use.

    Procedure:
    """

    number_of_features: int
    max_bits_per_feature: int
    unique_values: typing.List[np.ndarray]

    def __init__(self, max_bits_per_feature=25):
        self.max_bits_per_feature = max_bits_per_feature
        return

    def fit(self, X):
        self.number_of_features = 0
        self.unique_values = []
        for i in range(X.shape[1]):
            uv = np.unique(X[:, i])[1:]

            if uv.size > self.max_bits_per_feature:
                unique_values = np.empty(0)

                step_size = 1.0 * uv.size / self.max_bits_per_feature
                pos = 0.0
                while int(pos) < uv.size and unique_values.size < self.max_bits_per_feature:
                    unique_values = np.append(unique_values, np.array(uv[int(pos)]))
                    pos += step_size
            else:
                unique_values = uv

            self.unique_values.append(unique_values)
            self.number_of_features += self.unique_values[-1].size
        return

    def transform(self, X):
        X_transformed = np.zeros((X.shape[0], self.number_of_features))

        pos = 0
        for i in range(X.shape[1]):
            for j in range(self.unique_values[i].size):
                X_transformed[:, pos] = (X[:, i] >= self.unique_values[i][j])
                pos += 1

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ThresholdBasedBooleanizer:

    def __init__(self, threshold_bits_per_feature=25):
        self.threshold_bits_per_feature = threshold_bits_per_feature
        self.thresholds = []

    def fit(self, X, axis=0):
        # Handling the case when X is 1D array or single feature
        if len(X.shape) == 1 or X.shape[1] == 1:
            axis = 0

        # Feature-wise
        if axis == 1:
            self.thresholds = []
            for feature in X.T:
                unique_values = np.unique(feature)
                step_size = len(unique_values) // self.threshold_bits_per_feature
                if step_size == 0:
                    # Threshold bits are configured higher than the number of unique values
                    self.thresholds.append(unique_values.tolist())
                else:
                    indices = np.arange(0, len(unique_values), step_size)
                    self.thresholds.append(unique_values[indices].tolist())

        # Dataset-wise
        else:
            X_flat = X.flatten()
            unique_values = np.unique(X_flat)
            step_size = len(unique_values) // self.threshold_bits_per_feature
            if step_size == 0:
                self.thresholds = unique_values.tolist()
            else:
                self.thresholds = [unique_values[i] for i in range(0, len(unique_values), step_size)]

    def transform(self, X, axis=0):
        # Handling the case when X is 1D array or single feature
        if len(X.shape) == 1 or X.shape[1] == 1:
            axis = 0

        # Feature-wise
        if axis == 1:
            encoded = []
            for feature, thresholds in zip(X.T, self.thresholds):
                encoded.append([[int(x_i >= threshold) for threshold in thresholds] for x_i in feature])
            return np.array(encoded).T

        # Dataset-wise
        else:
            encoded = []
            for x_i in X.flatten():
                bits = [int(x_i >= threshold) for threshold in self.thresholds]
                encoded.append(bits)
            return encoded

    def fit_transform(self, X, axis=0):
        self.fit(X, axis)
        return self.transform(X, axis)



if __name__ == "__main__":
    import tmu.data
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    data = tmu.data.MNIST().get()
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=10000)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    binarizer = StandardBinarizer(max_bits_per_feature=5)
    binarizer.fit(X_train)
