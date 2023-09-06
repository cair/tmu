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