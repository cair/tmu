import abc
from typing import Tuple, Any

import numpy as np


class TMUDataset:

    def __init__(self):
        pass

    @abc.abstractmethod
    def booleanizer(self, name, dataset):
        raise NotImplementedError("You should override def threshold()")

    @abc.abstractmethod
    def retrieve_dataset(self) -> dict[str, np.ndarray]:
        raise NotImplementedError("You should override def retrieve_dataset()")

    def get(self):
        return {k: self.booleanizer(k, v) for k, v in self.retrieve_dataset().items()}

    def get_list(self):
        return list(self.get().values())


class MNIST(TMUDataset):
    def retrieve_dataset(self) -> dict[str, np.ndarray]:
        from keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )

    def booleanizer(self, name, dataset):
        if name.startswith("y"):
            return dataset
        
        return np.where(dataset.reshape((dataset.shape[0], 28*28)) > 75, 1, 0)

