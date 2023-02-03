import abc
from typing import Dict
from collections import namedtuple
import numpy as np



class TMUDataset:

    def __init__(self):
        pass

    @abc.abstractmethod
    def _transform(self, name, dataset):
        raise NotImplementedError("You should override def _transform()")

    @abc.abstractmethod
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("You should override def _retrieve_dataset()")

    def get(self):
        return {k: self._transform(k, v) for k, v in self._retrieve_dataset().items()}

    def get_list(self):
        return list(self.get().values())


class MNISTNew(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        from datasets import load_dataset
        dataset = load_dataset('mnist', split=['train', 'test'], cache_dir=".")
        return dataset

    def _transform(self, split, dataset):
        def pil_image_to_array(batch):
            images = np.asarray([np.asarray(img) for img in batch["image"]])
            return {
                "image": np.where(images.reshape((images.shape[0], 28*28)) > 75, 1, 0)
            }

        dataset.set_transform(pil_image_to_array, columns="image", output_all_columns=True)
        return dataset


class MNIST(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        from keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 28*28)) > 75, 1, 0)

