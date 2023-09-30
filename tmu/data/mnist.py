from typing import Dict
import numpy as np
from tmu.data import TMUDataset
from tmu.data.utils.downloader import get_file


class MNIST(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        origin_folder = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        )

        path = get_file(
            "mnist.npz",
            origin=origin_folder + "mnist.npz",
            file_hash=(  # noqa: E501
                "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
            ),
        )

        with np.load(path, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        return dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 28 * 28)) > 75, 1, 0)
