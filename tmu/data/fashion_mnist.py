from tmu.data.tmu_dataset import TMUDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import logging
import sklearn
from typing import Dict
import numpy as np
from packaging.version import parse as parse_version

_LOGGER = logging.getLogger(__name__)


class FashionMNIST(TMUDataset):

    def __init__(self):
        super().__init__()
        _LOGGER.warning("Threshold function is not tested yet.")

    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        kwargs = dict()

        # Parse the sklearn version string
        sklearn_version = parse_version(sklearn.__version__)

        # Check if the major version is >= 1 and the minor version is >= 2
        if sklearn_version >= parse_version("1.2"):
            kwargs["parser"] = "pandas"

        X, y = fetch_openml(
            "Fashion-MNIST",
            version=1,
            return_X_y=True,
            as_frame=False,
            **kwargs
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=10000)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        return dict(
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 28 * 28)) > 75, 1, 0)


class KuzushijiMNIST(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        kwargs = dict()
        # Parse the sklearn version string
        sklearn_version = parse_version(sklearn.__version__)

        # Check if the major version is >= 1 and the minor version is >= 2
        if sklearn_version >= parse_version("1.2"):
            kwargs["parser"] = "pandas"

        X, y = fetch_openml(
            "Kuzushiji-MNIST",
            version=1,
            return_X_y=True,
            as_frame=False,
            **kwargs
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=10000)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        return dict(
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 28 * 28)) > 75, 1, 0)


class CIFAR100(TMUDataset):

    def __init__(self):
        super().__init__()
        _LOGGER.warning("Threshold function is not implemented. Use add_transform(fn) to use custom transform.")

    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        kwargs = dict()
        # Parse the sklearn version string
        sklearn_version = parse_version(sklearn.__version__)

        # Check if the major version is >= 1 and the minor version is >= 2
        if sklearn_version >= parse_version("1.2"):
            kwargs["parser"] = "pandas"

        X, y = fetch_openml(
            "CIFAR-100",  # name of CIFAR-100 on OpenML
            version=1,
            return_X_y=True,
            as_frame=False,
            **kwargs
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=10000)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        return dict(
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 32 * 32 * 3)) > 75, 1, 0)
