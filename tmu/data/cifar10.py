from typing import Dict
import numpy as np
from tmu.data import TMUDataset
from tmu.data.utils.downloader import get_file


class CIFAR10(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        origin_folder = (
            "https://www.cs.toronto.edu/~kriz/"
        )

        path = get_file(
            "cifar-10-python.tar.gz",
            origin=origin_folder + "cifar-10-python.tar.gz",
            extract=True,
            extract_archive_format="tar",
            file_hash=(
                '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'
            )
        )
        path = path + "/cifar-10-batches-py"

        # Load the data from the files. Note: you might need to change the file paths based on the extraction
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        # Loading train data
        x_train = []
        y_train = []
        for i in range(1, 6):  # CIFAR-10 has 5 batches of training data
            batch_data = unpickle(path + f"/data_batch_{i}")
            x_train.append(batch_data[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1))
            y_train.append(np.array(batch_data[b'labels']))
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        # Loading test data
        test_data = unpickle(path + "/test_batch")
        x_test = test_data[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        y_test = np.array(test_data[b'labels'])

        return dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        return dataset


if __name__ == "__main__":

    cifar_ds = CIFAR10()
    cifar_ds.get()

