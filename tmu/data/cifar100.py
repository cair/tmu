import numpy as np
from typing import Dict
from tmu.data import TMUDataset
from tmu.data.utils.downloader import get_file


class CIFAR100(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        origin_folder = "https://www.cs.toronto.edu/~kriz/"

        path = get_file(
            "cifar-100-python.tar.gz",
            origin=origin_folder + "cifar-100-python.tar.gz",
            extract=True,
            extract_archive_format="tar",
            file_hash=(
                '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
            )
        )
        path = path + "/cifar-100-python"

        # Load the data from the files.
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        # Loading train data
        train_data = unpickle(path + "/train")
        x_train = train_data[b'data'].reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
        y_train = np.array(train_data[b'fine_labels'])  # CIFAR-100 has fine and coarse labels, using fine labels here

        # Loading test data
        test_data = unpickle(path + "/test")
        x_test = test_data[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        y_test = np.array(test_data[b'fine_labels'])  # Using fine labels

        return dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        return dataset

if __name__ == "__main__":
    cifar_ds = CIFAR100()
    data = cifar_ds.get()
    print(data)
