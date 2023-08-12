import pytest


class TestMNISTDatasetobject:

    def setup_class(self):
        from tmu.data import MNIST
        self.dataset_instance = MNIST()

    def teardown_class(self):
        self.dataset_instance = None

    def test_mnist_dict(self):
        dataset = self.dataset_instance.get()
        assert len(dataset) == 4
        assert "x_train" in dataset
        assert "y_train" in dataset
        assert "x_test" in dataset
        assert "y_test" in dataset
        assert len(dataset["x_train"]) == 60000
        assert len(dataset["y_train"]) == 60000
        assert len(dataset["x_test"]) == 10000
        assert len(dataset["y_test"]) == 10000

    def test_mnist_list(self):
        dataset = self.dataset_instance.get_list()
        assert len(dataset) == 4
        assert isinstance(dataset, list)


class TestFashionMNISTDataset:

    def setup_class(self):
        from tmu.data.fashion_mnist import FashionMNIST
        self.dataset_instance = FashionMNIST()

    def teardown_class(self):
        self.dataset_instance = None

    def test_fashion_mnist_dict(self):
        dataset = self.dataset_instance.get()
        assert len(dataset) == 4
        assert "x_train" in dataset
        assert "y_train" in dataset
        assert "x_test" in dataset
        assert "y_test" in dataset
        assert len(dataset["x_train"]) == 60000
        assert len(dataset["y_train"]) == 60000
        assert len(dataset["x_test"]) == 10000
        assert len(dataset["y_test"]) == 10000

    def test_fashion_mnist_list(self):
        dataset = self.dataset_instance.get_list()
        assert len(dataset) == 4
        assert isinstance(dataset, list)


class TestKuzushijiMNISTDataset:

    def setup_class(self):
        from tmu.data.fashion_mnist import KuzushijiMNIST
        self.dataset_instance = KuzushijiMNIST()

    def teardown_class(self):
        self.dataset_instance = None

    def test_kuzushiji_mnist_dict(self):
        dataset = self.dataset_instance.get()
        assert len(dataset) == 4
        assert "x_train" in dataset
        assert "y_train" in dataset
        assert "x_test" in dataset
        assert "y_test" in dataset
        assert len(dataset["x_train"]) == 60000
        assert len(dataset["y_train"]) == 60000
        assert len(dataset["x_test"]) == 10000
        assert len(dataset["y_test"]) == 10000

    def test_kuzushiji_mnist_list(self):
        dataset = self.dataset_instance.get_list()
        assert len(dataset) == 4
        assert isinstance(dataset, list)


class TestCIFAR100Dataset:

    def setup_class(self):
        from tmu.data.fashion_mnist import CIFAR100
        self.dataset_instance = CIFAR100()

    def teardown_class(self):
        self.dataset_instance = None

    def test_cifar100_dict(self):
        dataset = self.dataset_instance.get()
        assert len(dataset) == 4
        assert "x_train" in dataset
        assert "y_train" in dataset
        assert "x_test" in dataset
        assert "y_test" in dataset
        # adjust the following counts according to your train-test split
        assert len(dataset["x_train"]) == 50000
        assert len(dataset["y_train"]) == 50000
        assert len(dataset["x_test"]) == 10000
        assert len(dataset["y_test"]) == 10000

    def test_cifar100_list(self):
        dataset = self.dataset_instance.get_list()
        assert len(dataset) == 4
        assert isinstance(dataset, list)
