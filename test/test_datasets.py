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

