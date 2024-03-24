import unittest

class TestMNISTDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from tmu.data import MNIST
        cls.dataset_instance = MNIST()

    @classmethod
    def tearDownClass(cls):
        cls.dataset_instance = None

    def test_mnist_dict(self):
        dataset = self.dataset_instance.get()
        self.assertEqual(len(dataset), 4)
        self.assertIn("x_train", dataset)
        self.assertIn("y_train", dataset)
        self.assertIn("x_test", dataset)
        self.assertIn("y_test", dataset)
        self.assertEqual(len(dataset["x_train"]), 60000)
        self.assertEqual(len(dataset["y_train"]), 60000)
        self.assertEqual(len(dataset["x_test"]), 10000)
        self.assertEqual(len(dataset["y_test"]), 10000)

    def test_mnist_list(self):
        dataset = self.dataset_instance.get_list()
        self.assertEqual(len(dataset), 4)
        self.assertIsInstance(dataset, list)


class TestFashionMNISTDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from tmu.data.fashion_mnist import FashionMNIST
        cls.dataset_instance = FashionMNIST()

    @classmethod
    def tearDownClass(cls):
        cls.dataset_instance = None

    def test_fashion_mnist_dict(self):
        dataset = self.dataset_instance.get()
        self.assertEqual(len(dataset), 4)
        self.assertIn("x_train", dataset)
        self.assertIn("y_train", dataset)
        self.assertIn("x_test", dataset)
        self.assertIn("y_test", dataset)
        self.assertEqual(len(dataset["x_train"]), 60000)
        self.assertEqual(len(dataset["y_train"]), 60000)
        self.assertEqual(len(dataset["x_test"]), 10000)
        self.assertEqual(len(dataset["y_test"]), 10000)

    def test_fashion_mnist_list(self):
        dataset = self.dataset_instance.get_list()
        self.assertEqual(len(dataset), 4)
        self.assertIsInstance(dataset, list)


class TestKuzushijiMNISTDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from tmu.data.fashion_mnist import KuzushijiMNIST
        cls.dataset_instance = KuzushijiMNIST()

    @classmethod
    def tearDownClass(cls):
        cls.dataset_instance = None

    def test_kuzushiji_mnist_dict(self):
        dataset = self.dataset_instance.get()
        self.assertEqual(len(dataset), 4)
        self.assertIn("x_train", dataset)
        self.assertIn("y_train", dataset)
        self.assertIn("x_test", dataset)
        self.assertIn("y_test", dataset)
        self.assertEqual(len(dataset["x_train"]), 60000)
        self.assertEqual(len(dataset["y_train"]), 60000)
        self.assertEqual(len(dataset["x_test"]), 10000)
        self.assertEqual(len(dataset["y_test"]), 10000)

    def test_kuzushiji_mnist_list(self):
        dataset = self.dataset_instance.get_list()
        self.assertEqual(len(dataset), 4)
        self.assertIsInstance(dataset, list)


class TestCIFAR100Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from tmu.data.fashion_mnist import CIFAR100
        cls.dataset_instance = CIFAR100()

    @classmethod
    def tearDownClass(cls):
        cls.dataset_instance = None

    def test_cifar100_dict(self):
        dataset = self.dataset_instance.get()
        self.assertEqual(len(dataset), 4)
        self.assertIn("x_train", dataset)
        self.assertIn("y_train", dataset)
        self.assertIn("x_test", dataset)
        self.assertIn("y_test", dataset)
        self.assertEqual(len(dataset["x_train"]), 50000)
        self.assertEqual(len(dataset["y_train"]), 50000)
        self.assertEqual(len(dataset["x_test"]), 10000)
        self.assertEqual(len(dataset["y_test"]), 10000)

    def test_cifar100_list(self):
        dataset = self.dataset_instance.get_list()
        self.assertEqual(len(dataset), 4)
        self.assertIsInstance(dataset, list)
