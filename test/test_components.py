import unittest
from tmu.composite.components import image as image_components
from tmu.composite.components.base import TMComponent
from tmu.composite.composite import TMComposite
from tmu.composite.config import TMClassifierConfig
from tmu.data.cifar10 import CIFAR10
from tmu.models.classification.vanilla_classifier import TMClassifier


class TestTMCompositeImageComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load CIFAR-10 dataset
        cifar_ds = CIFAR10()
        data = cifar_ds.get()

        X_train_org = data["x_train"]
        Y_train = data["y_train"]
        X_test_org = data["x_test"]
        Y_test = data["y_test"]


        Y_test = Y_test.reshape(Y_test.shape[0])
        Y_train = Y_train.reshape(Y_train.shape[0])
        X_train_org = X_train_org[:50]
        Y_train = Y_train[:50]
        X_test_org = X_test_org[:50]
        Y_test = Y_test[:50]

        # Initialize datasets
        cls.data_train = dict(X=X_train_org, Y=Y_train)
        cls.data_test = dict(X=X_test_org, Y=Y_test)

        cls.platform = 'CPU'
        cls.epochs = 10

    def test_all_components(self):
        # Dynamically get all classes from tmcomposite.components.image
        components_classes = [getattr(image_components, attr) for attr in dir(image_components)
                              if isinstance(getattr(image_components, attr), type) and
                              issubclass(getattr(image_components, attr), TMComponent) and
                              getattr(image_components, attr) != TMComponent]

        # Create a composite model using all the loaded classes as components
        components = []
        for component_class in components_classes:
            component = component_class(
                TMClassifier,
                TMClassifierConfig(
                    number_of_clauses=50,
                    T=20,
                    s=10.0,
                    max_included_literals=32,
                    platform=self.platform,
                    weighted_clauses=True,
                    patch_dim=(10, 10)
                ),
                epochs=self.epochs
            )
            components.append(component)

        composite_model = TMComposite(components=components, use_multiprocessing=False)

        # Train and test
        composite_model.fit(
            data=self.data_train,
            callbacks=[
            ]
        )

        preds = composite_model.predict(data=self.data_test)
        y_true = self.data_test["Y"].flatten()

        for k, v in preds.items():
            accuracy = (v == y_true).mean()
            print(f"{k} Accuracy: %.1f" % (100 * accuracy))

if __name__ == '__main__':
    unittest.main()
