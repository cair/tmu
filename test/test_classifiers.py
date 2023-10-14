import unittest
from tmu.models.base import TMBaseModel


class ClassifierTests(object):
    model: TMBaseModel

    def setUp(self) -> None:
        from tmu.data.tmu_datasource import TMUDatasetSource
        self.num_epochs = 20
        self.data = TMUDatasetSource().get_dataset(
            "XOR_biased",
            cache=True,
            cache_max_age=1,
            features=["X", "Y"],
            labels=["xor"],
            shuffle=True,
            train_ratio=1000,
            test_ratio=1000,
            return_type=dict
        )

    def test_fit_function_defaults(self):
        for epoch in range(self.num_epochs):
            self.model.fit(
                self.data["x_train"],
                self.data["y_train"]
            )

    def test_predict_function(self):
        self.test_fit_function_defaults()
        y_preds = self.model.predict(
            self.data["x_test"]
        )
        accuracy = (y_preds == self.data["y_test"]).mean()

        self.assertGreater(accuracy, 0.7)


class VanillaClassifierTests(unittest.TestCase, ClassifierTests):

    def setUp(self) -> None:
        from tmu.models.classification.vanilla_classifier import TMClassifier
        self.model = TMClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        ClassifierTests.setUp(self)

@unittest.skip("Disabled TMMultiChannelClassifierTests for now.")
class TMMultiChannelClassifierTests(unittest.TestCase, ClassifierTests):

    def setUp(self) -> None:
        from tmu.models.classification.multichannel_classifier import TMMultiChannelClassifier
        import numpy as np
        self.model = TMMultiChannelClassifier(
            number_of_clauses=4,
            global_T=10.0,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        ClassifierTests.setUp(self)
        self.data["x_train"] = np.array([self.data["x_train"], self.data["x_train"]])


class TMCoalescedClassifierTests(unittest.TestCase, ClassifierTests):

    def setUp(self) -> None:
        from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
        self.model = TMCoalescedClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        ClassifierTests.setUp(self)

@unittest.skip("Disabled OneVSOneClassifier for now.")
class OneVSOneClassifierTests(unittest.TestCase, ClassifierTests):

    def setUp(self) -> None:
        from tmu.models.classification.one_vs_one_classifier import TMOneVsOneClassifier
        import numpy as np

        self.model = TMOneVsOneClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        ClassifierTests.setUp(self)
