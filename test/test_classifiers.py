import unittest
from models.base import TMBasis


class ClassifierTests(unittest.TestCase):
    model: TMBasis

    def setUp(self) -> None:
        from data import TMUDatasetSource
        self.num_epochs = 10
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


class VanillaClassifierTests(ClassifierTests):

    def setUp(self) -> None:
        from models.classification.vanilla_classifier import TMClassifier
        self.model = TMClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        super().setUp()


class TMMultiChannelClassifierTests(ClassifierTests):

    def setUp(self) -> None:
        from models.classification.multichannel_classifier import TMMultiChannelClassifier
        self.model = TMMultiChannelClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        super().setUp()


class TMCoalescedClassifierv2Tests(ClassifierTests):

    def setUp(self) -> None:
        from tmu.models.classification.coalesced_experimental_classifier import TMCoalescedClassifier
        self.model = TMCoalescedClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        super().setUp()


class TMCoalescedClassifierv1Tests(ClassifierTests):

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
        super().setUp()


class OneVSOneClassifierTests(ClassifierTests):

    def setUp(self) -> None:
        from models.classification.one_vs_one_classifier import TMOneVsOneClassifier

        self.model = TMOneVsOneClassifier(
            number_of_clauses=4,
            T=10,
            s=10.0,
            max_included_literals=32,
            platform="CPU",
            weighted_clauses=False
        )
        super().setUp()
