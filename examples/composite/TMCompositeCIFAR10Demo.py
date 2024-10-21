import argparse
import logging

from tmu.composite.callbacks.base import TMCompositeCallback
from tmu.composite.composite import TMComposite
from tmu.composite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmu.composite.components.color_thermometer_scoring import ColorThermometerComponent
from tmu.composite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmu.composite.config import TMClassifierConfig
from tmu.data.cifar10 import CIFAR10
from tmu.models.classification.vanilla_classifier import TMClassifier
import pathlib

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[]
    )

def main(args):
    experiment_results = metrics(args)

    platform = args.platform
    epochs = args.epochs
    checkpoint_path = pathlib.Path("checkpoints")
    checkpoint_path.mkdir(exist_ok=True)

    composite_path = checkpoint_path / "composite"
    composite_path.mkdir(exist_ok=True)

    component_path = checkpoint_path / "components"
    component_path.mkdir(exist_ok=True)

    data = CIFAR10().get()
    X_train_org = data["x_train"]
    Y_train = data["y_train"]
    X_test_org = data["x_test"]
    Y_test = data["y_test"]

    data_train = dict(
        X=X_train_org,
        Y=Y_train
    )

    data_test = dict(
        X=X_test_org,
        Y=Y_test
    )

    class TMCompositeCheckpointCallback(TMCompositeCallback):

        def on_epoch_component_begin(self, component, epoch, logs=None, **kwargs):
            pass

        def on_epoch_component_end(self, component, epoch, logs=None, **kwargs):
            component.save(component_path / f"{component}-{epoch}.pkl")

    class TMCompositeEvaluationCallback(TMCompositeCallback):

        def __init__(self, data):
            super().__init__()
            self.best_acc = 0.0
            self.data = data

    # def on_epoch_end(self, composite, epoch, logs=None):
    #     preds = composite.predict(data=self.data)
    #     print("Team Accuracy: %.1f" % (100 * (preds == self.data["Y"]).mean()))

    # Define the composite model
    composite_model = TMComposite(
        components=[
            AdaptiveThresholdingComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000,
                T=500,
                s=10.0,
                max_included_literals=32,
                platform=platform,
                weighted_clauses=True,
                patch_dim=(10, 10),
            ), epochs=epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000,
                T=1500,
                s=2.5,
                max_included_literals=32,
                platform=platform,
                weighted_clauses=True,
                patch_dim=(3, 3),
            ), resolution=8, epochs=epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000,
                T=1500,
                s=2.5,
                max_included_literals=32,
                platform=platform,
                weighted_clauses=True,
                patch_dim=(4, 4),
            ), resolution=8, epochs=epochs),

            HistogramOfGradientsComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000,
                T=50,
                s=10.0,
                max_included_literals=32,
                platform=platform,
                weighted_clauses=False
            ), epochs=epochs)
        ],
        use_multiprocessing=False
    )

    # Train the composite model
    composite_model.fit(
        data=data_train,
        callbacks=[
            TMCompositeCheckpointCallback(),
            TMCompositeEvaluationCallback(data=data_test)
        ]
    )

    preds = composite_model.predict(data=data_test)

    y_true = data_test["Y"].flatten()
    for k, v in preds.items():
        comp_acc = 100 * (v == y_true).mean()
        print(f"{k} Accuracy: %.1f" % (comp_acc,))

        if k not in experiment_results["accuracy"]:
            experiment_results["accuracy"][k] = []
        experiment_results["accuracy"][k].append(comp_acc)

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="CPU", type=str)
    parser.add_argument("--epochs", default=30, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
