import json
import time
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from joblib import Parallel, delayed
import numpy as np

from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.composite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmu.composite.components.color_thermometer_scoring import ColorThermometerComponent
from tmu.composite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmu.composite.composite import TMComposite
from tmu.composite.config import TMClassifierConfig


class TMCompositeTuner:

    def __init__(
            self,
            data_train,
            data_test,
            platform="CPU",
            max_epochs=200,
            n_jobs: int = 1,
            callbacks=None,
            use_multiprocessing=True,
            study_name="TMComposite_study"
    ):
        self.data_train = data_train
        self.data_test = data_test
        self.last_accuracy = 0.0
        self.n_components = 1
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.platform = platform
        self.max_epochs = max_epochs
        if callbacks is None:
            callbacks = []

        self.callbacks = callbacks
        self.use_multiprocessing = use_multiprocessing

    def objective(self, trial: optuna.trial.Trial) -> float:
        components_list = []

        for i in range(self.n_components):
            component_type = trial.suggest_categorical(f'component_type_{i}',
                                                       ['AdaptiveThresholdingComponent',
                                                        'ColorThermometerComponent',
                                                        'HistogramOfGradientsComponent'])

            num_clauses = trial.suggest_int(f'num_clauses_{i}', 1000, 3000)
            T = trial.suggest_int(f'T_{i}', 100, 1500)
            s = trial.suggest_float(f's_{i}', 2.0, 15.0)
            max_included_literals = trial.suggest_int(f'max_literals_{i}', 16, 64)
            weighted_clauses = trial.suggest_categorical(f'weighted_clauses_{i}', [True, False])
            epochs = trial.suggest_int(f'epochs_{i}', 1, self.max_epochs)

            config = TMClassifierConfig(
                num_clauses=num_clauses,
                T=T,
                s=s,
                max_included_literals=max_included_literals,
                platform=self.platform,
                weighted_clauses=weighted_clauses
            )

            if component_type == 'AdaptiveThresholdingComponent':
                patch_dim = (trial.suggest_int(f'patch_dim_1_{i}', 1, 10), trial.suggest_int(f'patch_dim_2_{i}', 1, 10))
                config.patch_dim = patch_dim
                components_list.append(AdaptiveThresholdingComponent(TMClassifier, config, epochs=epochs))

            elif component_type == 'ColorThermometerComponent':
                patch_dim = (trial.suggest_int(f'patch_dim_1_{i}', 1, 10), trial.suggest_int(f'patch_dim_2_{i}', 1, 10))
                config.patch_dim = patch_dim
                resolution = trial.suggest_int(f'resolution_{i}', 1, 10)
                components_list.append(
                    ColorThermometerComponent(TMClassifier, config, resolution=resolution, epochs=epochs))

            elif component_type == 'HistogramOfGradientsComponent':
                components_list.append(HistogramOfGradientsComponent(TMClassifier, config, epochs=epochs))

        composite_model = TMComposite(components=components_list, use_multiprocessing=self.use_multiprocessing)

        # Training and evaluation
        composite_model.fit(
            data=self.data_train,
            callbacks=self.callbacks
        )

        preds = composite_model.predict(data=self.data_test)
        accuracy = (preds['composite'] == self.data_test['Y'].flatten()).mean()

        # Adjust number of components for next trial
        if accuracy > self.last_accuracy:
            self.n_components += 1
        else:
            self.n_components = max(1, self.n_components - 1)

        self.last_accuracy = accuracy
        return accuracy

    def save_best_params(self, study, trial, filename="best_params.json"):
        best_data = {
            'params': study.best_params,
            'value': trial.value
        }
        with open(filename, "w") as f:
            json.dump(best_data, f)

    def gradual_saving_callback(self, study, trial):
        # Use np.isclose to handle potential floating-point precision issues
        if np.isclose(trial.value, study.best_value, atol=1e-10):
            self.save_best_params(study, trial, filename=f"best_params_trial_{trial.number}.json")

    def retry_optimize(self, study, objective, n_trials, callbacks, max_retries=5, wait_time=2.0):
        for _ in range(max_retries):
            try:
                study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
                return
            except Exception as e:
                if "database is locked" in str(e).lower():
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Max retries reached for database access")

    def tune(self, n_trials: int = 100):
        storage = JournalStorage(JournalFileStorage("optuna-journal.log"))
        with Parallel(n_jobs=self.n_jobs) as parallel:
            if self.n_jobs == 1:
                study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), storage=storage,
                                            load_if_exists=True)
                self.retry_optimize(study, self.objective, n_trials, [self.gradual_saving_callback])
            else:
                study = optuna.create_study(study_name=self.study_name, direction='maximize', storage=storage,
                                            load_if_exists=True, pruner=optuna.pruners.MedianPruner())
                parallel(
                    delayed(self.retry_optimize)(study, self.objective, n_trials // self.n_jobs,
                                                 [self.gradual_saving_callback])
                    for i in range(self.n_jobs)
                )

        return study.best_params, study.best_value
