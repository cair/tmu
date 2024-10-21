import abc
import uuid

import numpy as np
from pathlib import Path
from typing import Union, Tuple
import logging

_LOGGER = logging.getLogger(__name__)


class TMComponent(abc.ABC):

    def __init__(self, model_cls, model_config, epochs=1, **kwargs) -> None:
        self.model_cls = model_cls
        self.model_config = model_config
        self.epochs = epochs
        self.uuid = uuid.uuid4()

        # Warn about unused kwargs
        if kwargs:
            print(f"Warning: unused keyword arguments: {kwargs}")

        self.model_instance = self.model_cls(
            **self.model_config.model_dump()
        )

    def model(self):
        raise NotImplementedError

    def preprocess(self, data: dict) -> dict:
        assert "X" in data, "X must be in data"
        assert "Y" in data, "Y must be in data"

        # Check if this is called from a class that inherits from TMComponent
        if not isinstance(self, TMComponent):
            raise TypeError(f"{type(self).__name__} does not inherit from TMComponent")

        return data

    def fit(self, data: dict) -> None:
        try:
            x_train, y_train = data["X"], data["Y"]

            # Check if type is uint32
            if x_train.dtype != np.uint32:
                x_train = x_train.astype(np.uint32)

            if y_train.dtype != np.uint32:
                y_train = y_train.astype(np.uint32)

            self.model_instance.fit(
                x_train,
                y_train,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise e

    def predict(self, data: dict) -> Tuple[np.array, np.array]:
        X_test = data["X"]
        return self.model_instance.predict(X_test, return_class_sums=True)

    def save(self, path: Union[Path, str], format="pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if format == "pkl":
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f)

    def __str__(self):
        params = '-'.join([f"{k}={v}" for k, v in self.model_config.model_dump().items()])
        return f"{type(self).__name__}-{self.model_cls.__name__}-{params})"
