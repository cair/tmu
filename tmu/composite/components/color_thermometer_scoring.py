import numpy as np
from typing import Dict, Any
from tmu.composite.components.base import TMComponent


class ColorThermometerComponent(TMComponent):
    def __init__(self, model_cls, model_config, resolution=8, **kwargs) -> None:
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        if resolution < 2 or resolution > 255:
            raise ValueError("Resolution must be between 2 and 255")
        self.resolution = resolution
        self._thresholds = None

    def _create_thresholds(self) -> None:
        self._thresholds = np.linspace(0, 255, self.resolution + 1)[1:-1]

    def preprocess(self, data: dict) -> Dict[str, Any]:
        super().preprocess(data=data)
        X_org = data.get("X")
        Y = data.get("Y")

        if X_org is None:
            raise ValueError("Input data 'X' is missing")

        if X_org.ndim != 4:
            raise ValueError(f"Expected 4D input, got {X_org.ndim}D")

        if X_org.shape[-1] != 3:
            raise ValueError(f"Expected 3 color channels, got {X_org.shape[-1]}")

        if self._thresholds is None:
            self._create_thresholds()

        # Use broadcasting for efficient computation
        X = (X_org[:, :, :, :, np.newaxis] >= self._thresholds).astype(np.uint8)

        # Reshape correctly
        batch_size, height, width, channels, _ = X.shape
        X = X.transpose(0, 1, 2, 4, 3).reshape(batch_size, height, width, channels * (self.resolution - 1))

        return {
            "X": X,
            "Y": Y
        }

    def get_output_shape(self, input_shape: tuple) -> tuple:
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")
        return (*input_shape[:-1], input_shape[-1] * (self.resolution - 1))