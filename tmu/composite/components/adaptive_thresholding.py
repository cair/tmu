import numpy as np
import cv2
from tmu.composite.components.base import TMComponent


class AdaptiveThresholdingComponent(TMComponent):

    def __init__(self, model_cls, model_config, **kwargs) -> None:
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)

    def preprocess(self, data: dict):
        super().preprocess(data=data)

        X = np.empty((data["X"].shape[0], data["X"].shape[1], data["X"].shape[2], data["X"].shape[3]), dtype=np.uint8)
        Y = data["Y"]
        for i in range(X.shape[0]):
            for j in range(X.shape[3]):
                X[i, :, :, j] = cv2.adaptiveThreshold(data["X"][i, :, :, j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)

        return dict(
            X=X,
            Y=Y,
        )
