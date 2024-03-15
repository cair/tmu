import numpy as np
from tmu.composite.components.base import TMComponent


class ColorThermometerComponent(TMComponent):

    def __init__(self, model_cls, model_config, resolution=8, **kwargs) -> None:
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        self.resolution = resolution

    def preprocess(self, data: dict):
        super().preprocess(data=data)

        X_org = data["X"]
        Y = data["Y"]

        X = np.empty((X_org.shape[0], X_org.shape[1], X_org.shape[2], X_org.shape[3], self.resolution), dtype=np.uint8)
        for z in range(self.resolution):
            X[:, :, :, :, z] = X_org[:, :, :, :] >= (z + 1) * 255 / (self.resolution + 1)

        X = X.reshape((X_org.shape[0], X_org.shape[1], X_org.shape[2], 3 * self.resolution))

        return dict(
            X=X,
            Y=Y,
        )
