import numpy as np
from tmu.composite.gating.base import BaseGate


class LinearGate(BaseGate):

    def __init__(self, composite, **kwargs):
        super().__init__(composite, **kwargs)

    def predict(self, data: dict) -> np.ndarray:

        # In the linear gate, we simply select all the components
        n_components = len(self.composite.components)
        n_data_points = data["Y"].shape[0]

        return np.ones((n_data_points, n_components, 1), dtype=np.float32)
