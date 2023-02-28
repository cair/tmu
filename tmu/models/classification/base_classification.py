
import numpy as np
from tmu.models.base import TMBasis


class TMBaseClassifier(TMBasis):
    number_of_classes: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_before(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        pass

    def init_num_classes(self, X: np.ndarray, Y: np.ndarray) -> int:
        pass

    def init(self, X: np.ndarray, Y: np.ndarray):
        if self.initialized:
            return
        self.init_before(X, Y)
        self.number_of_classes = self.init_num_classes(X, Y)
        self.init_clause_bank(X, Y)
        self.init_weight_bank(X, Y)
        self.init_after(X, Y)
        self.initialized = True

