
class BaseGate:

    def __init__(self, composite, **kwargs):
        self.composite = composite

    def preprocess(self, data: dict):
        pass

    def fit(self, data: dict) -> None:
        pass

    def predict(self, data: dict) -> dict:
        pass
