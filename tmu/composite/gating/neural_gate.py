import numpy as np
from tmu.composite.gating.base import BaseGate
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class NeuralGate(BaseGate):

    def __init__(self, composite, input_dim, **kwargs):
        super().__init__(composite, **kwargs)
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim):
        """Construct a feed-forward neural network."""
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.composite.components), activation='softmax'))

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        """Train the neural network."""
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, data: dict) -> np.ndarray:
        """Predict the gating mechanism."""
        X_test = data["X"]
        return self.model.predict(X_test)
