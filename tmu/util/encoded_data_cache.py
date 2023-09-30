import hashlib
from typing import Optional

import numpy as np

class DataEncoderCache:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.array_hash: Optional[str] = None
        self.encoded_data: Optional[np.ndarray] = None

    def compute_hash(self, arr: np.ndarray) -> str:
        """Compute a hash for a numpy array."""

        sampled_indices = self.rng.choice(arr.size, min(15, arr.size), replace=False)
        sampled_values = arr.flat[sampled_indices]
        return hashlib.sha256(sampled_values.tobytes()).hexdigest()

    def get_encoded_data(self, data: np.ndarray, encoder_func) -> np.ndarray:
        """Get encoded data for an array, using cache if available."""
        current_hash = self.compute_hash(data)
        if current_hash != self.array_hash:
            self.encoded_data = encoder_func(data)
            self.array_hash = current_hash

        return self.encoded_data

    def __getstate__(self):
        # This method controls what gets pickled.
        # Return a dictionary of the object's state without the encoded_data attribute.
        state = self.__dict__.copy()
        del state['encoded_data']
        return state

    def __setstate__(self, state):
        # This method controls how the object is unpickled.
        # Set the object's dictionary to the pickled state and initialize encoded_data to None.
        self.__dict__.update(state)
        self.encoded_data = None
