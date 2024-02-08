from typing import Optional
import xxhash
import numpy as np
from scipy.sparse import issparse


class DataEncoderCache:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.array_hash: Optional[str] = None
        self.encoded_data: Optional[np.ndarray] = None

    def compute_hash_csr_matrix(self, csr_mat):
        # Convert the components of the csr_matrix to bytes
        data_bytes = csr_mat.data.tobytes()
        indices_bytes = csr_mat.indices.tobytes()
        indptr_bytes = csr_mat.indptr.tobytes()

        # Concatenate the bytes representations
        total_bytes = data_bytes + indices_bytes + indptr_bytes

        # Compute the hash on the concatenated bytes
        hash_value = xxhash.xxh3_64_hexdigest(total_bytes)

        return hash_value

    def compute_hash(self, arr):
        """Compute a hash for a numpy array or csr_matrix."""
        if issparse(arr):
            # It's a sparse matrix, handle specially
            return self.compute_hash_csr_matrix(arr)
        else:
            # It's a dense array, proceed as before
            return xxhash.xxh3_64_hexdigest(arr.tobytes())

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
