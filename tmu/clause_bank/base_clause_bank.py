import typing
import logging
import numpy as np

from tmu.tools import CFFISerializable

_LOGGER = logging.getLogger(__name__)


class BaseClauseBank(CFFISerializable):
    clause_bank: np.ndarray

    def __init__(
            self,
            X,
            number_of_clauses: int,
            number_of_state_bits_ta: int,
            patch_dim: typing.Union[tuple, None],
            **kwargs
    ):
        self._warn_unknown_arguments(**kwargs)
        assert isinstance(number_of_clauses, int)
        assert isinstance(number_of_state_bits_ta, int)
        assert isinstance(patch_dim, tuple) or patch_dim is None
        self.X = X
        self.number_of_clauses = number_of_clauses
        self.number_of_state_bits_ta = number_of_state_bits_ta
        self.patch_dim = patch_dim

        if len(X.shape) == 2:
            self.dim = (X.shape[1], 1, 1)
        elif len(X.shape) == 3:
            self.dim = (X.shape[1], X.shape[2], 1)
        elif len(X.shape) == 4:
            self.dim = (X.shape[1], X.shape[2], X.shape[3])

        if self.patch_dim is None:
            self.patch_dim = (self.dim[0] * self.dim[1] * self.dim[2], 1)

        self.number_of_features = int(
            self.patch_dim[0] * self.patch_dim[1] * self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (
                    self.dim[1] - self.patch_dim[1]))
        self.number_of_literals = self.number_of_features * 2

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.number_of_ta_chunks = int((self.number_of_literals - 1) / 32 + 1)

    def _warn_unknown_arguments(self, **kwargs):
        for k, v in kwargs.items():
            _LOGGER.warning(f"Unknown positional argument for {self}: argument_name={k}, argument_value={v}")

    def set_ta_state(self, clause, ta, state):
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(
            clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + ta_chunk * self.number_of_state_bits_ta)
        for b in range(self.number_of_state_bits_ta):
            if state & (1 << b) > 0:
                self.clause_bank[pos + b] |= (1 << chunk_pos)
            else:
                self.clause_bank[pos + b] &= ~(1 << chunk_pos)

    def get_ta_state(self, clause, ta):
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(
            clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + ta_chunk * self.number_of_state_bits_ta)
        state = 0
        for b in range(self.number_of_state_bits_ta):
            if self.clause_bank[pos + b] & (1 << chunk_pos) > 0:
                state |= (1 << b)
        return state

    def get_ta_action(self, clause, ta):
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(
            clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + ta_chunk * self.number_of_state_bits_ta + self.number_of_state_bits_ta - 1)
        return (self.clause_bank[pos] & (1 << chunk_pos)) > 0
