import typing
import logging
import numpy as np

from tmu.tools import CFFISerializable

_LOGGER = logging.getLogger(__name__)


class BaseClauseBank(CFFISerializable):
    clause_bank: np.ndarray

    def __init__(
            self,
            seed: int,
            X_shape: tuple,
            s: float,
            boost_true_positive_feedback: bool,
            reuse_random_feedback: bool,
            type_ia_ii_feedback_ratio: int,
            number_of_clauses: int,
            max_included_literals: int,
            patch_dim: typing.Union[tuple, None],
            **kwargs
    ):
        self._warn_unknown_arguments(**kwargs)
        assert isinstance(number_of_clauses, int)
        assert isinstance(patch_dim, tuple) or patch_dim is None

        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.number_of_clauses = int(number_of_clauses)

        self.patch_dim = patch_dim
        self.s = s
        self.boost_true_positive_feedback = int(boost_true_positive_feedback)
        self.reuse_random_feedback = int(reuse_random_feedback)
        self.type_ia_ii_feedback_ratio = type_ia_ii_feedback_ratio

        if len(X_shape) == 2:
            self.dim = (X_shape[1], 1, 1)
        elif len(X_shape) == 3:
            self.dim = (X_shape[1], X_shape[2], 1)
        elif len(X_shape) == 4:
            self.dim = (X_shape[1], X_shape[2], X_shape[3])
        else:
            raise RuntimeError(f"Invalid shape on X. Found shape: {X_shape}")

        if self.patch_dim is None:
            self.patch_dim = (self.dim[0] * self.dim[1] * self.dim[2], 1)

        self.number_of_features = int(
            self.patch_dim[0] * self.patch_dim[1] * self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (
                    self.dim[1] - self.patch_dim[1]))
        self.number_of_literals = self.number_of_features * 2

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.number_of_ta_chunks = int((self.number_of_literals - 1) / 32 + 1)

        self.max_included_literals = max_included_literals if max_included_literals else self.number_of_literals

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
