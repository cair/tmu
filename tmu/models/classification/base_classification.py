import numpy as np
from tmu.models.base import TMBasis
import logging

_LOGGER = logging.getLogger(__name__)


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

    def _build_cpu_bank(self, X: np.ndarray):
        from tmu.clause_bank.clause_bank import ClauseBank
        clause_bank_type = ClauseBank
        clause_bank_args = dict(
            X=X,
            d=self.d,
            s=self.s,
            boost_true_positive_feedback=self.boost_true_positive_feedback,
            reuse_random_feedback=self.reuse_random_feedback,
            max_included_literals=self.max_included_literals,
            number_of_clauses=self.number_of_clauses,
            number_of_state_bits_ta=self.number_of_state_bits_ta,
            number_of_state_bits_ind=self.number_of_state_bits_ind,
            patch_dim=self.patch_dim,
            batch_size=self.batch_size,
            incremental=self.incremental,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio
        )
        return clause_bank_type, clause_bank_args

    def _build_gpu_bank(self, X: np.ndarray):
        from tmu.clause_bank.clause_bank_cuda import ClauseBankCUDA, cuda_installed

        if not cuda_installed:
            _LOGGER.warning("CUDA not installed, using CPU clause bank")
            return self._build_cpu_bank(X=X)

        clause_bank_type = ClauseBankCUDA
        clause_bank_args = dict(
            X=X,
            s=self.s,
            boost_true_positive_feedback=self.boost_true_positive_feedback,
            reuse_random_feedback=self.reuse_random_feedback,
            number_of_clauses=self.number_of_clauses,
            number_of_state_bits_ta=self.number_of_state_bits_ta,
            patch_dim=self.patch_dim,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio,
            max_included_literals=self.max_included_literals,
        )
        return clause_bank_type, clause_bank_args

    def _build_cpu_sparse_bank(self, X: np.ndarray):
        from tmu.clause_bank.clause_bank_sparse import ClauseBankSparse
        clause_bank_type = ClauseBankSparse
        clause_bank_args = dict(
            X=X,
            d=self.d,
            s=self.s,
            boost_true_positive_feedback=self.boost_true_positive_feedback,
            reuse_random_feedback=self.reuse_random_feedback,
            max_included_literals=self.max_included_literals,
            number_of_clauses=self.number_of_clauses,
            number_of_states=2 ** self.number_of_state_bits_ta,
            patch_dim=self.patch_dim,
            absorbing=self.absorbing,
            absorbing_exclude=self.absorbing_exclude,
            absorbing_include=self.absorbing_include,
            literal_sampling=self.literal_sampling,
            feedback_rate_excluded_literals=self.feedback_rate_excluded_literals,
            literal_insertion_state=self.literal_insertion_state,
            type_ia_ii_feedback_ratio=self.type_ia_ii_feedback_ratio
        )
        return clause_bank_type, clause_bank_args

    def build_clause_bank(self, X: np.ndarray):
        _LOGGER.debug("Initializing clause bank....")

        if self.platform == "CPU":
            clause_bank_type, clause_bank_args = self._build_cpu_bank(X=X)

        elif self.platform in ["GPU", "CUDA"]:
            clause_bank_type, clause_bank_args = self._build_gpu_bank(X=X)
        elif self.platform == "CPU_sparse":
            clause_bank_type, clause_bank_args = self._build_cpu_sparse_bank(X=X)
        else:
            raise NotImplementedError(f"Could not find platform of type {self.platform}.")

        return clause_bank_type, clause_bank_args
