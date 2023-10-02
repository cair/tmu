import numpy as np


class SparseClauseContainer:

    def __init__(
            self,
            random_seed
    ):
        super().__init__()

        self._rng_np = np.random.RandomState(seed=random_seed)
        self._clause_type = None
        self._clause_args = None
        self._classes = []
        self._d = dict()

    def items(self):
        return self._d.items()

    @property
    def n_classes(self):
        return len(self._classes)

    def classes(self):
        return self._classes

    def __len__(self):
        return len(self._classes)

    def sample(self, n=1, exclude=None):
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)

        results = []
        for _ in range(n):
            while True:
                idx = self._rng_np.randint(0, self.n_classes)
                sampled_class = self._classes[idx]
                if sampled_class not in exclude:
                    results.append(sampled_class)
                    break

        return results[0] if n == 1 else results

    def __iter__(self):
        return self._d.__iter__()

    def __getitem__(self, item):
        try:
            return self._d[item]
        except KeyError:

            if self._clause_type is None:
                raise RuntimeError("You must call set_clause_init before running fit()")

            self._d[item] = self._clause_type(**self._clause_args)

            return self._d[item]

    def insert(self, key, value):
        if key not in self._d:
            self._classes.append(key)

        self._d[key] = value

    def set_clause_init(self, clause_type, clause_args):
        self._clause_type = clause_type
        self._clause_args = clause_args

    def populate(self, keys):
        for key in keys:
            self.insert(key, value=self._clause_type(**self._clause_args))

    def clear(self):
        self._d.clear()
        self._classes.clear()
