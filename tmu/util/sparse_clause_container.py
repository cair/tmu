import random


class SparseClauseContainer:

    def __init__(self, random_seed=None):
        super().__init__()
        self._rng = random.Random(x=random_seed)
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

    def sample(self, n=1):
        if n == 1:
            return self._rng.choice(self._classes)
        return self._rng.choices(self._classes, k=n)

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
