class ClauseDictionary(dict):

    def __init__(self):
        super().__init__()

        self._clause_type = None
        self._clause_args = None

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            self.__setitem__(item, self._clause_type(**self._clause_args))

            return super().__getitem__(item)

    def append(self, item):
        """
        To comply with list interface we assume that append will use the nth index in the dictionary
        :param item:
        :return:
        """
        self.__setitem__(len(self), item)

    def set_clause_init(self, clause_type, clause_args):
        self._clause_type = clause_type
        self._clause_args = clause_args
