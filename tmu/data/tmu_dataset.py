import typing
import abc
import numpy as np
from typing import Dict

class TMUDataset:

    def __init__(self):
        self._custom_transforms: typing.Optional[typing.List[typing.Callable]] = None
        pass

    def add_transform(self, transform: typing.Callable):
        if self._custom_transforms is None:
            self._custom_transforms = [transform]
        else:
            self._custom_transforms.append(transform)

    @abc.abstractmethod
    def _transform(self, name, dataset):
        raise NotImplementedError("You should override def _transform()")

    @abc.abstractmethod
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("You should override def _retrieve_dataset()")

    def _execute_transforms(self, k, v):
        if self._custom_transforms:
            for transform in self._custom_transforms:
                v = transform(k, v)
        else:
            v = self._transform(k, v)

        return v

    def get(self):
        return {k: self._execute_transforms(k, v) for k, v in self._retrieve_dataset().items()}

    def get_list(self):
        return list(self.get().values())