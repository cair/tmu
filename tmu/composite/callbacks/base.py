from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Dict


class CallbackMethod(Enum):
    ON_TRAIN_COMPOSITE_BEGIN = auto()
    ON_TRAIN_COMPOSITE_END = auto()
    ON_EPOCH_COMPONENT_BEGIN = auto()
    ON_EPOCH_COMPONENT_END = auto()
    UPDATE_PROGRESS = auto()

@dataclass
class CallbackMessage:
    method: CallbackMethod
    kwargs: Dict[str, Any]

class TMCompositeCallback:
    def __init__(self):
        pass

    def on_epoch_component_begin(self, component, epoch, logs=None, **kwargs):
        pass

    def on_epoch_component_end(self, component, epoch, logs=None, **kwargs):
        pass

    def on_train_composite_end(self, composite, logs=None, **kwargs):
        pass

    def on_train_composite_begin(self, composite, logs=None, **kwargs):
        pass
