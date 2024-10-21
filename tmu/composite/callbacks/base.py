from dataclasses import dataclass
from enum import auto, Enum
from multiprocessing import Queue
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

class TMCompositeCallbackProxy:
    def __init__(self, queue: Queue):
        self.queue = queue

    def on_epoch_component_begin(self, component, epoch, logs=None):
        self.queue.put(CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_BEGIN, {'component': component, 'epoch': epoch, 'logs': logs}))

    def on_epoch_component_end(self, component, epoch, logs=None):
        self.queue.put(CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_END, {'component': component, 'epoch': epoch, 'logs': logs}))

    def on_train_composite_end(self, composite, logs=None):
        self.queue.put(CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_END, {'composite': composite, 'logs': logs}))

    def on_train_composite_begin(self, composite, logs=None):
        self.queue.put(CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_BEGIN, {'composite': composite, 'logs': logs}))