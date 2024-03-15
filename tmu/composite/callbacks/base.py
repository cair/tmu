from multiprocessing import Queue


class TMCompositeCallback:

    def __init__(self):
        pass

    def on_epoch_component_begin(self, component, epoch, logs=None):
        pass

    def on_epoch_component_end(self, component, epoch, logs=None):
        pass

    def on_train_composite_end(self, composite, logs=None):
        pass

    def on_train_composite_begin(self, composite, logs=None):
        pass


class TMCompositeCallbackProxy:

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_epoch_component_begin(self, component, epoch, logs=None):
        self.queue.put(('on_epoch_component_begin', component, epoch, logs))

    def on_epoch_component_end(self, component, epoch, logs=None):
        self.queue.put(('on_epoch_component_end', component, epoch, logs))

    def on_train_composite_end(self, composite, logs=None):
        self.queue.put(('on_train_composite_end', composite, logs))

    def on_train_composite_begin(self, composite, logs=None):
        self.queue.put(('on_train_composite_begin', composite, logs))
