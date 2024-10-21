import concurrent.futures
import multiprocessing
import traceback
import uuid
from functools import partial
from multiprocessing import Manager, Queue, cpu_count, Pool
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import threading
from tmu.composite.callbacks.base import TMCompositeCallbackProxy, TMCompositeCallback
from tmu.composite.components.base import TMComponent
from tmu.composite.gating.base import BaseGate
from tmu.composite.gating.linear_gate import LinearGate
from tmu.composite.callbacks.base import CallbackMessage, CallbackMethod


@dataclass
class ComponentTask:
    component: TMComponent
    data: Any
    epochs: int
    progress: int = 0
    result: Any = None
    @property
    def component_id(self) -> uuid.UUID:
        return self.component.uuid

@dataclass
class FitResult:
    component: TMComponent
    success: bool
    error: Optional[Exception] = None

class TMCompositeBase:

    def __init__(self, composite) -> None:
        self.composite = composite

    def _component_predict(self, component, data):
        data_preprocessed = component.preprocess(data)
        _, scores = component.predict(data_preprocessed)

        votes = dict()
        votes["composite"] = np.zeros_like(scores, dtype=np.float32)
        votes[str(component)] = np.zeros_like(scores, dtype=np.float32)

        for i in range(scores.shape[0]):
            denominator = np.max(scores[i]) - np.min(scores[i])
            score = 1.0 * scores[i] / denominator if denominator != 0 else 0
            votes["composite"][i] += score
            votes[str(component)][i] += score

        return votes


class TMCompositeMP:
    def __init__(self, composite: 'TMComposite', **kwargs) -> None:
        self.composite = composite
        self.max_workers = min(cpu_count(), len(composite.components))
        self.remove_data_after_preprocess = kwargs.get('remove_data_after_preprocess', False)
        multiprocessing.set_start_method('spawn', force=True)

    def _process_callbacks(self, callbacks: List[TMCompositeCallback], message: CallbackMessage) -> None:
        method_name = message.method.name.lower()
        for callback in callbacks:
            try:
                getattr(callback, method_name)(**message.kwargs, composite=self.composite)
            except Exception as e:
                print(f"Error in callback {callback.__class__.__name__}.{method_name}: {e}")
                traceback.print_exc()

    def _fit_component(self, task, callback_queue) -> FitResult:
        try:
            data_preprocessed = task.component.preprocess(task.data)
            callbacks = []

            # remove task.data
            if self.remove_data_after_preprocess:
                task.data = None

            for epoch in range(task.epochs):

                if callback_queue:
                    callbacks.append(CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_BEGIN, {'component': task.component, 'epoch': epoch}))

                task.component.fit(data=data_preprocessed)
                task.progress += 1

                if callback_queue:
                    callbacks.append(CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_END, {'component': task.component, 'epoch': epoch}))

                if callback_queue:
                    callback_queue.put(callbacks)  # Send all callbacks at once
                    callbacks.clear()

            return FitResult(component=task.component, success=True)
        except Exception as e:
            print(f"Error in _fit_component for {task.component.__class__.__name__}: {e}")
            traceback.print_exc()
            return FitResult(component=task.component, success=False, error=e)

    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        with Manager() as manager:
            callback_queue: Optional[Queue] = manager.Queue() if callbacks else None
            error_queue: Queue = manager.Queue()

            self._process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_BEGIN, {}))

            tasks = [ComponentTask(component=component, data=data, epochs=component.epochs)
                     for component in self.composite.components]

            callback_thread = None
            if callbacks:
                callback_thread = self._start_callback_handler(callbacks, callback_queue, error_queue)


            results = self._execute_tasks(tasks, callback_queue)

            self._process_results(results, error_queue)

            self._cleanup(callback_queue, callback_thread)

            self._check_errors(error_queue)

            self._process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_END, {}))

    def _start_callback_handler(self, callbacks: List[TMCompositeCallback], callback_queue: Queue, error_queue: Queue):
        def callback_handler() -> None:
            while True:
                try:
                    message = callback_queue.get()
                    if message == 'DONE':
                        break
                    if isinstance(message, list):  # Handle batch of callbacks
                        for callback_message in message:
                            if isinstance(callback_message, CallbackMessage):
                                self._process_callbacks(callbacks, callback_message)
                except Exception as e:
                    print(f"Error in callback handler: {e}")
                    traceback.print_exc()
                    error_queue.put(('callback_handler', e))

        callback_thread = threading.Thread(target=callback_handler)
        callback_thread.start()
        return callback_thread





    def _execute_tasks(self, tasks: List[ComponentTask], callback_queue: Optional[Queue]) -> List[FitResult]:
        results = []

        # Create a partial function with the callback_queue
        fit_component_partial = partial(self._fit_component, callback_queue=callback_queue)

        with Pool(processes=self.max_workers) as pool:
            # Map the tasks to the pool
            async_results = [
                pool.apply_async(fit_component_partial, (task,)) for task in tasks
            ]

            # Collect results as they complete
            for async_result, task in zip(async_results, tasks):
                try:
                    result = async_result.get()  # This will wait for the task to complete
                    results.append(result)
                except Exception as e:
                    print(f"Exception when processing results for {task.component.__class__.__name__}: {e}")
                    traceback.print_exc()
                    results.append(
                        FitResult(component=task.component, success=False, error=e)
                    )

        return results

    def _process_results(self, results: List[FitResult], error_queue: Queue) -> None:
        for result in results:
            if result.success:
                matching_component = next(
                    (c for c in self.composite.components if c.uuid == result.component.uuid),
                    None
                )

                if matching_component is not None:
                    idx = self.composite.components.index(matching_component)
                    self.composite.components[idx] = result.component
                else:
                    error_message = f"Could not find a matching component for {result.component}"
                    print(error_message)
                    error_queue.put(('process_results', ValueError(error_message)))
            else:
                error_queue.put(('fit_component', result.error))

    def _cleanup(self, callback_queue: Optional[Queue], callback_thread: Optional[threading.Thread]) -> None:
        if callback_queue:
            callback_queue.put('DONE')
            if callback_thread:
                callback_thread.join()


    def _check_errors(self, error_queue: Queue) -> None:
        if not error_queue.empty():
            print("Errors occurred during fitting:")
            while not error_queue.empty():
                error_source, error = error_queue.get()
                print(f"Error in {error_source}: {error}")

    def predict(self, data: Dict[str, Any], votes: Dict[str, np.ndarray], gating_mask: np.ndarray) -> None:
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_component = {
                executor.submit(self._predict_component, component, data, gating_mask, i): component
                for i, component in enumerate(self.composite.components)
            }

            for future in concurrent.futures.as_completed(future_to_component):
                component = future_to_component[future]
                try:
                    result = future.result()
                    for key, score in result.items():
                        if key not in votes:
                            votes[key] = score
                        else:
                            votes[key] += score
                except Exception as e:
                    print(f"Exception when processing results for {component.__class__.__name__}: {e}")
                    traceback.print_exc()

    def _predict_component(
            self,
            component: TMComponent,
            data: Dict[str, Any],
            gating_mask: np.ndarray,
            component_idx: int
    ) -> Dict[str, np.ndarray]:
        try:
            # Preprocess data and get scores
            _, scores = component.predict(component.preprocess(data))
            scores = scores.reshape(scores.shape[0], -1)  # Ensure 2D

            # Normalize scores
            denominator = np.maximum(np.ptp(scores, axis=1), 1e-8)  # Avoid division by zero
            normalized_scores = scores / denominator[:, np.newaxis]

            # Apply gating mask
            mask = gating_mask[:, component_idx].reshape(-1, 1) if gating_mask.ndim > 1 else gating_mask.reshape(-1, 1)
            masked_scores = normalized_scores * mask

            # Create and return votes
            return {
                "composite": masked_scores,
                str(component): masked_scores
            }
        except Exception as e:
            print(f"Error in predict_component for {component}: {str(e)}")
            print(f"Shapes - scores: {scores.shape}, gating_mask: {gating_mask.shape}, mask: {mask.shape}")
            traceback.print_exc()
            return {}


class TMCompositeSingleCPU:
    def __init__(self, composite, **kwargs) -> None:
        self.composite = composite

    def _component_predict(self, component, data):
        data_preprocessed = component.preprocess(data)
        _, scores = component.predict(data_preprocessed)

        votes = dict()
        votes["composite"] = np.zeros_like(scores, dtype=np.float32)
        votes[str(component)] = np.zeros_like(scores, dtype=np.float32)

        for i in range(scores.shape[0]):
            denominator = np.max(scores[i]) - np.min(scores[i])
            score = 1.0 * scores[i] / denominator if denominator != 0 else 0
            votes["composite"][i] += score
            votes[str(component)][i] += score

        return votes

    def _process_callbacks(self, callbacks: List[TMCompositeCallback], message: CallbackMessage) -> None:
        method_name = message.method.name.lower()
        for callback in callbacks:
            try:
                getattr(callback, method_name)(**message.kwargs)
            except Exception as e:
                print(f"Error in callback {callback.__class__.__name__}.{method_name}: {e}")
                import traceback
                traceback.print_exc()

    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        if callbacks is None:
            callbacks = []

        data_preprocessed = [component.preprocess(data) for component in self.composite.components]
        epochs_left = [component.epochs for component in self.composite.components]

        self._process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_BEGIN, {'composite': self.composite}))

        epoch = 0
        while any(epochs_left):
            for idx, component in enumerate(self.composite.components):
                if epochs_left[idx] > 0:
                    self._process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_BEGIN, {'component': component, 'epoch': epoch}))

                    component.fit(data=data_preprocessed[idx])

                    self._process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_END, {'component': component, 'epoch': epoch}))

                    epochs_left[idx] -= 1

            epoch += 1

        self._process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_END, {'composite': self.composite}))

    def predict(self, data: Dict[str, Any], votes: Dict[str, np.ndarray], gating_mask: np.ndarray) -> None:
        for i, component in enumerate(self.composite.components):
            component_votes = self._component_predict(component, data)
            for key, score in component_votes.items():
                # Apply gating mask
                masked_score = score * gating_mask[:, i]

                if key not in votes:
                    votes[key] = masked_score
                else:
                    votes[key] += masked_score

class TMComposite:
    def __init__(
            self,
            components: Optional[List[TMComponent]] = None,
            gate_function: Optional[type[BaseGate]] = None,
            gate_function_params: Optional[Dict[str, Any]] = None,
            use_multiprocessing: bool = False,
            **kwargs
    ) -> None:
        self.components: List[TMComponent] = components or []
        self.use_multiprocessing = use_multiprocessing

        if gate_function_params is None:
            gate_function_params = {}

        self.gate_function_instance: BaseGate = gate_function(self, **gate_function_params) if gate_function else LinearGate(self, **gate_function_params)

        self.logic: Union[TMCompositeSingleCPU, TMCompositeMP] = TMCompositeMP(composite=self, **kwargs) if use_multiprocessing else TMCompositeSingleCPU(composite=self, **kwargs)

    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        self.logic.fit(data, callbacks)

    def predict(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        votes: Dict[str, np.ndarray] = {}

        # Gating Mechanism
        gating_mask: np.ndarray = self.gate_function_instance.predict(data)
        assert gating_mask.shape[1] == len(self.components)
        assert gating_mask.shape[0] == data["Y"].shape[0]

        self.logic.predict(data, votes, gating_mask)

        return {k: v.argmax(axis=1) for k, v in votes.items()}

    def save_model(self, path: Union[Path, str], format: str = "pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if format == "pkl":
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError(f"Format {format} not supported")

    def load_model(self, path: Union[Path, str], format: str = "pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if format == "pkl":
            import pickle
            with open(path, "rb") as f:
                loaded_model = pickle.load(f)
                self.__dict__.update(loaded_model.__dict__)
        else:
            raise NotImplementedError(f"Format {format} not supported")

    def load_model_from_components(self, path: Union[Path, str], format: str = "pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")

        # Get all files with the desired format
        files = [f for f in path.iterdir() if f.is_file() and f.suffix == f".{format}"]

        # Group files by component details
        from collections import defaultdict
        component_groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
        for file in files:
            parts = file.stem.split('-')
            epoch = int(parts[-1])
            component_name = '-'.join(parts[:-1])
            component_groups[component_name].append((epoch, file))

        # Sort files in each group based on the epoch and get the highest epoch file
        latest_files = [sorted(group, key=lambda x: x[0], reverse=True)[0][1] for group in component_groups.values()]

        # Load all components and add to components list
        self.components = []
        if format == "pkl":
            import pickle
            for component_path in latest_files:
                with open(component_path, "rb") as f:
                    loaded_component = pickle.load(f)
                    self.components.append(loaded_component)
        else:
            raise NotImplementedError(f"Format {format} not supported")
