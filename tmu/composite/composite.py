import concurrent.futures
import multiprocessing
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Manager, cpu_count, Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import numpy as np
from tmu.composite.callbacks.base import TMCompositeCallback
from tmu.composite.components.base import TMComponent
from tmu.composite.gating.base import BaseGate
from tmu.composite.gating.linear_gate import LinearGate
from tmu.composite.callbacks.base import CallbackMessage, CallbackMethod
from enum import Enum, auto
from datetime import datetime

class ComponentStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class ComponentState:
    """Tracks the complete state of a component during training."""
    status: ComponentStatus = ComponentStatus.PENDING
    current_epoch: int = 0
    total_epochs: int = 0
    last_update: float = 0.0  # timestamp
    error: Optional[Exception] = None
    metrics: Dict[str, float] = field(default_factory=dict)

class SharedDataManager:
    """
    A shared data manager that encapsulates shared objects for use across processes.

    This implementation creates a top-level shared dictionary with default keys:
      - progress: to track per-task progress (e.g., epoch updates)
      - logs: a list to aggregate log messages
      - metrics: to store any metric values

    The API can be extended seamlessly to add more shared data.
    """
    def __init__(self):
        self._manager = Manager()
        self.data = self._manager.dict()
        self.initialize_defaults()

    def initialize_defaults(self):
        self.data["progress"] = self._manager.dict()
        self.data["logs"] = self._manager.list()
        self.data["metrics"] = self._manager.dict()
        self.data["states"] = self._manager.dict()  # Track complete component states

    def set_progress(self, key, value):
        self.data["progress"][key] = value

    def get_progress(self, key):
        return self.data["progress"].get(key, 0)

    def add_log(self, message):
        self.data["logs"].append(message)

    def get_logs(self):
        return list(self.data["logs"])

    def set_metric(self, key, value):
        self.data["metrics"][key] = value

    def get_metric(self, key):
        return self.data["metrics"].get(key)

    def update_component_state(self, component_id: uuid.UUID, **kwargs) -> None:
        """Update specific fields of a component's state."""
        current = self.data["states"].get(component_id, ComponentState())
        for key, value in kwargs.items():
            setattr(current, key, value)
        current.last_update = datetime.now().timestamp()
        self.data["states"][component_id] = current

    def get_component_state(self, component_id: uuid.UUID) -> ComponentState:
        """Get the current state of a component."""
        return self.data["states"].get(component_id, ComponentState())

    def shutdown(self):
        self._manager.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

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

def process_callbacks(
    callbacks: List[TMCompositeCallback],
    message: CallbackMessage,
    composite: Optional[Any] = None
) -> None:
    """Helper to process a callback message on each callback."""
    for callback in callbacks:
        try:
            func = getattr(callback, message.method.name.lower())
            # If a composite is provided, pass it as a keyword argument.
            if composite is not None:
                func(**message.kwargs, composite=composite)
            else:
                func(**message.kwargs)
        except Exception as e:
            print(f"Error in callback {callback.__class__.__name__}.{message.method.name.lower()}: {e}")
            traceback.print_exc()

def compute_scores(component: TMComponent, data: Dict[str, Any]) -> np.ndarray:
    """
    Helper to preprocess data, obtain raw scores from a component, and return normalized scores.
    The normalization is done row-wise.
    """
    preprocessed = component.preprocess(data)
    _, scores = component.predict(preprocessed)
    # Ensure scores are 2D
    scores = scores.reshape(scores.shape[0], -1)
    denominator = np.maximum(np.ptp(scores, axis=1), 1e-8)  # Avoid division by zero
    normalized_scores = scores / denominator[:, np.newaxis]
    return normalized_scores

class TMCompositeRunner(ABC):
    """Abstract base class for running composite training and prediction."""

    def __init__(self, composite: "TMComposite") -> None:
        self.composite = composite

    @abstractmethod
    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        pass

    @abstractmethod
    def predict(self, data: Dict[str, Any], votes: Dict[str, np.ndarray], gating_mask: np.ndarray) -> None:
        pass

def _fit_component(task: ComponentTask, progress_dict, states_dict) -> FitResult:
    try:
        # Update state directly in the shared dict
        state = ComponentState(
            status=ComponentStatus.RUNNING,
            total_epochs=task.epochs
        )
        states_dict[task.component.uuid] = state

        pe_data = task.component.preprocess(task.data)
        for epoch in range(task.epochs):
            task.component.fit(data=pe_data)
            # Update progress
            progress_dict[task.component.uuid] = epoch + 1
            # Update state
            state.current_epoch = epoch + 1
            states_dict[task.component.uuid] = state

        state.status = ComponentStatus.COMPLETED
        states_dict[task.component.uuid] = state
        return FitResult(component=task.component, success=True)
    except Exception as e:
        state.status = ComponentStatus.FAILED
        state.error = e
        states_dict[task.component.uuid] = state
        return FitResult(component=task.component, success=False, error=e)

class TMCompositeMP(TMCompositeRunner):
    """Multiprocessing implementation of the composite runner."""

    def __init__(self, composite: "TMComposite", remove_data_after_preprocess: bool = False) -> None:
        super().__init__(composite)
        self.max_workers = min(cpu_count(), len(composite.components))
        self.remove_data_after_preprocess = remove_data_after_preprocess
        multiprocessing.set_start_method("spawn", force=True)
        self.use_shared_progress = True

    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        tasks: List[ComponentTask] = [
            ComponentTask(component=c, data=data, epochs=c.epochs) for c in self.composite.components
        ]

        if callbacks:
            process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_BEGIN, {}), composite=self.composite)

        with SharedDataManager() as shared:
            progress_dict = shared.data["progress"]
            states_dict = shared.data["states"]
            for task in tasks:
                # Initialize state
                states_dict[task.component.uuid] = ComponentState(
                    status=ComponentStatus.PENDING,
                    total_epochs=task.epochs
                )
                progress_dict[task.component.uuid] = 0

            with Pool(processes=self.max_workers) as pool:
                async_results = [
                    pool.apply_async(_fit_component, (task, progress_dict, states_dict))
                    for task in tasks
                ]

                pending = list(async_results)
                import time
                last_reported_epochs = {task.component.uuid: -1 for task in tasks}

                while pending:
                    for async_res in pending.copy():
                        if async_res.ready():
                            pending.remove(async_res)
                    if callbacks:
                        for task in tasks:
                            state = states_dict.get(task.component.uuid, ComponentState())
                            if (state.current_epoch > 0 and
                                state.current_epoch - 1 > last_reported_epochs[task.component.uuid]):
                                process_callbacks(
                                    callbacks,
                                    CallbackMessage(
                                        CallbackMethod.ON_EPOCH_COMPONENT_END,
                                        {
                                            "component": task.component,
                                            "epoch": state.current_epoch - 1,
                                            "status": state.status,
                                            "metrics": state.metrics
                                        }
                                    ),
                                    composite=self.composite
                                )
                                last_reported_epochs[task.component.uuid] = state.current_epoch - 1
                    time.sleep(0.5)

                results: List[FitResult] = [r.get() for r in async_results]

        for result in results:
            if not result.success:
                print(f"Error fitting component {result.component}: {result.error}")
            else:
                # Replace the component in the composite if needed.
                for i, comp in enumerate(self.composite.components):
                    if comp.uuid == result.component.uuid:
                        self.composite.components[i] = result.component
                        break

        if callbacks:
            process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_END, {}), composite=self.composite)

    def predict(self, data: Dict[str, Any], votes: Dict[str, np.ndarray], gating_mask: np.ndarray) -> None:
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._predict_component, comp, data, gating_mask, idx): comp
                for idx, comp in enumerate(self.composite.components)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    for key, score in result.items():
                        if key in votes:
                            votes[key] += score
                        else:
                            votes[key] = score
                except Exception as e:
                    comp = futures[future]
                    print(f"Error in prediction for component {comp.__class__.__name__}: {e}")
                    traceback.print_exc()

    def _predict_component(
        self, component: TMComponent, data: Dict[str, Any], gating_mask: np.ndarray, idx: int
    ) -> Dict[str, np.ndarray]:
        scores = compute_scores(component, data)
        mask = gating_mask[:, idx].reshape(-1, 1)
        masked_scores = scores * mask
        return {"composite": masked_scores, str(component): masked_scores}

class TMCompositeSingleCPU(TMCompositeRunner):
    """Single CPU (sequential) implementation of the composite runner."""

    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        if callbacks is None:
            callbacks = []
        preprocessed: List[Any] = [comp.preprocess(data) for comp in self.composite.components]
        epochs_left: List[int] = [comp.epochs for comp in self.composite.components]
        process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_BEGIN, {}), composite=self.composite)
        epoch = 0
        while any(epochs_left):
            for i, comp in enumerate(self.composite.components):
                if epochs_left[i] > 0:
                    process_callbacks(
                        callbacks,
                        CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_BEGIN, {"component": comp, "epoch": epoch}),
                        composite=self.composite,
                    )
                    comp.fit(data=preprocessed[i])
                    process_callbacks(
                        callbacks,
                        CallbackMessage(CallbackMethod.ON_EPOCH_COMPONENT_END, {"component": comp, "epoch": epoch}),
                        composite=self.composite,
                    )
                    epochs_left[i] -= 1
            epoch += 1
        process_callbacks(callbacks, CallbackMessage(CallbackMethod.ON_TRAIN_COMPOSITE_END, {}), composite=self.composite)

    def predict(self, data: Dict[str, Any], votes: Dict[str, np.ndarray], gating_mask: np.ndarray) -> None:
        for idx, comp in enumerate(self.composite.components):
            scores = compute_scores(comp, data)
            mask = gating_mask[:, idx].reshape(-1, 1)
            masked_scores = scores * mask
            key = str(comp)
            if "composite" in votes:
                votes["composite"] += masked_scores
            else:
                votes["composite"] = masked_scores
            if key in votes:
                votes[key] += masked_scores
            else:
                votes[key] = masked_scores

class TMComposite:
    """
    The main composite class: It holds a list of TMComponents and a gating function.
    It delegates fitting and prediction to an underlying runner (multiprocessing or single-CPU).
    """

    def __init__(
        self,
        components: Optional[List[TMComponent]] = None,
        gate_function: Optional[Type[BaseGate]] = None,
        gate_function_params: Optional[Dict[str, Any]] = None,
        use_multiprocessing: bool = False,
        **kwargs: Any
    ) -> None:
        self.components: List[TMComponent] = components or []
        self.use_multiprocessing = use_multiprocessing
        if gate_function_params is None:
            gate_function_params = {}
        self.gate_function_instance: BaseGate = (
            gate_function(self, **gate_function_params) if gate_function else LinearGate(self, **gate_function_params)
        )
        if use_multiprocessing:
            self.runner: TMCompositeRunner = TMCompositeMP(self, **kwargs)
        else:
            self.runner = TMCompositeSingleCPU(self, **kwargs)

    def fit(self, data: Dict[str, Any], callbacks: Optional[List[TMCompositeCallback]] = None) -> None:
        self.runner.fit(data, callbacks)

    def predict(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        votes: Dict[str, np.ndarray] = {}
        gating_mask: np.ndarray = self.gate_function_instance.predict(data)
        # Validate gating mask dimensions
        assert gating_mask.shape[1] == len(self.components), "Gating mask column count must equal number of components"
        assert gating_mask.shape[0] == data["Y"].shape[0], "Gating mask row count must equal number of samples"
        self.runner.predict(data, votes, gating_mask)
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
        files = [f for f in path.iterdir() if f.is_file() and f.suffix == f".{format}"]
        component_groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
        for file in files:
            parts = file.stem.split("-")
            epoch = int(parts[-1])
            component_name = "-".join(parts[:-1])
            component_groups[component_name].append((epoch, file))
        latest_files = [sorted(group, key=lambda x: x[0], reverse=True)[0][1] for group in component_groups.values()]
        self.components = []
        if format == "pkl":
            import pickle

            for component_path in latest_files:
                with open(component_path, "rb") as f:
                    loaded_component = pickle.load(f)
                    self.components.append(loaded_component)
        else:
            raise NotImplementedError(f"Format {format} not supported")

