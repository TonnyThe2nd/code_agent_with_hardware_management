from dataclasses import dataclass, field
from uuid import uuid4

from .value_objects import ModelSpec, ModelTier, Subtask


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[ModelSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.models, tuple) or not self.models:
            raise ValueError("Catalog must be a nonempty tuple")
        if any(not isinstance(model, ModelSpec) for model in self.models):
            raise ValueError("Invalid catalog entry")
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Duplicate model names")

    def by_tier(self, tier: ModelTier) -> list[ModelSpec]:
        return [model for model in self.models if model.tier is tier]

    def fits(self, budget_gb: float) -> list[ModelSpec]:
        return [model for model in self.models if model.size_gb <= budget_gb]

    def largest_that_fits(self, budget_gb: float, tier: ModelTier | None = None) -> ModelSpec | None:
        candidates = [model for model in self.fits(budget_gb) if tier is None or model.tier is tier]
        return max(candidates, key=lambda model: (model.params_b, model.size_gb, model.name), default=None)

    def smallest(self) -> ModelSpec:
        return min(self.models, key=lambda model: (model.params_b, model.size_gb, model.name))

    def next_larger(self, current: str, budget_gb: float, available: set[str]) -> ModelSpec | None:
        previous = next((model for model in self.models if model.name == current), None)
        if previous is None:
            return None
        candidates = [model for model in self.fits(budget_gb)
                      if model.name in available and model.params_b > previous.params_b]
        return min(candidates, key=lambda model: (model.params_b, model.size_gb, model.name), default=None)


@dataclass
class RoutingPlan:
    original_task: str
    subtasks: list[Subtask] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    total_models_used: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not isinstance(self.original_task, str) or not self.original_task.strip():
            raise ValueError("Original task must not be empty")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Plan requires an id")
        initial = list(self.subtasks)
        self.subtasks = []
        for subtask in initial:
            self.add_subtask(subtask)
        self.total_models_used = self.models_in_plan()

    def add_subtask(self, subtask: Subtask) -> None:
        if not isinstance(subtask, Subtask):
            raise ValueError("Invalid subtask")
        previous = {item.id for item in self.subtasks}
        if subtask.id in previous or not set(subtask.depends_on) <= previous:
            raise ValueError("Dependencies must reference earlier unique subtasks")
        self.subtasks.append(subtask)
        self.total_models_used = self.models_in_plan()

    def models_in_plan(self) -> set[str]:
        return {item.assigned_model for item in self.subtasks if item.assigned_model is not None}
