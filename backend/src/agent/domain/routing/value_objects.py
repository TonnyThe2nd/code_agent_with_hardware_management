from dataclasses import dataclass
from enum import Enum


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ModelTier(str, Enum):
    EXECUTOR = "executor"
    PLANNER = "planner"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    size_gb: float
    tier: ModelTier
    context_window: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model requires a name")
        for value in (self.params_b, self.size_gb):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError("Model size and parameters must be positive and finite")
        if not isinstance(self.tier, ModelTier):
            raise ValueError("Invalid model tier")
        if type(self.context_window) is not int or self.context_window < 1:
            raise ValueError("Context window must be positive")


@dataclass(frozen=True)
class Subtask:
    id: str
    description: str
    complexity: TaskComplexity
    expected_output: str
    assigned_model: str | None = None
    depends_on: tuple[str, ...] = ()
    target_files: tuple[str, ...] = ()
    risk: str = "medium"
    expected_evidence: str = "Relevant inspection and validation output"
    success_criteria: str = "Requested behavior is implemented without regressions"

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.id, self.description, self.expected_output)):
            raise ValueError("Subtask requires id, description and expected output")
        if not isinstance(self.complexity, TaskComplexity):
            raise ValueError("Invalid task complexity")
        if self.assigned_model is not None and (
            not isinstance(self.assigned_model, str) or not self.assigned_model.strip()
        ):
            raise ValueError("Invalid assigned model")
        if not isinstance(self.depends_on, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.depends_on
        ):
            raise ValueError("Dependencies must be immutable ids")
        if self.id in self.depends_on or len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("Self dependencies and duplicate dependencies are forbidden")
        if not isinstance(self.target_files, tuple) or any(not isinstance(path, str) or not path.strip()
                                                           for path in self.target_files):
            raise ValueError("Target files must be immutable nonempty paths")
        if self.risk not in {"low", "medium", "high"}:
            raise ValueError("Risk must be low, medium, or high")
        if not isinstance(self.expected_evidence, str) or not self.expected_evidence.strip():
            raise ValueError("Expected evidence must not be empty")
        if not isinstance(self.success_criteria, str) or not self.success_criteria.strip():
            raise ValueError("Success criteria must not be empty")
