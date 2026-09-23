from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class AgentRunResult:
    session_id: str
    final_message: str | None
    iterations: int
    messages_count: int

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "session_id": self.session_id,
            "final_message": self.final_message,
            "iterations": self.iterations,
            "messages_count": self.messages_count,
        }


@dataclass(frozen=True)
class SubtaskDTO:
    id: str
    description: str
    complexity: str
    expected_output: str
    assigned_model: str | None
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["depends_on"] = list(self.depends_on)
        return result


@dataclass(frozen=True)
class RoutingPlanDTO:
    plan_id: str
    original_task: str
    subtasks: list[SubtaskDTO]
    budget_gb: float
    models_used: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id, "original_task": self.original_task,
            "subtasks": [item.to_dict() for item in self.subtasks],
            "budget_gb": self.budget_gb, "models_used": list(self.models_used),
            "warnings": list(self.warnings),
        }
