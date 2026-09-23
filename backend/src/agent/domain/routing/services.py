import json
import re
from dataclasses import replace

from ..ports import LLMProvider
from ..value_objects import Message, MessageRole
from .entities import ModelCatalog, RoutingPlan
from .value_objects import ModelSpec, ModelTier, Subtask, TaskComplexity


DECOMPOSITION_SYSTEM_PROMPT = (
    "Decompose the task into an ordered JSON array, no tool calls. "
    "Each item has id (string), description, complexity, expected_output, depends_on (array of earlier ids). "
    "Use ids 1, 2, 3, ... and only backward dependencies. "
    "SIMPLE: rename, format, extract, list, move. "
    "MODERATE: implement a function, write a test, local refactoring. "
    "COMPLEX: architecture, deep debugging, design decisions, cross-module integration. "
    "Use complexity values simple, moderate, complex. Never select a model."
)


class NoModelFitsError(Exception):
    pass


class TaskDecomposer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def decompose(self, task: str) -> list[Subtask]:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must not be empty")
        response = self._llm.complete((
            Message(MessageRole.SYSTEM, DECOMPOSITION_SYSTEM_PROMPT),
            Message(MessageRole.USER, task),
        ))
        try:
            if response.role is not MessageRole.ASSISTANT or response.tool_calls:
                raise ValueError("Expected a planner response without tool calls")
            match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if match is None:
                raise ValueError("Missing JSON array")
            items = json.loads(match.group())
            if not isinstance(items, list) or not items:
                raise ValueError("Empty decomposition")
            subtasks: list[Subtask] = []
            for index, item in enumerate(items, 1):
                dependencies = item.get("depends_on", [])
                if not isinstance(dependencies, list):
                    raise ValueError("Dependencies must be an array")
                subtasks.append(Subtask(
                    id=item.get("id", str(index)), description=item["description"],
                    complexity=TaskComplexity(item["complexity"]),
                    expected_output=item["expected_output"], depends_on=tuple(dependencies),
                ))
            return RoutingPlan(original_task=task, subtasks=subtasks).subtasks
        except (ValueError, TypeError, KeyError, AttributeError):
            return [Subtask("1", task, TaskComplexity.COMPLEX, "Complete the original task")]


class ModelRouter:
    def __init__(self, catalog: ModelCatalog, budget_gb: float) -> None:
        if isinstance(budget_gb, bool) or not isinstance(budget_gb, (int, float)) or not 0 < budget_gb < float("inf"):
            raise NoModelFitsError("Memory budget must be positive and finite")
        self._catalog = catalog
        self._budget_gb = budget_gb

    def route(self, subtask: Subtask) -> ModelSpec:
        candidates = self._catalog.fits(self._budget_gb)
        if not candidates:
            raise NoModelFitsError("No catalog model fits the hardware budget")
        executors = [model for model in candidates if model.tier is ModelTier.EXECUTOR]
        planners = [model for model in candidates if model.tier is ModelTier.PLANNER]
        key = lambda model: (model.params_b, model.size_gb, model.name)
        if subtask.complexity is TaskComplexity.SIMPLE:
            return min(executors or candidates, key=key)
        if subtask.complexity is TaskComplexity.MODERATE:
            return max(executors or candidates, key=key)
        return max(planners or executors, key=key)

    def route_plan(self, plan: RoutingPlan) -> RoutingPlan:
        routed = [replace(item, assigned_model=self.route(item).name) for item in plan.subtasks]
        plan.subtasks = routed
        plan.total_models_used = plan.models_in_plan()
        return plan
