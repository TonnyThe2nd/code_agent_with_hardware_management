from typing import Callable

from src.agent.application.dto import AgentRunResult, RoutingPlanDTO, SubtaskDTO
from src.agent.application.errors import NoToolActivityError
from src.agent.application.execute_plan import ExecuteRoutingPlanUseCase
from src.agent.domain.value_objects import Message
from src.agent.domain.routing.entities import ModelCatalog
from src.agent.domain.routing.value_objects import ModelSpec, ModelTier


def test_escalates_only_absent_tool_activity() -> None:
    for error in (NoToolActivityError("no tools"), RuntimeError("partial write")):
        calls: list[str] = []

        class Runner:
            def __init__(self, model: str) -> None:
                self.model = model

            def execute(self, user_prompt: str, on_message: Callable[[Message], None] | None = None) -> AgentRunResult:
                calls.append(self.model)
                if self.model == "small":
                    raise error
                return AgentRunResult("s", "done", 2, 5)

        plan = RoutingPlanDTO("p", "task", [SubtaskDTO("1", "task", "simple", "result", "small")], 5, ["small"], [])
        result = ExecuteRoutingPlanUseCase(Runner, lambda name: "large").execute(plan)
        assert result.completed is isinstance(error, NoToolActivityError)
        assert calls == (["small", "large"] if result.completed else ["small"])


def test_fallback_respects_budget_and_installed_models() -> None:
    catalog = ModelCatalog(tuple(ModelSpec(name, params, size, ModelTier.EXECUTOR, 4096)
                                 for name, params, size in (("small", 1, 1), ("medium", 3, 2), ("large", 7, 4))))
    assert catalog.next_larger("small", 5, {"large"}).name == "large"
    assert catalog.next_larger("small", 3, {"large"}) is None
    assert catalog.next_larger("large", 5, {"small", "medium"}) is None
