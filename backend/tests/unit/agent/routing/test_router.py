import pytest

from src.agent.domain.routing.entities import ModelCatalog, RoutingPlan
from src.agent.domain.routing.value_objects import ModelSpec, ModelTier, Subtask, TaskComplexity
from src.agent.domain.routing.services import ModelRouter, NoModelFitsError
from src.agent.domain.routing.policies import RoutingPolicy


def catalog() -> ModelCatalog:
    return ModelCatalog((
        ModelSpec("small", 1.5, 1.1, ModelTier.EXECUTOR, 32768),
        ModelSpec("medium", 3, 2.2, ModelTier.EXECUTOR, 32768),
        ModelSpec("large", 7, 4.8, ModelTier.PLANNER, 32768),
        ModelSpec("largest", 14, 9.5, ModelTier.PLANNER, 32768),
    ))


def task(complexity: TaskComplexity) -> Subtask:
    return Subtask("1", "Task", complexity, "Result")


def test_simple_usa_menor_modelo() -> None:
    assert ModelRouter(catalog(), 10).route(task(TaskComplexity.SIMPLE)).name == "small"


def test_complex_usa_maior_modelo() -> None:
    assert ModelRouter(catalog(), 10).route(task(TaskComplexity.COMPLEX)).name == "largest"


def test_fallback_quando_nenhum_planner_cabe() -> None:
    assert ModelRouter(catalog(), 3).route(task(TaskComplexity.COMPLEX)).name == "medium"


def test_levanta_erro_quando_nada_cabe() -> None:
    with pytest.raises(NoModelFitsError):
        ModelRouter(catalog(), 0.5).route(task(TaskComplexity.SIMPLE))


def test_budget_zero_levanta_erro() -> None:
    with pytest.raises(NoModelFitsError):
        ModelRouter(catalog(), 0)


def test_moderate_and_exact_budget() -> None:
    assert ModelRouter(catalog(), 2.2).route(task(TaskComplexity.MODERATE)).name == "medium"


def test_plan_and_policy() -> None:
    plan = RoutingPlan("Task", [task(TaskComplexity.SIMPLE),
        Subtask("2", "Next", TaskComplexity.COMPLEX, "Result", depends_on=("1",))])
    assert not RoutingPolicy().is_viable(plan)
    ModelRouter(catalog(), 5).route_plan(plan)
    assert plan.models_in_plan() == plan.total_models_used == {"small", "large"}
    assert plan.subtasks[1].depends_on == ("1",)
    assert RoutingPolicy().is_viable(plan)
    assert len(RoutingPolicy().validate(plan)) == 1


def test_rejects_invalid_dependencies() -> None:
    with pytest.raises(ValueError):
        RoutingPlan("Task", [Subtask("1", "Task", TaskComplexity.SIMPLE, "Result", depends_on=("2",))])
