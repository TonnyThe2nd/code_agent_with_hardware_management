import pytest

from src.agent.domain.routing.entities import ModelCatalog
from src.agent.domain.routing.value_objects import ModelSpec, ModelTier
from src.agent.presentation.planning import select_planner_model


def catalog() -> ModelCatalog:
    return ModelCatalog((
        ModelSpec("small", 7, 4.8, ModelTier.EXECUTOR, 32768),
        ModelSpec("large", 27, 18.4, ModelTier.PLANNER, 32768),
    ))


def test_explicit_planner_model_overrides_automatic_selection() -> None:
    assert select_planner_model("small", {"small": {}, "large": {}}, catalog(), 32) == "small"


def test_explicit_planner_model_accepts_implicit_latest_tag() -> None:
    assert select_planner_model("small", {"small:latest": {}}, catalog(), 32) == "small:latest"


def test_explicit_planner_model_must_be_installed() -> None:
    with pytest.raises(ValueError, match="nao instalado"):
        select_planner_model("missing", {}, catalog(), 32)


def test_automatic_planner_selection_accepts_implicit_latest_tag() -> None:
    assert select_planner_model(None, {"large:latest": {}}, catalog(), 32) == "large:latest"
