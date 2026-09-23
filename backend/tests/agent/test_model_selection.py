from src.agent.application.model_selection import SelectModelUseCase


def test_selects_largest_fitting_tool_model() -> None:
    models = [
        {"name": "small", "size": 1024 ** 3, "parameter_count": 1_000_000_000, "capabilities": ["tools"]},
        {"name": "medium", "size": 3 * 1024 ** 3, "parameter_count": 4_000_000_000, "capabilities": ["tools"]},
        {"name": "large", "size": 10 * 1024 ** 3, "parameter_count": 14_000_000_000, "capabilities": ["tools"]},
        {"name": "embedding", "size": 1024 ** 3, "parameter_count": 20_000_000_000, "capabilities": ["embedding"]},
    ]
    selector = SelectModelUseCase()
    assert selector.execute(models, 3).name == "small"
    assert selector.execute(models, 6).name == "medium"
    assert selector.execute(models, 16).name == "large"
    assert selector.execute(models, 6, 16384).name == "small"


def test_supports_235b_without_selecting_models_above_250b() -> None:
    models = [
        {"name": "235b", "size": 142_000_000_000, "parameter_count": 235_000_000_000, "capabilities": ["tools"]},
        {"name": "480b", "size": 290_000_000_000, "parameter_count": 480_000_000_000, "capabilities": ["tools"]},
    ]
    assert SelectModelUseCase().execute(models, 220).name == "235b"
    assert SelectModelUseCase().execute(models, 800).name == "235b"


def test_refuses_when_no_model_fits_or_context_is_unsupported() -> None:
    models = [{"name": "model", "size": 3 * 1024 ** 3, "capabilities": ["tools"], "context_length": 2048}]
    for budget in (1, 100):
        try:
            SelectModelUseCase().execute(models, budget, 4096)
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible model selected")
