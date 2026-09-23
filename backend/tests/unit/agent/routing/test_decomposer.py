from pathlib import Path

from src.agent.domain.value_objects import Message, MessageRole
from src.agent.domain.routing.services import TaskDecomposer
from src.agent.domain.routing.value_objects import TaskComplexity
from src.agent.infrastructure.catalog.yaml_loader import load_catalog
from src.agent.application.use_cases import PlanAndRouteUseCase
from src.hardware.application.dto import HardwareDTO


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, messages: tuple[Message, ...]) -> Message:
        return Message(MessageRole.ASSISTANT, self.content)


VALID = '[{"description":"List files","complexity":"simple","expected_output":"File list","depends_on":[]}]'


def test_decomposer_parseia_json_valido() -> None:
    subtasks = TaskDecomposer(FakeLLM(VALID)).decompose("Explore")
    assert subtasks[0].id == "1"
    assert subtasks[0].complexity is TaskComplexity.SIMPLE


def test_decomposer_faz_fallback_em_json_invalido() -> None:
    for content in ("invalid", "[]", '[{"description":"incomplete"}]'):
        subtasks = TaskDecomposer(FakeLLM(content)).decompose("Original task")
        assert len(subtasks) == 1
        assert subtasks[0].complexity is TaskComplexity.COMPLEX
        assert subtasks[0].description == "Original task"


def test_decomposer_extrai_json_em_texto_ao_redor() -> None:
    subtasks = TaskDecomposer(FakeLLM("Here is the plan:\n```json\n" + VALID + "\n```Done")).decompose("Task")
    assert subtasks[0].complexity is TaskComplexity.SIMPLE


def test_invalid_dependency_falls_back() -> None:
    content = VALID.replace('"depends_on":[]', '"depends_on":["future"]')
    assert TaskDecomposer(FakeLLM(content)).decompose("Task")[0].complexity is TaskComplexity.COMPLEX


def test_use_case_and_catalog() -> None:
    class Hardware:
        def execute(self) -> HardwareDTO:
            return HardwareDTO(8, 4, "cpu", "fake", None, 5, "ram")

    catalog = load_catalog(Path(__file__).resolve().parents[4] / "config/models.yaml")
    result = PlanAndRouteUseCase(Hardware(), FakeLLM(VALID), catalog).execute("Task")
    assert result.budget_gb == 5
    assert result.models_used == ["qwen2.5-coder:1.5b"]
    assert result.to_dict()["subtasks"][0]["depends_on"] == []
