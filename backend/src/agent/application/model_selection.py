from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSelection:
    name: str
    estimated_gb: float
    budget_gb: float


class SelectModelUseCase:
    """Use a conservative size heuristic, not a claim of exact inference memory."""

    def execute(self, models: list[dict[str, Any]], budget_gb: float, num_ctx: int = 4096) -> ModelSelection:
        if not 0 < budget_gb < float("inf") or num_ctx < 1:
            raise ValueError("Orcamento e contexto devem ser positivos")
        candidates: list[tuple[int, float, str]] = []
        for model in models:
            if "tools" not in model.get("capabilities", []):
                continue
            size = model.get("size", 0)
            parameters = model.get("parameter_count", 0)
            if not isinstance(size, (int, float)) or size <= 0:
                continue
            if not isinstance(parameters, int) or parameters < 0 or parameters > 250_000_000_000:
                continue
            context_limit = model.get("context_length", 0)
            if context_limit and context_limit < num_ctx:
                continue
            estimate = size / (1024 ** 3) * 1.25 + max(1.0, num_ctx / 4096)
            if estimate <= budget_gb:
                candidates.append((parameters, estimate, model["name"]))
        if not candidates:
            raise ValueError(
                f"Nenhum modelo instalado com tools cabe no orcamento estimado de {budget_gb:.1f} GiB. "
                "Instale um modelo menor ou use --model para uma escolha manual."
            )
        _, estimate, name = max(candidates)
        return ModelSelection(name, estimate, budget_gb)
