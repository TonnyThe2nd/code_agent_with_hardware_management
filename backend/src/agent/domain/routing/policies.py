from .entities import RoutingPlan
from .value_objects import TaskComplexity


class RoutingPolicy:
    def __init__(self, min_executor_params_b: float = 1.0, max_complex_ratio: float = 0.4) -> None:
        if not 0 < min_executor_params_b < float("inf") or not 0 <= max_complex_ratio <= 1:
            raise ValueError("Invalid routing policy limits")
        self.min_executor_params_b = min_executor_params_b
        self.max_complex_ratio = max_complex_ratio

    def validate(self, plan: RoutingPlan) -> list[str]:
        warnings: list[str] = []
        if not plan.subtasks:
            warnings.append("Plano sem subtarefas")
        complex_count = sum(item.complexity is TaskComplexity.COMPLEX for item in plan.subtasks)
        if plan.subtasks and complex_count / len(plan.subtasks) > self.max_complex_ratio:
            warnings.append("Muitas subtarefas complexas, considere simplificar")
        for item in plan.subtasks:
            if item.assigned_model is None:
                warnings.append(f"Subtarefa {item.id} sem modelo atribuído")
        count = len(plan.models_in_plan())
        if count > 2:
            warnings.append(f"Plano usa {count} modelos, troca de contexto pode custar caro")
        return warnings

    def is_viable(self, plan: RoutingPlan) -> bool:
        return bool(plan.subtasks) and all(item.assigned_model is not None for item in plan.subtasks)
