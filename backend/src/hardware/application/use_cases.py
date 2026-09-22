from __future__ import annotations
from ..domain.entities import HardwareProfile
from ..domain.ports import HardwareProbe
from ..domain.services import HardwareBudgetCalculator, MemoryBudget
from .dto import HardwareDTO


class DetectHardwareUseCase:
    """caso de uso: detectar hardware e calcular orçamento.
    recebe o probe por injeção — não sabe qual implementação é.
    """

    def __init__(self, probe: HardwareProbe):
        self._probe = probe
        self._calculator = HardwareBudgetCalculator()

    def execute(self) -> HardwareDTO:
        profile: HardwareProfile = self._probe.detect()
        budget: MemoryBudget = self._calculator.calculate(profile)
        return self._to_dto(profile, budget)

    def _to_dto(self, profile: HardwareProfile, budget: MemoryBudget) -> HardwareDTO:
        return HardwareDTO(
            total_ram_gb=profile.total_ram.gigabytes,
            cpu_cores=profile.cpu_cores,
            backend=profile.capability.backend.value,
            device_name=profile.capability.device_name,
            vram_gb=profile.capability.vram.gigabytes if profile.capability.vram else None,
            budget_gb=budget.usable.gigabytes,
            budget_source=budget.source,
        )