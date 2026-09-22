from __future__ import annotations
from dataclasses import dataclass
from .entities import HardwareProfile
from .value_objects import Memory

SAFETY_MARGIN = 0.70
KV_CACHE_OVERHEAD = 0.15

@dataclass(frozen=True)
class MemoryBudget:

    """orçamento de memória para o modelo"""
    usable: Memory
    source: str

    def cabe(self, modelo_gb: float) -> bool:
        """verifica se o modelo cabe na memória disponível"""
        return modelo_gb <= self.usable.gigabytes

    def max_params_b(self, bytes_por_param: float = 0.55) -> int:
        """calcula o número máximo de parâmetros que cabem na memória disponível"""
        return int(self.usable.gigabytes / (bytes_por_param * KV_CACHE_OVERHEAD))

    def __str__(self) -> str:
        return f"Memória utilizável: {self.usable} | Fonte: {self.source}"

class HardwareBudgetCalculator:
    """serviço de domínio: calcula o orçamento de memória."""

    def calculate(self, profile: HardwareProfile) -> MemoryBudget:
        ram_usable = profile.total_ram.gigabytes * SAFETY_MARGIN

        if profile.has_gpu and profile.capability.vram is not None:
            vram_usable = profile.capability.vram.gigabytes * SAFETY_MARGIN
            #o modelo roda na GPU OU na CPU. Pegamos o maior.
            if vram_usable >= ram_usable:
                return MemoryBudget(Memory(vram_usable), source="vram")
            return MemoryBudget(Memory(ram_usable), source="ram")

        return MemoryBudget(Memory(ram_usable), source="ram")