from __future__ import annotations
from dataclasses import dataclass
from .value_objects import ComputeBackend, ComputeCapability, Memory

@dataclass(frozen=True)
class HardwareProfile:

    """perfil de hardware do usuário"""
    capability: ComputeCapability
    total_ram: Memory
    cpu_cores: int
    def __str__(self) -> str:
        return (
            f"RAM: {self.total_ram} | "
            f"CPU: {self.cpu_cores} cores | "
            f"Compute: {self.capability}"
        )

    @property
    def has_gpu(self) -> bool:
        return self.capability.has_gpu

    def __post_init__(self):
        if self.cpu_cores <= 0:
            raise ValueError("Número de núcleos da CPU deve ser positivo")