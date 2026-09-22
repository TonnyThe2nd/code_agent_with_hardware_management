from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class ComputeBackend(Enum):
    CPU = "cpu"
    GPU = "gpu"
    METAL = "metal"
    CUDA = "cuda"
    ROCM = "rocm"

@dataclass(frozen=True)
class Memory:

    """memória em GB pois é imutável e nunca negativa"""
    gigabytes: float

    def __post_init__(self):
        if self.gigabytes < 0:
            raise ValueError("Memória não pode ser negativa")

    def __add__(self, other: "Memory") -> "Memory":
        return Memory(self.gigabytes + other.gigabytes)

    def __str__(self) -> str:
        return f"{self.gigabytes} GB"

@dataclass(frozen=True)
class ComputeCapability:

    """capacidade de computação do hardware"""
    backend: ComputeBackend
    device_name: str
    vram: Memory | None = None

    @property
    def has_gpu(self) -> bool:
        return self.backend != ComputeBackend.CPU and self.vram is not None