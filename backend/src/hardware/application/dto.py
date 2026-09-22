from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareDTO:
    total_ram_gb: float
    cpu_cores: int
    backend: str
    device_name: str
    vram_gb: float | None
    budget_gb: float
    budget_source: str

    def to_dict(self) -> dict:
        return {
            "total_ram_gb": self.total_ram_gb,
            "cpu_cores": self.cpu_cores,
            "backend": self.backend,
            "device_name": self.device_name,
            "vram_gb": self.vram_gb,
            "budget_gb": round(self.budget_gb, 2),
            "budget_source": self.budget_source,
        }