from src.hardware.domain.entities import HardwareProfile
from src.hardware.domain.value_objects import Memory, ComputeCapability, ComputeBackend
from src.hardware.domain.services import HardwareBudgetCalculator


def test_budget_cpu_only():
    profile = HardwareProfile(
        total_ram=Memory(32),
        cpu_cores=8,
        capability=ComputeCapability(
            backend=ComputeBackend.CPU,
            device_name="Intel i7",
        ),
    )
    budget = HardwareBudgetCalculator().calculate(profile)
    assert budget.source == "ram"
    assert abs(budget.usable.gigabytes - 22.4) < 0.01


def test_budget_prefere_vram_quando_maior():
    profile = HardwareProfile(
        total_ram=Memory(16),
        cpu_cores=16,
        capability=ComputeCapability(
            backend=ComputeBackend.CUDA,
            device_name="RTX 4090",
            vram=Memory(24),
        ),
    )
    budget = HardwareBudgetCalculator().calculate(profile)
    assert budget.source == "vram"
    assert abs(budget.usable.gigabytes - 16.8) < 0.01