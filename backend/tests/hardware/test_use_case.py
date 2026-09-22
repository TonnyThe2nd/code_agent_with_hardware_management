from src.hardware.domain.entities import HardwareProfile
from src.hardware.domain.value_objects import Memory, ComputeCapability, ComputeBackend
from src.hardware.application.use_cases import DetectHardwareUseCase


class FakeProbe:
    def is_available(self) -> bool:
        return True

    def detect(self) -> HardwareProfile:
        return HardwareProfile(
            total_ram=Memory(64),
            cpu_cores=12,
            capability=ComputeCapability(
                backend=ComputeBackend.CUDA,
                device_name="Fake GPU",
                vram=Memory(12),
            ),
        )


def test_use_case_com_probe_fake():
    use_case = DetectHardwareUseCase(probe=FakeProbe())
    dto = use_case.execute()
    assert dto.total_ram_gb == 64
    assert dto.vram_gb == 12
    assert dto.budget_source == "ram"