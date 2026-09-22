import platform
import psutil

from ..domain.entities import HardwareProfile
from ..domain.value_objects import Memory, ComputeCapability, ComputeBackend


class SystemProbe:
    """probe básico: sempre disponível, só detecta RAM e CPU."""

    def is_available(self) -> bool:
        return True

    def detect(self) -> HardwareProfile:
        ram = Memory(psutil.virtual_memory().total / (1024 ** 3))
        cpu_cores = psutil.cpu_count(logical=False) or 1
        device = platform.processor() or platform.machine() or "unknown"

        return HardwareProfile(
            total_ram=ram,
            capability=ComputeCapability(
                backend=ComputeBackend.CPU,
                device_name=device,
                vram=None,
            ),
            cpu_cores=cpu_cores,
        )