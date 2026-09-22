import platform
import subprocess

from ..domain.entities import HardwareProfile
from ..domain.value_objects import Memory, ComputeCapability, ComputeBackend
from .system_probe import SystemProbe


class AppleProbe:
    """detecta Apple Silicon (Metal). Memória unificada = RAM."""

    def __init__(self):
        self._fallback = SystemProbe()

    def is_available(self) -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def detect(self) -> HardwareProfile:
        if not self.is_available():
            return self._fallback.detect()

        base = self._fallback.detect()
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        except Exception:
            chip = "Apple Silicon"

        return HardwareProfile(
            total_ram=base.total_ram,
            cpu_cores=base.cpu_cores,
            capability=ComputeCapability(
                backend=ComputeBackend.METAL,
                device_name=chip,
                vram=base.total_ram,   
            ),
        )