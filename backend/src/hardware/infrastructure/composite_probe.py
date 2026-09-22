from ..domain.entities import HardwareProfile
from ..domain.ports import HardwareProbe
from .system_probe import SystemProbe
from .nvidia_probe import NvidiaProbe
from .apple_probe import AppleProbe
# from .amd_probe import AmdProbe  # se implementar


class CompositeProbe:
    """
    Tenta os probes em ordem de prioridade.
    O primeiro disponível vence.
    """

    def __init__(self, probes: list[HardwareProbe] | None = None):
        self._probes = probes or [
            NvidiaProbe(),
            AppleProbe(),
            SystemProbe(),   # fallback universal, sempre disponível
        ]

    def is_available(self) -> bool:
        return True

    def detect(self) -> HardwareProfile:
        for probe in self._probes:
            if probe.is_available():
                return probe.detect()
        # Nunca deve chegar aqui, pois SystemProbe sempre está disponível
        return SystemProbe().detect()