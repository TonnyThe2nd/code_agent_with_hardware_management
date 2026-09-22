from typing import Protocol
from .entities import HardwareProfile


class HardwareProbe(Protocol):
    """contrato: qualquer coisa que consiga detectar hardware."""

    def is_available(self) -> bool:
        """se este probe consegue rodar nesta máquina."""
        ...

    def detect(self) -> HardwareProfile:
        """retorna o perfil de hardware detectado."""
        ...