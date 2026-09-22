from .hardware.infrastructure.composite_probe import CompositeProbe
from .hardware.application.use_cases import DetectHardwareUseCase


def build_hardware_use_case() -> DetectHardwareUseCase:
    return DetectHardwareUseCase(probe=CompositeProbe())


if __name__ == "__main__":
    from .hardware.presentation.cli import app
    app()