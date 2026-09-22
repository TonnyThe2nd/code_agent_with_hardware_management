from .hardware.infrastructure.composite_probe import CompositeProbe
from .hardware.application.use_cases import DetectHardwareUseCase
from .hardware.presentation.cli import app
from .agent.presentation.cli import app as agent_app


app.add_typer(agent_app, name="agent")


def build_hardware_use_case() -> DetectHardwareUseCase:
    return DetectHardwareUseCase(probe=CompositeProbe())


if __name__ == "__main__":
    app()
