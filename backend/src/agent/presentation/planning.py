from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...hardware.application.dto import HardwareDTO
from ...hardware.application.use_cases import DetectHardwareUseCase
from ...hardware.infrastructure.composite_probe import CompositeProbe
from ..application.use_cases import PlanAndRouteUseCase
from ..domain.routing.value_objects import ModelTier
from ..domain.routing.services import NoModelFitsError
from ..infrastructure.catalog.yaml_loader import load_catalog
from ..infrastructure.llm.ollama_provider import OllamaCatalog, OllamaProvider


class HardwareSnapshot:
    """Keep planner selection and subtask routing on the same measured budget."""

    def __init__(self, hardware: HardwareDTO) -> None:
        self._hardware = hardware

    def execute(self) -> HardwareDTO:
        return self._hardware


def plan(
    task: str = typer.Argument(...),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", exists=True,
                                  file_okay=False, resolve_path=True),
    catalog_path: Path = typer.Option(Path(__file__).resolve().parents[3] / "config/models.yaml", "--catalog"),
    timeout: float = typer.Option(120.0, "--timeout", min=0.1),
) -> None:
    console = Console()
    try:
        if not task.strip():
            raise ValueError("Task must not be empty")
        catalog = load_catalog(catalog_path)
        hardware = DetectHardwareUseCase(CompositeProbe()).execute()
        planner = (catalog.largest_that_fits(hardware.budget_gb, ModelTier.PLANNER)
                   or catalog.largest_that_fits(hardware.budget_gb, ModelTier.EXECUTOR))
        if planner is None:
            raise NoModelFitsError("Nenhum modelo do catalogo cabe no orcamento detectado.")
        installed = {item["name"] for item in OllamaCatalog().models()}
        if planner.name not in installed:
            raise ValueError(f"Planner nao instalado: {planner.name}. Execute ollama pull {planner.name}.")
        provider = OllamaProvider(planner.name, timeout=timeout)
        result = PlanAndRouteUseCase(HardwareSnapshot(hardware), provider, catalog).execute(task)
        console.print(f"Planner: {planner.name} | Orcamento: {result.budget_gb:.1f} GB", markup=False)
        table = Table("#", "Subtarefa", "Complexidade", "Modelo", "Depende de")
        for item in result.subtasks:
            table.add_row(*(Text(value) for value in (
                item.id, item.description, item.complexity, item.assigned_model or "-",
                ", ".join(item.depends_on) or "-",
            )))
        console.print(table)
        for warning in result.warnings:
            console.print(warning, style="yellow", markup=False)
        for name in result.models_used:
            if name not in installed:
                console.print(f"Modelo planejado ainda nao instalado: ollama pull {name}", style="yellow", markup=False)
        console.print(f"Plano apenas; nenhum arquivo alterado em {workspace}.", markup=False)
    except Exception as exc:
        Console(stderr=True).print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc
