from pathlib import Path
import sys

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...hardware.application.dto import HardwareDTO
from ...hardware.application.use_cases import DetectHardwareUseCase
from ...hardware.infrastructure.composite_probe import CompositeProbe
from ..application.use_cases import PlanAndRouteUseCase, RunAgentSessionUseCase
from ..application.execute_plan import ExecuteRoutingPlanUseCase
from ..domain.value_objects import Message, MessageRole
from ..infrastructure.tools import build_default_registry
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
    execute: bool = typer.Option(False, "--execute", help="Executa as subtarefas e pode alterar arquivos."),
    max_iterations: int = typer.Option(10, "--max-iter", min=1),
    allow_tests: bool = typer.Option(False, "--allow-tests"),
    test_image: str = typer.Option("code-agent-tests:local", "--test-image"),
    action_mode: str = typer.Option("structured", "--action-mode", help="structured ou native"),
) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")
    console = Console()
    try:
        if not task.strip():
            raise ValueError("Task must not be empty")
        if action_mode not in ("structured", "native"):
            raise ValueError("--action-mode deve ser structured ou native")
        catalog = load_catalog(catalog_path)
        hardware = DetectHardwareUseCase(CompositeProbe()).execute()
        planner = (catalog.largest_that_fits(hardware.budget_gb, ModelTier.PLANNER)
                   or catalog.largest_that_fits(hardware.budget_gb, ModelTier.EXECUTOR))
        if planner is None:
            raise NoModelFitsError("Nenhum modelo do catalogo cabe no orcamento detectado.")
        installed = {item["name"]: item for item in OllamaCatalog().models()}
        if planner.name not in installed:
            raise ValueError(f"Planner nao instalado: {planner.name}. Execute ollama pull {planner.name}.")
        provider = OllamaProvider(planner.name, timeout=timeout, keep_alive=0)
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
        if not execute:
            console.print(f"Plano apenas; nenhum arquivo alterado em {workspace}.", markup=False)
            return
        for name in {item.assigned_model for item in result.subtasks}:
            if name not in installed or "tools" not in installed[name].get("capabilities", []):
                raise ValueError(f"Execucao requer modelo instalado com tools: {name}. Nenhuma subtarefa iniciada.")
        registry = build_default_registry(workspace, allow_tests=allow_tests, test_image=test_image)

        def runner(model: str) -> RunAgentSessionUseCase:
            llm = OllamaProvider(model, tools_schema=registry.schemas(), timeout=timeout,
                                 num_ctx=4096, keep_alive=0, action_mode=action_mode)
            return RunAgentSessionUseCase(llm, registry, workspace, max_iterations,
                                         allow_tests=allow_tests, fail_on_tool_error=True,
                                         require_tool_activity=True)

        def on_message(message: Message) -> None:
            if message.content:
                content = message.content[:200] if message.role is MessageRole.TOOL else message.content
                console.print(content, style="dim" if message.role is MessageRole.TOOL else "cyan", markup=False)
            for call in message.tool_calls:
                console.print(f"tool: {call.name}", style="yellow", markup=False)

        def on_subtask(task_id: str, model: str) -> None:
            console.print(f"Executando {task_id} com {model}", style="yellow", markup=False)

        def fallback_model(current: str) -> str | None:
            available = {name for name, info in installed.items() if "tools" in info.get("capabilities", [])}
            replacement = catalog.next_larger(current, hardware.budget_gb, available)
            if replacement is not None:
                console.print(f"{current} nao executou ferramentas; tentando {replacement.name}.",
                              style="yellow", markup=False)
                return replacement.name
            return None

        execution = ExecuteRoutingPlanUseCase(runner, fallback_model).execute(result, on_message, on_subtask)
        if not execution.completed:
            console.print(f"Interrompido em {execution.failed_subtask}: {execution.error}. "
                          "Alteracoes anteriores permanecem no workspace.", style="red", markup=False)
            raise typer.Exit(2)
        if registry.changed_files:
            console.print("Arquivos escritos pelas ferramentas: " + ", ".join(sorted(registry.changed_files)),
                          style="green", markup=False)
            console.print("Sessoes encerradas. Escritas registradas nao comprovam que a alteracao funciona; revise o diff.")
        else:
            console.print("Sessoes encerradas SEM alteracoes de arquivos registradas. "
                          "Nao foi comprovada a implementacao solicitada.", style="yellow")
    except typer.Exit:
        raise
    except Exception as exc:
        Console(stderr=True).print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc
