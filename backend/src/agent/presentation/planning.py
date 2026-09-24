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
from ..application.impact_review import ReviewImpactUseCase
from ..domain.value_objects import Message, MessageRole
from ..infrastructure.tools import build_default_registry
from ..domain.routing.value_objects import ModelTier
from ..domain.routing.entities import ModelCatalog
from ..domain.routing.services import NoModelFitsError
from ..infrastructure.catalog.yaml_loader import load_catalog
from ..infrastructure.llm.ollama_provider import OllamaCatalog, OllamaProvider


def select_planner_model(
    requested_model: str | None, installed: dict[str, dict], catalog: ModelCatalog, budget_gb: float,
) -> str:
    if requested_model:
        candidates = (requested_model, requested_model + ":latest")
        selected = next((name for name in candidates if name in installed), None)
        if selected is None:
            raise ValueError(
                f"Modelo de planejamento nao instalado: {requested_model}. "
                f"Execute ollama pull {requested_model}."
            )
        return selected
    planner = (catalog.largest_that_fits(budget_gb, ModelTier.PLANNER)
               or catalog.largest_that_fits(budget_gb, ModelTier.EXECUTOR))
    if planner is None:
        raise NoModelFitsError("Nenhum modelo do catalogo cabe no orcamento detectado.")
    candidates = (planner.name, planner.name + ":latest")
    selected = next((name for name in candidates if name in installed), None)
    if selected is None:
        raise ValueError(f"Planner nao instalado: {planner.name}. Execute ollama pull {planner.name}.")
    return selected


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
    model: str | None = typer.Option(
        None, "--model", "-m", envvar="CODE_AGENT_PLANNER_MODEL",
        help="Modelo instalado usado somente para decompor a tarefa.",
    ),
    timeout: float = typer.Option(120.0, "--timeout", min=0.1),
    execute: bool = typer.Option(False, "--execute", help="Executa as subtarefas e pode alterar arquivos."),
    max_iterations: int = typer.Option(10, "--max-iter", min=1),
    allow_tests: bool = typer.Option(False, "--allow-tests"),
    test_image: str = typer.Option("code-agent-tests:local", "--test-image"),
    action_mode: str = typer.Option("structured", "--action-mode", help="structured ou native"),
    confirm_high_risk: bool = typer.Option(False, "--confirm-high-risk", help="Confirma execucao de subtarefas de alto risco."),
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
        installed = {item["name"]: item for item in OllamaCatalog().models()}
        planner_model = select_planner_model(model, installed, catalog, hardware.budget_gb)
        provider = OllamaProvider(planner_model, timeout=timeout, keep_alive=0)
        result = PlanAndRouteUseCase(HardwareSnapshot(hardware), provider, catalog).execute(task)
        console.print(f"Planner: {planner_model} | Orcamento: {result.budget_gb:.1f} GB", markup=False)
        table = Table("#", "Subtarefa", "Complexidade", "Arquivos alvo", "Risco", "Modelo", "Evidencia")
        for item in result.subtasks:
            table.add_row(*(Text(value) for value in (
                item.id, item.description, item.complexity,
                ", ".join(item.target_files) or "a descobrir", item.risk,
                item.assigned_model or "-", item.expected_evidence,
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
        if any(item.risk == "high" for item in result.subtasks) and not confirm_high_risk:
            raise ValueError("Plano contem alteracao de alto risco. Revise o plano e execute novamente com --confirm-high-risk.")
        for name in {item.assigned_model for item in result.subtasks}:
            if name not in installed or "tools" not in installed[name].get("capabilities", []):
                raise ValueError(f"Execucao requer modelo instalado com tools: {name}. Nenhuma subtarefa iniciada.")
        registry = build_default_registry(workspace, allow_tests=allow_tests, test_image=test_image)

        def runner(model: str) -> RunAgentSessionUseCase:
            llm = OllamaProvider(model, tools_schema=registry.schemas(), timeout=timeout,
                                 num_ctx=4096, keep_alive=0, action_mode=action_mode)
            return RunAgentSessionUseCase(llm, registry, workspace, max_iterations,
                                         allow_tests=allow_tests, fail_on_tool_error=False,
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
        review = ReviewImpactUseCase().execute(registry)
        if review.changed_files:
            console.print("Arquivos impactados: " + ", ".join(review.changed_files), style="green", markup=False)
            console.print("Diff revisado: " + ("sim" if review.inspected_diff else "nao"), markup=False)
            if review.validation:
                console.print("Validacao: " + " | ".join(review.validation), style="green", markup=False)
            if review.failures:
                console.print("Evidencias de falha: " + " | ".join(review.failures), style="red", markup=False)
            elif not review.inspected_diff:
                console.print("Mudanca sem revisao de diff comprovada; revise antes de integrar.", style="yellow", markup=False)
        else:
            console.print("Sessoes encerradas SEM alteracoes de arquivos registradas. "
                          "Nao foi comprovada a implementacao solicitada.", style="yellow")
    except typer.Exit:
        raise
    except Exception as exc:
        Console(stderr=True).print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc
