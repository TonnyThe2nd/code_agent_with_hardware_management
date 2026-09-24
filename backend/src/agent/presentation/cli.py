import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich.console import Console
import psutil

from ...hardware.application.use_cases import DetectHardwareUseCase
from ...hardware.infrastructure.composite_probe import CompositeProbe
from ..application.model_selection import SelectModelUseCase

from ..application.use_cases import RunAgentSessionUseCase
from ..application.impact_review import ReviewImpactUseCase
from ..domain.value_objects import Message, MessageRole
from ..infrastructure.llm.ollama_provider import OllamaCatalog, OllamaProvider
from ..infrastructure.tools import build_default_registry
from .planning import plan


app = typer.Typer(help="Execute o agente local com Ollama.")
app.command("plan")(plan)


def configure_output() -> None:
    """Keep redirected Windows terminals from aborting on model Unicode output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")


@app.command("run")
def run(
    prompt: str = typer.Argument(...),
    workspace: Path = typer.Option(
        Path("."), "--workspace", "-w", exists=True, file_okay=False, resolve_path=True,
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", envvar="CODE_AGENT_MODEL",
        help="Modelo instalado no Ollama com suporte a ferramentas.",
    ),
    max_iterations: int = typer.Option(10, "--max-iter", min=1),
    base_url: str = typer.Option("http://localhost:11434", "--base-url", envvar="OLLAMA_BASE_URL"),
    timeout: float = typer.Option(120.0, "--timeout", min=0.1, envvar="CODE_AGENT_TIMEOUT"),
    allow_tests: bool = typer.Option(False, "--allow-tests", help="Permite pytest apenas em Docker sem rede."),
    test_image: str = typer.Option("code-agent-tests:local", "--test-image", envvar="CODE_AGENT_TEST_IMAGE"),
    num_ctx: int = typer.Option(4096, "--num-ctx", min=1),
    memory_budget: float | None = typer.Option(None, "--memory-budget", min=0.1,
                                              help="Orcamento utilizavel do servidor Ollama, em GiB."),
) -> None:
    configure_output()
    console = Console()
    error_console = Console(stderr=True)

    def on_message(message: Message) -> None:
        if message.role is MessageRole.ASSISTANT:
            if message.content:
                console.print(message.content, style="cyan", markup=False, highlight=False)
            for call in message.tool_calls:
                arguments = json.dumps(dict(call.arguments), ensure_ascii=False)
                console.print(
                    f"→ tool: {call.name}({arguments})",
                    style="yellow", markup=False, highlight=False,
                )
        elif message.role is MessageRole.TOOL:
            console.print(message.content[:200], style="dim", markup=False, highlight=False)

    try:
        catalog = OllamaCatalog(base_url)
        if model:
            selected_model = catalog.select(model)
        else:
            budget = memory_budget
            if budget is None:
                if urlparse(base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
                    raise ValueError("Ollama remoto: informe --memory-budget do servidor ou --model.")
                hardware = DetectHardwareUseCase(CompositeProbe()).execute()
                budget = hardware.budget_gb
                if hardware.budget_source == "ram":
                    budget = min(budget, psutil.virtual_memory().available / (1024 ** 3) * 0.85)
            selection = SelectModelUseCase().execute(catalog.models(), budget, num_ctx)
            selected_model = selection.name
            error_console.print(
                f"Modelo automatico: {selected_model} | estimativa {selection.estimated_gb:.1f} GiB "
                f"/ orcamento {selection.budget_gb:.1f} GiB | contexto {num_ctx}",
                markup=False,
            )
        registry = build_default_registry(workspace, allow_tests=allow_tests, test_image=test_image)
        provider = OllamaProvider(selected_model, base_url=base_url, tools_schema=registry.schemas(), timeout=timeout, num_ctx=num_ctx)
        use_case = RunAgentSessionUseCase(provider, registry, workspace, max_iterations, allow_tests=allow_tests)
        result = use_case.execute(prompt, on_message=on_message)
    except Exception as exc:
        error_console.print(
            f"Erro ao executar agente: {exc}", style="red", markup=False, highlight=False,
        )
        raise typer.Exit(code=1) from exc
    if result.final_message is None:
        error_console.print("Limite de iterações atingido sem resposta final.", style="yellow")
        raise typer.Exit(code=2)
    review = ReviewImpactUseCase().execute(registry)
    if review.changed_files:
        console.print("Arquivos impactados: " + ", ".join(review.changed_files), style="green", markup=False)
        console.print("Diff revisado: " + ("sim" if review.inspected_diff else "nao"), markup=False)
        if review.validation:
            console.print("Validacao: " + " | ".join(review.validation), style="green", markup=False)
        if review.failures:
            console.print("Falhas de ferramenta: " + " | ".join(review.failures), style="red", markup=False)
    console.print(result.final_message, style="green", markup=False, highlight=False)


@app.command("models")
def models(
    base_url: str = typer.Option("http://localhost:11434", "--base-url", envvar="OLLAMA_BASE_URL"),
) -> None:
    configure_output()
    console = Console()
    try:
        installed = OllamaCatalog(base_url).models()
        for item in installed:
            capabilities = ", ".join(item["capabilities"]) or "nao informadas"
            console.print(f"{item['name']} | {item['size'] / 1e9:.1f} GB | {capabilities}", markup=False)
        if not installed:
            console.print("Nenhum modelo instalado. Use ollama pull NOME:TAG.")
    except Exception as exc:
        Console(stderr=True).print(f"Falha ao consultar Ollama: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc
