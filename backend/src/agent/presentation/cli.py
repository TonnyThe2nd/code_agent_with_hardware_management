import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from ..application.use_cases import RunAgentSessionUseCase
from ..domain.value_objects import Message, MessageRole
from ..infrastructure.llm.ollama_provider import OllamaCatalog, OllamaProvider
from ..infrastructure.tools import build_default_registry


app = typer.Typer(help="Execute o agente local com Ollama.")


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
        selected_model = OllamaCatalog(base_url).select(model)
        registry = build_default_registry(workspace, allow_tests=allow_tests, test_image=test_image)
        provider = OllamaProvider(selected_model, base_url=base_url, tools_schema=registry.schemas(), timeout=timeout)
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
