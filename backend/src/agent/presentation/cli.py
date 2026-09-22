import json
from pathlib import Path

import typer
from rich.console import Console

from ..application.use_cases import RunAgentSessionUseCase
from ..domain.value_objects import Message, MessageRole
from ..infrastructure.llm.ollama_provider import OllamaProvider
from ..infrastructure.tools import build_default_registry


app = typer.Typer(help="Execute o agente local com Ollama.")


@app.command("run")
def run(
    prompt: str = typer.Argument(...),
    workspace: Path = typer.Option(
        Path("."), "--workspace", "-w", exists=True, file_okay=False, resolve_path=True,
    ),
    model: str = typer.Option(
        "qwen2.5-coder:7b", "--model", "-m",
        help="Modelo instalado no Ollama com suporte a ferramentas.",
    ),
    max_iterations: int = typer.Option(10, "--max-iter", min=1),
) -> None:
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
        registry = build_default_registry(workspace)
        provider = OllamaProvider(model, tools_schema=registry.schemas())
        use_case = RunAgentSessionUseCase(provider, registry, workspace, max_iterations)
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
