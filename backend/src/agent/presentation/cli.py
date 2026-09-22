from pathlib import Path

import httpx
import typer

from ..application.use_cases import RunAgentSessionUseCase
from ..domain.policies import SafetyPolicy
from ..infrastructure.llm.ollama_provider import OllamaProvider
from ..infrastructure.tools.list_dir import ListDir
from ..infrastructure.tools.read_file import ReadFile
from ..infrastructure.tools.registry import build_default_registry
from ..infrastructure.tools.run_command import RunCommand
from ..infrastructure.tools.write_file import WriteFile


app = typer.Typer(help="Execute o agente local com Ollama.")


@app.command("run")
def run(
    prompt: str = typer.Argument(...),
    workspace: Path = typer.Option(Path("."), exists=True, file_okay=False, resolve_path=True),
    model: str = typer.Option(..., help="Modelo instalado no Ollama com suporte a ferramentas."),
    base_url: str = typer.Option("http://localhost:11434"),
    max_iterations: int = typer.Option(10, min=1),
    python_executable: Path | None = typer.Option(None, exists=True, dir_okay=False),
    git_executable: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    executables = {name: path for name, path in (
        ("python", python_executable), ("git", git_executable),
    ) if path is not None}
    try:
        policy = SafetyPolicy()
        registry = build_default_registry(
            ReadFile(workspace), WriteFile(workspace), ListDir(workspace),
            RunCommand(workspace, executables, policy), policy,
        )
        with httpx.Client(base_url=base_url, timeout=120.0) as client:
            provider = OllamaProvider(client, model)
            use_case = RunAgentSessionUseCase(provider, registry, workspace, max_iterations)
            result = use_case.execute(prompt)
    except (httpx.HTTPError, ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        typer.echo(f"Erro ao executar agente: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.final_message is None:
        typer.echo("Limite de iteracoes atingido sem resposta final.", err=True)
        raise typer.Exit(code=2)
    typer.echo(result.final_message)
