import json
import typer

from ..application.use_cases import DetectHardwareUseCase
from ..infrastructure.composite_probe import CompositeProbe


app = typer.Typer(help="Comandos do agente local")


@app.callback(invoke_without_command=True)
def mostrar_hardware(json_output: bool = typer.Option(False, "--json")) -> None:
    """detecta e exibe o hardware da máquina."""
    probe = CompositeProbe()
    use_case = DetectHardwareUseCase(probe=probe)
    dto = use_case.execute()

    if json_output:
        typer.echo(json.dumps(dto.to_dict(), indent=2))
    else:
        typer.echo("─── Hardware Detectado ───")
        typer.echo(f"  RAM total:    {dto.total_ram_gb:.1f} GB")
        typer.echo(f"  CPU:          {dto.cpu_cores} cores")
        typer.echo(f"  Backend:      {dto.backend}")
        typer.echo(f"  Dispositivo:  {dto.device_name}")
        if dto.vram_gb:
            typer.echo(f"  VRAM:         {dto.vram_gb:.1f} GB")
        typer.echo(f"  Orçamento:    {dto.budget_gb:.1f} GB ({dto.budget_source})")
