import typer

from .hardware.presentation.cli import app as hardware_app
from .agent.presentation.cli import app as agent_app


app = typer.Typer(help="Comandos do agente local.")
app.add_typer(hardware_app, name="hardware")
app.add_typer(agent_app, name="agent")


if __name__ == "__main__":
    app()
