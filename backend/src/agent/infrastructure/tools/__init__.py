from pathlib import Path

from .list_dir import SCHEMA as LIST_DIR_SCHEMA, list_dir
from .read_file import SCHEMA as READ_FILE_SCHEMA, read_file
from .registry import ToolRegistry
from .run_command import SCHEMA as RUN_COMMAND_SCHEMA, run_command
from .write_file import SCHEMA as WRITE_FILE_SCHEMA, write_file


def build_default_registry(
    workspace: Path, allow_tests: bool = False, test_image: str = "code-agent-tests:local",
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("read_file", read_file(workspace), READ_FILE_SCHEMA)
    registry.register("write_file", write_file(workspace), WRITE_FILE_SCHEMA)
    registry.register("list_dir", list_dir(workspace), LIST_DIR_SCHEMA)
    registry.register("run_command", run_command(workspace, allow_tests=allow_tests, test_image=test_image), RUN_COMMAND_SCHEMA)
    return registry
