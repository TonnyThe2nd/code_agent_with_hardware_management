from pathlib import Path

from .list_dir import SCHEMA as LIST_DIR_SCHEMA, list_dir
from .read_file import SCHEMA as READ_FILE_SCHEMA, read_file
from .registry import ToolRegistry
from .run_command import SCHEMA as RUN_COMMAND_SCHEMA, run_command
from .write_file import SCHEMA as WRITE_FILE_SCHEMA, write_file
from .developer_inspection import (RANGE_SCHEMA, SEARCH_SCHEMA, SYMBOL_SCHEMA, STATUS_SCHEMA, DIFF_SCHEMA, DIAGNOSTICS_SCHEMA,
                                   read_file_range, search_code, git_inspection, inspect_diagnostics)


def build_default_registry(
    workspace: Path, allow_tests: bool = False, test_image: str = "code-agent-tests:local",
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("read_file", read_file(workspace), READ_FILE_SCHEMA)
    registry.register("write_file", write_file(workspace), WRITE_FILE_SCHEMA)
    registry.register("list_dir", list_dir(workspace), LIST_DIR_SCHEMA)
    registry.register("run_command", run_command(workspace, allow_tests=allow_tests, test_image=test_image), RUN_COMMAND_SCHEMA)
    registry.register("read_file_range", read_file_range(workspace), RANGE_SCHEMA)
    registry.register("search_code", search_code(workspace), SEARCH_SCHEMA)
    registry.register("find_symbol", search_code(workspace, symbols=True), SYMBOL_SCHEMA)
    registry.register("git_status", git_inspection(workspace), STATUS_SCHEMA)
    registry.register("git_diff", git_inspection(workspace, diff=True), DIFF_SCHEMA)
    registry.register("inspect_diagnostics", inspect_diagnostics(workspace), DIAGNOSTICS_SCHEMA)
    return registry
