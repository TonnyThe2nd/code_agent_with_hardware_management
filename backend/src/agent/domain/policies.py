import shlex

from .value_objects import FilePath, ToolCall


class SafetyPolicy:

    def __init__(self, allow_tests: bool = False) -> None:
        self._allow_tests = allow_tests

    allowed_commands: tuple[tuple[str, ...], ...] = (
        ("python", "--version"), ("git", "--version"),
    )

    def is_allowed(self, call: ToolCall) -> tuple[bool, str | None]:
        if not isinstance(call, ToolCall):
            return False, "Invalid tool call"
        arguments = dict(call.arguments)
        expected: dict[str, set[str]] = {
            "read_file": {"path"}, "write_file": {"path", "content"},
            "list_dir": set(), "run_command": {"command"}, "read_file_range": {"path", "start_line", "end_line"},
            "search_code": {"query"}, "find_symbol": {"symbol"}, "git_status": set(),
            "git_diff": set(), "inspect_diagnostics": set(),
        }
        optional: dict[str, set[str]] = {
            "read_file": {"max_bytes"}, "write_file": set(),
            "list_dir": {"path"}, "run_command": {"cwd"}, "read_file_range": set(),
            "search_code": {"path", "max_results"}, "find_symbol": {"path", "max_results"},
            "git_status": set(), "git_diff": {"path"}, "inspect_diagnostics": {"path"},
        }
        if call.name not in expected:
            return False, "Unknown tool: " + call.name
        if not expected[call.name] <= set(arguments) or set(arguments) - expected[call.name] - optional[call.name]:
            return False, "Invalid tool arguments"
        if any(not isinstance(value, str) for key, value in arguments.items() if key not in {"max_bytes", "start_line", "end_line", "max_results"}):
            return False, "Tool arguments must be strings except numeric limits"
        for numeric in {"max_bytes", "start_line", "end_line", "max_results"} & set(arguments):
            limit = arguments[numeric]
            if type(limit) is not int or limit < 1:
                return False, f"{numeric} must be a positive integer"
        for key in ("path", "cwd"):
            if key not in arguments:
                continue
            try:
                value = arguments[key]
                if not isinstance(value, str):
                    return False, "Path must be a string"
                FilePath(value)
            except ValueError as exc:
                return False, str(exc)
        if call.name == "run_command":
            command = arguments["command"]
            try:
                words = tuple(shlex.split(command))
            except ValueError:
                return False, "Invalid command syntax"
            allowed = self.allowed_commands
            test_command = words[:3] == ("python", "-m", "pytest") and all(
                not part.startswith("-") or part == "-q" for part in words[3:]
            )
            if not isinstance(command, str) or (words not in allowed and not (self._allow_tests and test_command)):
                return False, "Command blocked; allowed: python --version, git --version"
        return True, None

    def validate(self, call: ToolCall) -> None:
        allowed, reason = self.is_allowed(call)
        if not allowed:
            raise ValueError(reason)
