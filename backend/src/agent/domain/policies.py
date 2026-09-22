from .value_objects import FilePath, ToolCall


class SafetyPolicy:

    allowed_commands: tuple[tuple[str, ...], ...] = (
        ("python", "--version"), ("git", "--version"),
    )

    def is_allowed(self, call: ToolCall) -> tuple[bool, str | None]:
        if not isinstance(call, ToolCall):
            return False, "Invalid tool call"
        arguments = dict(call.arguments)
        expected: dict[str, set[str]] = {
            "read_file": {"path"}, "write_file": {"path", "content"},
            "list_dir": set(), "run_command": {"command"},
        }
        optional: dict[str, set[str]] = {
            "read_file": {"max_bytes"}, "write_file": set(),
            "list_dir": {"path"}, "run_command": {"cwd"},
        }
        if call.name not in expected:
            return False, "Unknown tool: " + call.name
        if not expected[call.name] <= set(arguments) or set(arguments) - expected[call.name] - optional[call.name]:
            return False, "Invalid tool arguments"
        if any(not isinstance(value, str) for key, value in arguments.items() if key != "max_bytes"):
            return False, "Tool arguments must be strings except max_bytes"
        if "max_bytes" in arguments:
            limit = arguments["max_bytes"]
            if type(limit) is not int or limit < 1:
                return False, "max_bytes must be a positive integer"
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
            if not isinstance(command, str) or tuple(command.split()) not in self.allowed_commands:
                return False, "Command blocked; allowed: python --version, git --version"
        return True, None

    def validate(self, call: ToolCall) -> None:
        """Preserve the exception-based contract used by existing adapters."""
        allowed, reason = self.is_allowed(call)
        if not allowed:
            raise ValueError(reason)
