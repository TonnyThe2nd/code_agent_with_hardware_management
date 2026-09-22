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
            "list_dir": {"path"}, "run_command": {"command"},
        }
        if call.name not in expected:
            return False, "Unknown tool: " + call.name
        if set(arguments) != expected[call.name]:
            return False, "Invalid tool arguments"
        if "path" in arguments:
            try:
                FilePath(arguments["path"])
            except ValueError as exc:
                return False, str(exc)
        if call.name == "run_command":
            if tuple(arguments["command"].split()) not in self.allowed_commands:
                return False, "Command blocked; allowed: python --version, git --version"
        return True, None

    def validate(self, call: ToolCall) -> None:
        """Preserve the exception-based contract used by existing adapters."""
        allowed, reason = self.is_allowed(call)
        if not allowed:
            raise ValueError(reason)
