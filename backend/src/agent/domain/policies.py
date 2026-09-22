from .value_objects import FilePath, ToolCall


class SafetyPolicy:
    """A closed command vocabulary avoids relying on shell blacklists."""

    allowed_commands: tuple[tuple[str, ...], ...] = (
        ("python", "--version"), ("git", "--version"),
    )

    def validate(self, call: ToolCall) -> None:
        arguments = dict(call.arguments)
        expected = {
            "read_file": {"path"}, "write_file": {"path", "content"},
            "list_dir": {"path"}, "run_command": {"command"},
        }
        if call.name not in expected:
            raise ValueError("Unknown tool: " + call.name)
        if set(arguments) != expected[call.name]:
            raise ValueError("Invalid tool arguments")
        if "path" in arguments:
            FilePath(arguments["path"])
        if call.name == "run_command":
            if tuple(arguments["command"].split()) not in self.allowed_commands:
                raise ValueError("Command blocked; allowed: python --version, git --version")
