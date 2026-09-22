from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRunResult:
    session_id: str
    final_message: str | None
    iterations: int
    messages_count: int

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "session_id": self.session_id,
            "final_message": self.final_message,
            "iterations": self.iterations,
            "messages_count": self.messages_count,
        }
