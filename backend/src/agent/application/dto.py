from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRunResult:
    session_id: str
    response: str
    iterations: int
    completed: bool
