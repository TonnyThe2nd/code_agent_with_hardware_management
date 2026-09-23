class NoToolActivityError(RuntimeError):
    """Retrying is safe only when no tool was invoked in the session."""
