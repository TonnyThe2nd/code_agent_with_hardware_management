"""Convert tool evidence into a compact developer-facing delivery review."""
from dataclasses import dataclass
from typing import Protocol


class ActivitySource(Protocol):
    changed_files: set[str]
    activities: list[object]


@dataclass(frozen=True)
class ImpactReview:
    changed_files: tuple[str, ...]
    validation: tuple[str, ...]
    failures: tuple[str, ...]
    inspected_diff: bool

    @property
    def ready(self) -> bool:
        return bool(self.changed_files) and self.inspected_diff and not self.failures


class ReviewImpactUseCase:
    """Reports only evidence actually produced by tools; it never assumes success."""
    def execute(self, source: ActivitySource) -> ImpactReview:
        activities = source.activities
        names = [getattr(item, "name") for item in activities]
        validation = tuple(getattr(item, "summary") for item in activities
                           if getattr(item, "name") in {"inspect_diagnostics", "run_command"}
                           and not getattr(item, "is_error"))
        failures = tuple(f"{getattr(item, 'name')}: {getattr(item, 'summary')}" for item in activities
                         if getattr(item, "is_error"))
        return ImpactReview(tuple(sorted(source.changed_files)), validation, failures, "git_diff" in names)
