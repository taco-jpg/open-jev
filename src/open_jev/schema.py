from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Decision:
    """One closed-world decision. Options are the only values the model may emit."""

    key: str
    question: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.question:
            raise ValueError("decision key and question must be non-empty")
        if not 2 <= len(self.options) <= 255:
            raise ValueError("a decision requires 2..255 options")
        if len(set(self.options)) != len(self.options) or any(not x for x in self.options):
            raise ValueError("options must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class Example:
    state: str
    decisions: tuple[Decision, ...]
    labels: tuple[int, ...]
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.decisions) != len(self.labels):
            raise ValueError("one label is required per decision")
        if self.weights and len(self.weights) != len(self.labels):
            raise ValueError("weights must be empty or match labels")
        for decision, label in zip(self.decisions, self.labels, strict=True):
            if not 0 <= label < len(decision.options):
                raise ValueError(f"label {label} is outside {decision.key}'s options")


def parse_example(value: dict[str, Any]) -> Example:
    decisions = tuple(
        Decision(str(x["key"]), str(x["question"]), tuple(map(str, x["options"])))
        for x in value["decisions"]
    )
    return Example(
        state=str(value["state"]),
        decisions=decisions,
        labels=tuple(map(int, value["labels"])),
        weights=tuple(map(float, value.get("weights", ()))),
    )


OutputMode = Literal["argmax", "abstain"]

