from __future__ import annotations

from .schema import Decision

SPECIAL_TOKENS = ("<|state|>", "<|decision|>", "<|option|>", "<|end|>")


def prompt(state: str, decision: Decision) -> str:
    """Canonical representation shared by training and serving."""
    options = "".join(f"<|option|>{i}:{value}\n" for i, value in enumerate(decision.options))
    return (
        f"<|state|>\n{state.strip()}\n<|decision|>\n"
        f"key:{decision.key}\nquestion:{decision.question.strip()}\n{options}<|end|>"
    )

