from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .formatting import prompt
from .schema import Decision


@dataclass(frozen=True, slots=True)
class Prediction:
    value: str
    probabilities: dict[str, float]
    confidence: float
    abstained: bool = False


class JevScorer:
    """Closed-world likelihood scorer using an MLX language-model backbone.

    Only caller-provided options can be returned: text generation is deliberately
    not part of this API. Candidate continuations are evaluated in one MLX batch.
    """

    def __init__(self, model_path: str | Path, adapter_path: str | Path | None = None):
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError("Install the project on Apple Silicon: pip install -e .") from exc
        self.mx = mx
        self.model, self.tokenizer = load(
            str(model_path), adapter_path=str(adapter_path) if adapter_path else None
        )

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def probabilities(self, state: str, decision: Decision, temperature: float = 1.0) -> list[float]:
        mx = self.mx
        prefix = self._tokenize(prompt(state, decision) + "\nanswer:")
        suffixes = [self._tokenize(" " + value) for value in decision.options]
        lengths = [len(prefix) + len(suffix) for suffix in suffixes]
        width = max(lengths)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        rows = [prefix + suffix + [pad] * (width - length) for suffix, length in zip(suffixes, lengths)]
        logits = self.model(mx.array(rows))
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        scores = []
        for row, suffix in enumerate(suffixes):
            positions = mx.arange(len(prefix) - 1, len(prefix) + len(suffix) - 1)
            token_ids = mx.array(suffix)
            selected = log_probs[row, positions, token_ids]
            # Length normalization avoids systematically preferring short labels.
            scores.append(mx.mean(selected))
        scores = mx.stack(scores) / max(temperature, 1e-4)
        probs = mx.softmax(scores)
        return [float(x) for x in probs.tolist()]

    def predict(
        self, state: str, decision: Decision, *, temperature: float = 1.0, abstain_below: float = 0.0
    ) -> Prediction:
        probabilities = self.probabilities(state, decision, temperature)
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence = probabilities[index]
        abstained = confidence < abstain_below
        return Prediction(
            value=decision.options[index],
            probabilities=dict(zip(decision.options, probabilities, strict=True)),
            confidence=confidence,
            abstained=abstained,
        )
