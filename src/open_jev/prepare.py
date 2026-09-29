from __future__ import annotations

import json
from pathlib import Path

from .formatting import prompt
from .io import deterministic_split, read_jsonl


def prepare_mlx(source: str, output: str, validation_fraction: float = 0.02) -> dict[str, int]:
    """Expand multi-decision records into mlx-lm completion records."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    streams = {
        name: (root / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "valid")
    }
    counts = {"train": 0, "valid": 0}
    try:
        for example in read_jsonl(source):
            split = "valid" if deterministic_split(example, validation_fraction) == "validation" else "train"
            for decision, label in zip(example.decisions, example.labels, strict=True):
                record = {
                    "prompt": prompt(example.state, decision) + "\nanswer:",
                    "completion": " " + decision.options[label],
                }
                streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    return counts

