from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from .schema import Example, parse_example


def read_jsonl(path: str | Path) -> Iterator[Example]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield parse_example(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc


def deterministic_split(example: Example, validation_fraction: float, seed: int = 17) -> str:
    """Stable split by content, preventing reordered downloads from leaking examples."""
    import hashlib

    key = f"{seed}\0{example.state}\0{example.decisions[0].key}".encode()
    bucket = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def reservoir_shuffle(items: Iterator[Example], size: int, seed: int) -> Iterator[Example]:
    rng = random.Random(seed)
    buffer: list[Example] = []
    for item in items:
        if len(buffer) < size:
            buffer.append(item)
        else:
            index = rng.randrange(size)
            yield buffer[index]
            buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer

