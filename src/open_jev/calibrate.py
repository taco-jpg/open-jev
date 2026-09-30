from __future__ import annotations

import math
from collections.abc import Sequence


def temperature_scale(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Fit scalar temperature with dependency-free golden-section search."""
    if not labels or len(logits) != len(labels):
        raise ValueError("non-empty logits and matching labels are required")

    def loss(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        total = 0.0
        for row, label in zip(logits, labels, strict=True):
            scaled = [x / temperature for x in row]
            peak = max(scaled)
            total += math.log(sum(math.exp(x - peak) for x in scaled)) + peak - scaled[label]
        return total / len(labels)

    left, right = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    y1, y2 = loss(x1), loss(x2)
    for _ in range(80):
        if y1 < y2:
            right, x2, y2 = x2, x1, y1
            x1 = right - ratio * (right - left)
            y1 = loss(x1)
        else:
            left, x1, y1 = x1, x2, y2
            x2 = left + ratio * (right - left)
            y2 = loss(x2)
    return math.exp((left + right) / 2)

