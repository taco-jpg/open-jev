from __future__ import annotations

import math
from collections.abc import Sequence


def accuracy(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return sum(max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probabilities, labels)) / len(labels)


def negative_log_likelihood(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return -sum(math.log(max(p[y], 1e-12)) for p, y in zip(probabilities, labels)) / len(labels)


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 15
) -> float:
    if not labels:
        return 0.0
    totals = [0] * bins
    correct = [0] * bins
    confidence = [0.0] * bins
    for probs, label in zip(probabilities, labels, strict=True):
        prediction = max(range(len(probs)), key=probs.__getitem__)
        conf = probs[prediction]
        bucket = min(int(conf * bins), bins - 1)
        totals[bucket] += 1
        correct[bucket] += prediction == label
        confidence[bucket] += conf
    return sum(
        (n / len(labels)) * abs(correct[i] / n - confidence[i] / n)
        for i, n in enumerate(totals)
        if n
    )


def brier_score(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return sum(
        sum((p - float(i == y)) ** 2 for i, p in enumerate(probs))
        for probs, y in zip(probabilities, labels, strict=True)
    ) / len(labels)

