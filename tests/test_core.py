import json

import pytest

from open_jev.calibrate import temperature_scale
from open_jev.formatting import prompt
from open_jev.io import read_jsonl
from open_jev.metrics import accuracy, brier_score, expected_calibration_error
from open_jev.prepare import prepare_mlx
from open_jev.schema import Decision, Example


def test_schema_rejects_invalid_values():
    with pytest.raises(ValueError):
        Decision("x", "question", ("same", "same"))
    decision = Decision("x", "question", ("no", "yes"))
    with pytest.raises(ValueError):
        Example("state", (decision,), (2,))


def test_prompt_is_stable():
    result = prompt(" state ", Decision("risk", "High?", ("no", "yes")))
    assert result.startswith("<|state|>\nstate\n<|decision|>")
    assert "<|option|>1:yes" in result


def test_prepare_expands_decisions(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({
            "state": "s",
            "decisions": [
                {"key": "a", "question": "q", "options": ["n", "y"]},
                {"key": "b", "question": "q", "options": ["x", "z"]},
            ],
            "labels": [1, 0],
        }) + "\n"
    )
    counts = prepare_mlx(str(source), str(tmp_path / "out"), validation_fraction=0)
    assert counts == {"train": 2, "valid": 0}
    rows = [json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    assert [row["completion"] for row in rows] == [" y", " x"]
    assert len(list(read_jsonl(source))) == 1


def test_metrics_and_calibration():
    probabilities = [[0.8, 0.2], [0.1, 0.9]]
    assert accuracy(probabilities, [0, 1]) == 1
    assert brier_score(probabilities, [0, 1]) == pytest.approx(0.05)
    assert expected_calibration_error(probabilities, [0, 1]) == pytest.approx(0.15)
    assert temperature_scale([[4, 0], [0, 4]], [0, 1]) < 1

