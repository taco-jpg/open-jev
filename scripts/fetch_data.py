#!/usr/bin/env python3
"""Download permissively hosted classification corpora and normalize to Open Jev JSONL.

The script records upstream dataset names and revisions in the manifest. Review each
dataset card and license before commercial use; this project does not redistribute it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCES: dict[str, tuple[str, str | None, str, str, str]] = {
    # alias: (HF path, config, split, text field, label field)
    "ag_news": ("fancyzhx/ag_news", None, "train", "text", "label"),
    "banking77": ("PolyAI/banking77", None, "train", "text", "label"),
    "boolq": ("google/boolq", None, "train", "passage", "answer"),
    "emotion": ("dair-ai/emotion", "split", "train", "text", "label"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--max-per-source", type=int, default=0, help="0 means all")
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install data dependencies: pip install -e '.[data]'") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"format": 1, "sources": {}}
    with args.output.open("w", encoding="utf-8") as stream:
        for alias in args.sources:
            path, config, split, text_field, label_field = SOURCES[alias]
            dataset = load_dataset(path, config, split=split, streaming=True)
            features = dataset.features
            label_feature = features[label_field]
            names = getattr(label_feature, "names", None)
            if names is None:
                names = ["false", "true"]
            count = 0
            for row in dataset:
                if args.max_per_source and count >= args.max_per_source:
                    break
                state = str(row[text_field])
                question = f"Classify this example for the {alias} task."
                if alias == "boolq":
                    state = f"Passage: {state}\nQuestion: {row['question']}"
                    question = "Is the answer to the question true?"
                label = row[label_field]
                if isinstance(label, bool):
                    label = int(label)
                record = {
                    "state": state,
                    "decisions": [{"key": alias, "question": question, "options": names}],
                    "labels": [int(label)],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            manifest["sources"][alias] = {
                "dataset": path,
                "config": config,
                "split": split,
                "rows": count,
                "fingerprint": getattr(dataset, "_fingerprint", None),
            }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

