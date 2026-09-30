from __future__ import annotations

import argparse
import json

from .prepare import prepare_mlx
from .schema import Decision


def main() -> None:
    parser = argparse.ArgumentParser(prog="open-jev")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="validate and convert canonical JSONL for mlx-lm")
    prep.add_argument("source")
    prep.add_argument("output")
    prep.add_argument("--validation-fraction", type=float, default=0.02)
    predict = commands.add_parser("predict", help="make one schema-constrained prediction")
    predict.add_argument("--model", required=True)
    predict.add_argument("--adapter")
    predict.add_argument("--state", required=True)
    predict.add_argument("--key", required=True)
    predict.add_argument("--question", required=True)
    predict.add_argument("--options", nargs="+", required=True)
    predict.add_argument("--temperature", type=float, default=1.0)
    predict.add_argument("--abstain-below", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_mlx(args.source, args.output, args.validation_fraction)))
    else:
        from dataclasses import asdict

        from .model import JevScorer

        scorer = JevScorer(args.model, args.adapter)
        decision = Decision(args.key, args.question, tuple(args.options))
        result = scorer.predict(
            args.state,
            decision,
            temperature=args.temperature,
            abstain_below=args.abstain_below,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()

