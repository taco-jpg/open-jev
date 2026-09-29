# Open Jev

An Apple-Silicon-first recipe for **fast, calibrated, closed-world decisions**: text state in,
typed probabilities out. It follows the useful public interface described in TypeSafe AI's
[System One / Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
but is an independent implementation and is not affiliated with TypeSafe AI.

> [!IMPORTANT]
> Jev's architecture, training corpus, RLCD algorithm, weights, and evaluation suite are not
> public. No honest implementation can promise “90% of Jev” from the announcement alone.
> This repository therefore makes **no unsupported performance promise**. It provides a
> reproducible benchmark gate (accuracy, calibration, latency, throughput) so that a 90%
> target can be measured against API results you collect on your own tasks.

## What is implemented

* **Impossible-to-hallucinate output values.** Inference scores every schema option and can
  only return one of those options. It does not generate then parse JSON.
* **Probabilities and abstention.** Every result contains a normalized distribution,
  confidence, and an optional calibrated abstention decision.
* **Parallel option scoring.** All candidate continuations are scored in one MLX forward pass.
* **Native MLX fine-tuning.** LoRA/Q-LoRA through `mlx-lm`, unified-memory-friendly streaming
  data preparation, gradient checkpointing, and a high-memory M3 Ultra profile.
* **Leak-resistant data plumbing.** Canonical multi-decision JSONL, validation, content-hash
  splitting, source manifests, and four starter public datasets.
* **Calibration and metrics.** Dependency-free temperature fitting, NLL, multiclass Brier
  score, accuracy, and expected calibration error (ECE).

This is a strong distillation/fine-tuning baseline, not a clone of a proprietary model. Unlike
Jev's claimed architecture, its backbone remains autoregressive internally; output choices are
evaluated in parallel and exposed through a non-generative, type-safe API.

## M3 Ultra quick start

Use Python 3.11 or 3.12 on macOS 15+, install Xcode command-line tools, and put the Mac Studio
in **High Power** mode for a long run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[data,dev]'

# Smoke test the complete data path without a large download.
open-jev prepare examples/tiny.jsonl data/smoke --validation-fraction 0

# Download and normalize starter corpora. Review their dataset cards/licenses first.
python scripts/fetch_data.py data/raw/starter.jsonl
open-jev prepare data/raw/starter.jsonl data/mlx

# Load the 512 GB M3 Ultra defaults and train. Start small, then scale batch size.
set -a; source configs/m3-ultra.env; set +a
bash scripts/train_mac.sh
```

The default 4-bit 8B backbone is intentionally conservative. On a 512 GB machine, compare an
8B full-precision or 32B 4-bit backbone after the pipeline is proven. Bigger is not necessarily
better for latency. Increase `BATCH_SIZE` until peak memory and step time stop improving; MLX
uses unified memory, so leave ample headroom for macOS and dataset caches. Keep Activity Monitor
memory pressure green. Do not enable speculative settings before establishing a baseline.

## Data format

Each line may contain many independent decisions over the same state:

```json
{
  "state": "free-form program or business state",
  "decisions": [
    {"key": "route", "question": "Where should this go?", "options": ["sales", "support"]},
    {"key": "urgent", "question": "Is it urgent?", "options": ["no", "yes"]}
  ],
  "labels": [1, 0],
  "weights": [1.0, 0.5]
}
```

There must be 2–255 unique options and exactly one integer label per decision. For quality:

1. Build examples from real workflow traces with time-based holdouts.
2. Decompose workflows into narrow decisions; preserve difficult and abstention-worthy cases.
3. Deduplicate before splitting. Never train on the benchmark.
4. Add teacher distributions when available, but retain human/verifiable gold labels.
5. Balance option positions and paraphrase questions so the model cannot exploit formatting.
6. Track license, revision, provenance, PII removal, and contamination for every source.

The included downloader is only a pipeline check. Generic classification datasets will not
produce a production-quality business decision model. Your domain data is the dominant input.

## Inference

```bash
open-jev predict \
  --model mlx-community/Qwen3-8B-4bit \
  --adapter artifacts/qwen3-8b-open-jev \
  --state 'Usage fell 62%; renewal is in 14 days.' \
  --key churn --question 'Churn risk?' --options low medium high \
  --temperature 1.18 --abstain-below 0.65
```

The result is always structurally valid:

```json
{"value":"high","probabilities":{"low":0.02,"medium":0.13,"high":0.85},"confidence":0.85,"abstained":false}
```

Fit temperature **once on untouched validation logits**, store it beside the adapter, and reuse
it at inference. A single global temperature is a baseline; high-stakes deployments should test
per-domain calibration and conformal risk control. An abstained result still carries the best
schema value so callers can log/debug it, but callers must branch on `abstained`.

## A defensible “90%” acceptance gate

Export an evaluation set and compare this model with Jev on the exact same schema and inputs.
Do not reduce quality to one number. A candidate passes only if, with bootstrap confidence
intervals, it achieves all of the agreed gates, for example:

| Dimension | Suggested gate relative to Jev |
|---|---:|
| accuracy / macro-F1 | at least 90% |
| Brier score and NLL | no more than 110% (lower is better) |
| ECE | no more than 2 percentage points worse |
| p50 and p95 latency at the same batch size | no more than 110% |
| sustained decisions/second | at least 90% |
| schema validity | 100% (guaranteed here) |

Report task-level results and worst slices, not only an average. Measure after MLX warm-up, with
the same input lengths/cardinalities and network boundary. “90% as good” is otherwise undefined.

## Suggested training progression

1. **Smoke:** tiny data, 20 steps, verify loss and checkpoint reload.
2. **SFT:** LoRA on gold domain labels; choose checkpoint on validation NLL, not train loss.
3. **Distill:** add soft targets from multiple strong teachers and disagreement examples. The
   current mlx-lm completion route uses hard labels; soft-label training is the next extension.
4. **Calibrate:** fit temperature on a second, untouched calibration split.
5. **Benchmark:** frozen test set, latency harness, slice metrics, bootstrap intervals.
6. **Scale:** compare 8B/14B/32B and LoRA rank/full fine-tuning. Select the smallest model on
   the accuracy-calibration-latency Pareto frontier.

## Limitations

Candidate likelihood is not identical to a purpose-built multi-head classifier and can be
sensitive to option wording. Very high cardinality increases memory linearly; chunk options if
needed. The public starter labels are hard targets rather than calibrated distributions. This
code has not been safety-certified; use human review and explicit fail-closed policy for medical,
financial, employment, infrastructure, or other consequential decisions.

