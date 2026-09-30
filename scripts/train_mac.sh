#!/usr/bin/env bash
set -euo pipefail

# Tuned for an M3 Ultra with abundant unified memory. Override every value via env.
MODEL="${MODEL:-mlx-community/Qwen3-8B-4bit}"
DATA="${DATA:-data/mlx}"
ADAPTER="${ADAPTER:-artifacts/adapter}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ITERS="${ITERS:-4000}"
LAYERS="${LAYERS:-32}"

mkdir -p "$ADAPTER"
export TOKENIZERS_PARALLELISM=true
# The wired max-performance power mode avoids long-run throttling on Mac Studio.
python -m mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --adapter-path "$ADAPTER" \
  --batch-size "$BATCH_SIZE" \
  --num-layers "$LAYERS" \
  --iters "$ITERS" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --save-every 200 \
  --grad-checkpoint

