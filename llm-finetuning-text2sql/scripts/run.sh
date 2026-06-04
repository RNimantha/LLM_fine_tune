#!/usr/bin/env bash
# Convenience wrapper for the full pipeline. Usage: bash scripts/run.sh
set -euo pipefail

CONFIG="configs/qlora_qwen.yaml"
ADAPTER="outputs/qwen2.5-1.5b-text2sql-qlora"
BASE="Qwen/Qwen2.5-1.5B-Instruct"

echo "==> [1/4] Training (QLoRA SFT)"
python -m src.train --config "$CONFIG"

echo "==> [2/4] Baseline evaluation"
python -m src.evaluate --model "$BASE" --config "$CONFIG" --limit 300

echo "==> [3/4] Fine-tuned evaluation"
python -m src.evaluate --model "$ADAPTER" --config "$CONFIG" --limit 300

echo "==> [4/4] Sample inference"
python -m src.infer --model "$ADAPTER" \
  --schema "CREATE TABLE head (age INTEGER)" \
  --question "How many heads are older than 56?"
