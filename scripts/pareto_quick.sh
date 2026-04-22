#!/usr/bin/env bash
# 80/20: smallest train that exercises the whole pipeline (you tune numbers later).
# Usage: from repo root (homework4_aug_4):
#   chmod +x scripts/pareto_quick.sh
#   ./scripts/pareto_quick.sh
#
# Optional env:
#   UTID=ahc982          # zip filename (default ahc982)
#   VENV=.venv           # Python venv under repo root (default .venv)
#   SKIP_DATA=1         # skip regenerating *_qa_pairs / *_captions json

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
UTID="${UTID:-ahc982}"
VENV="${VENV:-.venv}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -d "$VENV" ]]; then
  echo "Missing venv at $ROOT/$VENV — create with: python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  echo "==> Generate training labels (needs data/train/*_info.json + *_im.jpg)"
  python -m homework.generate_qa build_train_qa
  python -m homework.generate_captions build_train_captions
fi

echo "==> Quick VLM (LoRA) → homework/vlm_model (grader default)"
python -m homework.finetune train \
  --output_dir vlm_model \
  --max_train_samples 80000 \
  --max_steps 400 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_workers 0 \
  --learning_rate 4e-4

echo "==> Quick CLIP → homework/clip_model (grader default)"
python -m homework.clip train \
  --output_dir clip_model \
  --max_train_samples 80000 \
  --max_steps 400 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_workers 0 \
  --learning_rate 4e-4

echo "==> Local smoke test on valid_grader (optional sanity)"
python -m homework.finetune test vlm_model || true
python -m homework.clip test clip_model || true

echo "==> Bundle (zip must stay under ~50MB for Canvas — delete huge logs/checkpoints if needed)"
python3 bundle.py homework "$UTID"

echo "==> Grade zip (same as README)"
python3 -m grader "${UTID}.zip"

echo "Done. Zip: $ROOT/${UTID}.zip"
