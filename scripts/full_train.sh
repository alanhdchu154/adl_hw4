#!/usr/bin/env bash
# Heavier training when you want accuracy (long runtime; adjust for GPU/MPS).
# Usage: ./scripts/full_train.sh
#
# Optional env: UTID, VENV, SKIP_DATA — same as pareto_quick.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
UTID="${UTID:-ahc982}"
VENV="${VENV:-.venv}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# shellcheck source=/dev/null
source "$VENV/bin/activate"

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  python -m homework.generate_qa build_train_qa
  python -m homework.generate_captions build_train_captions
fi

# Full dataset, no max_steps → controlled by num_train_epochs only.
# Lower batch if OOM; raise num_train_epochs for better fit.
python -m homework.finetune train \
  --output_dir vlm_model \
  --num_train_epochs 0.12 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_workers 2 \
  --learning_rate 3e-4

python -m homework.clip train \
  --output_dir clip_model \
  --num_train_epochs 0.08 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_workers 2 \
  --learning_rate 3e-4

python -m homework.finetune test vlm_model
python -m homework.clip test clip_model

python3 bundle.py homework "$UTID"
python3 -m grader "${UTID}.zip"
