#!/usr/bin/env bash
set -e

SEEDS=(0 1 2 3)

for SEED in "${SEEDS[@]}"; do
  python main.py \
      --task date \
      --model1 mistralai/Mistral-7B-Instruct-v0.3 \
      --model2 google/gemma-7b-it \
      --seed "$SEED" \
      --epochs 10 --bs 4 --eval_bs 8 \
      |& tee "logs/${SEED}.log"
done

      # --freeze_s2 true \