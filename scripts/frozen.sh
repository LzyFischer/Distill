#!/usr/bin/env bash
set -e

DATASETS=(date)

for TASK in "${DATASETS[@]}"; do
  python main.py \
      --task "$TASK" \
      --model1 mistralai/Mistral-7B-Instruct-v0.3 \
      --model2 google/gemma-7b-it \
      --freeze_s2 true \
      --seed 43 \
      --epochs 10 --bs 4 --eval_bs 8 \
      --lr_s1 5e-6 --lr_s2 2e-4 \
      | tee "logs/ablation2_mistral_${TASK}_43.log"

  python main.py \
      --task "$TASK" \
      --model1 mistralai/Mistral-7B-Instruct-v0.3 \
      --model2 google/gemma-7b-it \
      --freeze_s1 true \
      --seed 43 \
      --epochs 10 --bs 4 --eval_bs 8 \
      --lr_s1 5e-6 --lr_s2 2e-4 \
      | tee "logs/ablation2_gemini_${TASK}_43.log"
done