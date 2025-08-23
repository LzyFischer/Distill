#!/usr/bin/env bash
set -e
export HF_HOME=$HOME/.cache/huggingface   # (optional) shared cache

DATASETS=(date arc_challenge anli strategy_qa math)

for TASK in "${DATASETS[@]}"; do
  python main.py \
      --task "$TASK" \
      --model1 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --model2 Qwen/Qwen2.5-3B-Instruct \
      --epochs 10 --bs 4 --eval_bs 8 \
      |& tee "logs/ablation1_${TASK}.log"
done