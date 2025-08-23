#!/usr/bin/env bash
set -e
export HF_HOME=$HOME/.cache/huggingface   # (optional) shared cache

DATASETS=(anli)


# for TASK in "${DATASETS[@]}"; do
#   python main.py \
#       --task "$TASK" \
#       --model1 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --model2 Qwen/Qwen2.5-3B-Instruct \
#       --epochs 10 --bs 4 --eval_bs 8 \
#       |& tee "logs/ablation1_${TASK}.log"
# done


# for TASK in "${DATASETS[@]}"; do
#   python distill_naive.py \
#       --task "$TASK" \
#       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --epochs 10 --bs 4 --eval_bs 8 \
#       |& tee "logs/ablation1_TinyLlama_${TASK}.log"
# done


      # --model2 Qwen/Qwen2.5-3B-Instruct \
for TASK in "${DATASETS[@]}"; do
  python distill_naive.py \
      --task "$TASK" \
      --model Qwen/Qwen2.5-3B-Instruct \
      --epochs 10 --bs 4 --eval_bs 8 \
      | tee "logs/ablation1_Qwen_${TASK}.log"
done