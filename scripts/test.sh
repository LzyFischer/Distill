#!/usr/bin/env bash
# run_sft_all.sh — loop through each dataset folder and train a single‑student SFT

TASKS=(arc_challenge anli commonsense_qa date strategy_qa table_mwp)

for t in "${TASKS[@]}"; do
    # results file
    results_file="results/${t}_results.txt"
    echo "▶️  SFT on $t"
    accelerate launch distill_naive.py\
        --task    "$t" \
        --model   mistralai/Mistral-7B-Instruct-v0.3 \
        --epochs  0 \
        --bs      4 \
        --max_len 1024 \
        --lr      5e-6 > "$results_file" 
done

# --multi_gpu --num_processes 2 \