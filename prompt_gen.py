"""Single‑prompt CoT pipeline.

This variant keeps everything the same as the multi‑style version but uses **one
powerful wrapper prompt** for every sample:

    "Read the problem carefully. Break it into small sub‑problems. Solve each
     sub‑problem step by step. Double‑check the answers."

Usage is identical to the previous script except it ignores --n because each
sample always gets exactly one prompt.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List
import asyncio
import pdb

# -----------------------------------------------------------------------------
#  External helpers (same as before)
# -----------------------------------------------------------------------------
from utils import (
    get_alphabet_choice, get_number_choice, get_true_false, get_yes_no  # ← existing sync call returning list[str]
)
from math_utils import is_math_correct, parse_math_boxed, parse_boxed

def batch_call_gemini_api(prompts: List[str], model_name: str) -> List[str]:
    """Call the Gemini API in a batch.

    Args:
        prompts: List of prompts to send to the Gemini API.
        model_name: Name of the model to use.

    Returns:
        List of responses from the Gemini API.
    """
    # This is a placeholder for the actual API call. Replace with your implementation.
    return ["Placeholder response"] * len(prompts)

###############################################################################
# Prompt builders (dataset‑specific core + single wrapper)                    #
###############################################################################

def _mc_prompt(question: str, choice_lines: str, n_choices: int) -> str:
    letters = ", ".join(chr(65+i) for i in range(n_choices))
    return (
        f"Given the following question and {n_choices} candidate answers ({letters}), choose the best answer.\n"
        f"Question: {question}\n{choice_lines}\n"
        f"Please conclude with your choice. Your response should end with \"The best answer is [LETTER]\" where LETTER is one of {letters}."
    )

dataset_prompt: Dict[str, Any] = {
    "math":           lambda s: f"Question: {s['question']}\n\nPlease put your final answer within \\boxed{{}}.",
    "gsm8k":          lambda s: f"Question: {s['question']}\n\nPlease put your final answer within \\boxed{{}}.",
    "table_mwp":      lambda s: f"Read the table then answer the question:\n\n{s['table']}\n\nQuestion: {s['question']}\n\nPlease put your final answer within \\boxed{{}}.",
    "commonsense_qa": lambda s: _mc_prompt(
        s["question"],
        "\n".join(f"{l}. {t}" for l, t in zip(s["choices"]["label"], s["choices"]["text"])),
        5,
    ),
    "date":           lambda s: _mc_prompt(
        s["question"],
        "\n".join(f"{l}. {t}" for l, t in zip(s["choices"]["label"], s["choices"]["text"])),
        6,
    ),
    "arc_challenge":  lambda s: _mc_prompt(
        s["question"],
        "\n".join(f"{l}. {t}" for l, t in zip(s["choices"]["label"], s["choices"]["text"])),
        4,
    ),
    "anli":           lambda s: f"Given that \"{s['premise']}\"\nQuestion: {s['hypothesis']} True, False, or Neither?",
    "strategy_qa":    lambda s: f"Question: Yes or No: {s['question']}",
}

# single powerful wrapper
# POWER_WRAPPER = (
#     "Read the problem carefully. Break it into small sub‑problems. Solve each "
#     "sub‑problem step by step. Double‑check the answers.\n\n{}"
# )

POWER_WRAPPER = (
    "{}"
    )

###############################################################################
# Parse predictions & gold                                                    #
###############################################################################

TFN_RE = re.compile(r"(true|false|neither)", re.I)


def extract_pred(dataset: str, text: str):
    if not text:
        return "N/A"
    if dataset in {"commonsense_qa", "arc_challenge", "date",}:
        return get_alphabet_choice(text).upper()
    if dataset == "anli":
        # text = remove_backward_answer(text)
        return extract_answer_anli(text)
    if dataset == "strategy_qa":
        return get_yes_no(text)
    if dataset in {"math", "gsm8k", "table_mwp"}:
        return parse_math_boxed(text)
    return "N/A"


def gold_norm(dataset: str, samp: Dict[str, Any]):
    if dataset in {"commonsense_qa", "arc_challenge", "date"}:
        return samp["answerKey"].upper()
    if dataset == "anli":
        return samp["label"].lower()
    if dataset == "strategy_qa":
        return "yes" if samp["answer"] else "no"
    if dataset in {"math", "gsm8k"}:
        return samp["answer"]
    if dataset == "table_mwp":
        return samp["answer"]
    return "N/A"


def evaluate_pred(dataset: str, pred: str, gold: str) -> bool:
    if dataset in {"math", "gsm8k", "table_mwp"}:
        return is_math_correct(pred, gold)
    return pred == gold

###############################################################################
# Core processing                                                             #
###############################################################################

def make_prompt(samp: Dict[str, Any], dataset: str) -> str:
    base = dataset_prompt[dataset](samp)
    return POWER_WRAPPER.format(base)


def process_file(path: Path, dataset: str, model_name: str):
    rows = [json.loads(l) for l in path.open()]
    enriched, correct_only = [], []

    for samp in rows:
        samp["gold_answer"] = gold_norm(dataset, samp)
        prompt  = make_prompt(samp, dataset)
        reply   = asyncio.run(batch_call_gemini_api(prompt, model_name))
        pred    = extract_pred(dataset, reply)
        if isinstance(pred, float):
            pdb.set_trace()
        correct = evaluate_pred(dataset, pred, samp["gold_answer"])

        samp.update({"prompt": prompt, "response": reply, "pred": pred, "correct": correct})
        enriched.append(samp)

        if correct:
            correct_only.append({
                "id": samp.get("id") or samp.get("uid") or samp.get("qid") or samp.get("pid"),
                "gold_answer": samp["gold_answer"],
                "prompt": prompt,
                "response": reply,
                "pred": pred,
            })

    # write output files next to original
    enriched_p = path.with_name(path.stem + ".enriched.jsonl")
    with enriched_p.open("w") as f:
        for obj in enriched:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    corr_p = path.with_name(path.stem + ".correct.jsonl")
    # with corr_p.open("w") as f:
    #     for obj in correct_only:
    #         f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"   {len(correct_only)}/{len(enriched)} correct → {corr_p.name}")

###############################################################################
# CLI                                                                         #
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".data/")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dataset", choices=list(dataset_prompt)+["all"], default="all")
    ap.add_argument("--model", default="flash", help="flash or pro")
    args = ap.parse_args()

    targets = [args.dataset] if args.dataset != "all" else list(dataset_prompt)
    for ds in targets:
        dir_path = Path(args.root) / ds
        if not dir_path.exists():
            print(f"! {ds} directory not found – skip")
            continue
        # if files contains "train/"
        for jsonl in dir_path.rglob("cot_response.jsonl"):
            if "train/" in str(jsonl):
                print(f"! {jsonl} – skip")
                continue
            process_file(jsonl, ds, args.model)

if __name__ == "__main__":
    main()
