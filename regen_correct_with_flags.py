import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List

# ------------------ import existing utilities ------------------
from utils import (
    get_alphabet_choice,
    get_yes_no,
    extract_answer_anli
)
from math_utils import is_math_correct, parse_math_boxed, parse_boxed
import pdb

# ------------------ regex helpers ------------------
TFN_RE = re.compile(r"(true|false|neither)", re.I)

# ------------------ gold normalization ------------------
def gold_norm(dataset: str, sample: Dict[str, Any]):
    if dataset in {"commonsense_qa", "arc_challenge", "date"}:
        return sample["answerKey"].upper()
    if dataset == "anli":
        return sample["label"].lower()
    if dataset == "strategy_qa":
        return "yes" if sample["answer"] else "no"
    if dataset in {"math", "gsm8k"}:
        return sample["answer"]
    if dataset == "table_mwp":
        return sample["answer"]
    return "N/A"

# ------------------ prediction extraction ------------------
def extract_pred(dataset: str, text: str):
    if not text:
        return "N/A"
    if dataset in {"commonsense_qa", "arc_challenge", "date"}:
        return get_alphabet_choice(text).upper()
    if dataset == "anli":
        m = extract_answer_anli(text)
        return m[-1].lower() if m else "N/A"
    if dataset == "strategy_qa":
        return get_yes_no(text)
    if dataset in {"math", "gsm8k", "table_mwp"}:
        return parse_math_boxed(text)
    return "N/A"

# ------------------ prediction evaluation ------------------
def evaluate_pred(dataset: str, pred: str, gold: str) -> bool:
    if dataset in {"math", "gsm8k", "table_mwp"}:
        return is_math_correct(pred, gold)
    if dataset == "anli":
        if pred == 'false' and gold == 'contradiction':
            return True
        elif pred == 'true' and gold == 'entailment':
            return True
        elif pred == 'neither' and gold == 'neutral':
            return True
    return pred.lower() == gold.lower()

# ------------------ processing function ------------------
def regenerate_flags_and_correct(enriched_path: Path, dataset: str):
    with enriched_path.open() as f:
        rows = [json.loads(l) for l in f]

    enriched, correct_subset = [], []

    for samp in rows:
        samp["gold_answer"] = gold_norm(dataset, samp)

        preds = samp.get("preds", [])
        replies = samp.get("responses", [])
        prompts = samp.get("prompts", [])

        if len(preds) != len(replies):
            preds = [extract_pred(dataset, r) for r in replies]
        flags = [evaluate_pred(dataset, p, samp["gold_answer"]) for p in preds]

        samp.update({
            "prompts": prompts,
            "responses": replies,
            "preds": preds,
            "correct_flags": flags,
        })

        if any(flags):
            idx_ok = [i for i, ok in enumerate(flags) if ok]
            correct_subset.append({
                "id": samp.get("id") or samp.get("uid") or samp.get("qid") or samp.get("pid"),
                "gold_answer": samp["gold_answer"],
                "prompts": [prompts[i] for i in idx_ok],
                "responses": [replies[i] for i in idx_ok],
                "preds": [preds[i] for i in idx_ok],
            })

        enriched.append(samp)

    # overwrite enriched file
    with enriched_path.open("w") as f:
        for obj in enriched:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # write correct-only file
    corr_path = enriched_path.with_name(enriched_path.stem.replace(".enriched", ".correct") + ".jsonl")
    with corr_path.open("w") as f:
        for obj in correct_subset:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✓ {enriched_path.name} → {corr_path.name} ({len(correct_subset)}/{len(enriched)} correct samples)")

# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/", help="Root directory containing enriched files")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or 'all'")
    args = parser.parse_args()

    root_path = Path(args.root)
    all_datasets = {"math", "gsm8k", "table_mwp", "commonsense_qa", "arc_challenge", "date", "anli", "strategy_qa"}

    targets = [args.dataset] if args.dataset != "all" else list(all_datasets)

    for dataset in targets:
        for enriched_file in (root_path / dataset).rglob("*.enriched.jsonl"):
            regenerate_flags_and_correct(enriched_file, dataset)


if __name__ == "__main__":
    main()