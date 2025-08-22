#!/usr/bin/env python3
"""
count_cot_accuracy.py

Traverse a dataset tree, inspect every *.enriched.jsonl that lives in a **train/**
folder, and compute accuracy *from the enriched file alone*.

How correctness is derived
--------------------------
* If a JSON line contains a list field called ``correct_flags``,
  • its length   → #examples for that line  
  • #True values → #correct for that line
* Otherwise each line counts as **one** example.  
  If both ``gold_answer`` and a prediction field (``pred`` / ``answer`` /
  ``prediction``) are present, they are compared case-insensitively to decide
  correctness; absent → assumed wrong.

Output
------
Two Markdown tables are printed:

1. | Dataset | Enriched | Correct | Ratio (%) |
2. A **transposed** view with metrics as rows and datasets as columns.

Usage
-----
python count_cot_accuracy.py --root .data
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


# ── helpers ──────────────────────────────────────────────────────────────
def count_enriched(path: Path) -> Tuple[int, int]:
    """Return (n_examples, n_correct) for one *.enriched.jsonl file."""
    total = correct = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            # ① preferred: explicit list of flags
            if isinstance(ex.get("correct_flags"), list):
                flags = ex["correct_flags"]
                total   += len(flags)
                correct += sum(bool(b) for b in flags)
                continue

            # ② fallback: one example per line
            total += 1
            gold = ex.get("gold_answer")
            pred = (
                ex.get("pred")
                or ex.get("answer")
                or ex.get("prediction")
            )
            if gold is not None and pred is not None:
                correct += str(pred).strip().lower() == str(gold).strip().lower()

    return total, correct


def collect_stats(root: Path) -> List[Tuple[str, int, int]]:
    """
    Return (dataset, n_enriched, n_correct) tuples.

    • Only files whose **immediate parent** is 'train' are counted.
    • *dataset*   = first ancestor directory whose name is *not* 'train'/'test'.
    """
    stats: list[tuple[str, int, int]] = []

    for enriched in root.rglob("*.enriched.jsonl"):
        if enriched.parent.name != "train":        # skip /test/ and others
            continue

        # climb until leaving split level
        p = enriched.parent
        while p.name in {"train", "test"}:
            p = p.parent
        ds_name = p.name                           # e.g. anli, arc_challenge …

        n_total, n_correct = count_enriched(enriched)
        stats.append((ds_name, n_total, n_correct))

    return stats


# ── printing helpers ─────────────────────────────────────────────────────
def aggregate(stats) -> Dict[str, Tuple[int, int]]:
    """Sum counts for each dataset name."""
    agg: Dict[str, list[int]] = defaultdict(lambda: [0, 0])  # {ds: [enr, cor]}
    for ds, n_enr, n_cor in stats:
        agg[ds][0] += n_enr
        agg[ds][1] += n_cor
    return agg


def print_markdown_tables(agg: Dict[str, Tuple[int, int]]) -> None:
    # ── table 1: standard orientation ────────────────────────────────────
    print("\n### Per-dataset accuracy\n")
    print("| Dataset | Enriched | Correct | Ratio (%) |")
    print("|---------|---------:|--------:|----------:|")

    tot_enr = tot_cor = 0
    for ds in sorted(agg):
        n_enr, n_cor = agg[ds]
        ratio = n_cor / n_enr * 100 if n_enr else 0
        print(f"| {ds} | {n_enr} | {n_cor} | {ratio:.2f} |")
        tot_enr += n_enr
        tot_cor += n_cor

    tot_ratio = tot_cor / tot_enr * 100 if tot_enr else 0
    print(f"| **TOTAL** | {tot_enr} | {tot_cor} | {tot_ratio:.2f} |")

    # ── table 2: transposed view ─────────────────────────────────────────
    dsets = sorted(agg)
    print("\n### Transposed view\n")
    header = "| Metric | " + " | ".join(dsets) + " | **TOTAL** |"
    sep    = "|--------" + "|--------:" * (len(dsets) + 1) + "|"
    print(header)
    print(sep)

    # Enriched row
    row_enr = ["Enriched"] + [str(agg[d][0]) for d in dsets] + [str(tot_enr)]
    print("| " + " | ".join(row_enr) + " |")

    # Correct row
    row_cor = ["Correct"] + [str(agg[d][1]) for d in dsets] + [str(tot_cor)]
    print("| " + " | ".join(row_cor) + " |")

    # Ratio row
    row_rat = ["Ratio (%)"] + [
        f"{agg[d][1] / agg[d][0] * 100:.2f}" if agg[d][0] else "0.00"
        for d in dsets
    ] + [f"{tot_ratio:.2f}"]
    print("| " + " | ".join(row_rat) + " |")


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default="data/", help="root folder that holds the datasets"
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"Root directory {root} not found.")

    stats = collect_stats(root)
    if not stats:
        raise SystemExit("No *.enriched.jsonl files found — nothing to count.")

    agg = aggregate(stats)
    print_markdown_tables(agg)


if __name__ == "__main__":
    main()