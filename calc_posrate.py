# -*- coding: utf-8 -*-
"""
compute_positive_rate.py
~~~~~~~~~~~~~~~~~~~~~~~~
Compute the *per‑CoT* positive‑routing rate for each student (0 or 1)
from the JSONL logs emitted by *main_router_logging.py*.

For every epoch log found under a given directory (default: ``runs``),
the script prints a small table:

    Epoch 1
       CoT   S0_rate   S1_rate   total
       0     0.67      0.33      12
       1     0.48      0.52      25
       ...

and finally an *all‑epochs* aggregate at the end.

Positive‑rate definition
------------------------
For a given CoT index *c* and student *s* ∈ {0,1}:

    rate(c, s) =  (# times variant for CoT *c* was routed to student *s*)
                  ------------------------------------------------------
                  (# times variant for CoT *c* appeared overall)

Because every variant is always assigned to either student‑0 or student‑1,
``rate(c,0) + rate(c,1) == 1``.

Usage
-----
```bash
python compute_positive_rate.py --log_dir runs
```
"""

import argparse, json, pathlib, re
from collections import defaultdict


###############################################################################
# helpers
###############################################################################

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="runs",
                    help="directory containing router_logs_epoch<k>.jsonl files")
    ap.add_argument("--min_count", type=int, default=1,
                    help="skip CoTs with fewer than this many occurrences")
    return ap.parse_args()


def load_epoch(path):
    """Return list of dicts loaded from one epoch log."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def accumulate(records, counts_pos, counts_tot):
    """Update nested dicts with one epoch's records."""
    for rec in records:
        # variant‑0
        cot0, route0 = rec["cot_idx_0"], rec["route_0"]  # ints
        cot1, route1 = rec["cot_idx_1"], rec["route_1"]
        if route0 == 0:
            counts_pos[cot0][0] += 1
        counts_tot[cot0][0] += 1
        if route1 == 0:
            counts_pos[cot1][1] += 1
        counts_tot[cot1][1] += 1
                

def compute_rates(counts_pos, counts_tot):
    rates = {}
    for cot, pos_by_s in counts_pos.items():
        r0 = pos_by_s[0] / counts_tot[cot][0] if counts_tot[cot][0] else 0.0
        r1 = pos_by_s[1] / counts_tot[cot][1] if counts_tot[cot][1] else 0.0
        rates[cot] = (r0, r1, counts_tot[cot][0])  # total per‑cot
    return dict(sorted(rates.items()))


def print_table(title, rates, min_count=1):
    print(f"\n{title}")
    print(f"{'CoT':>6} {'S0_rate':>8} {'S1_rate':>8} {'total':>6}")
    for cot, (r0, r1, tot) in rates.items():
        if tot < min_count:
            continue
        print(f"{cot:>6d} {r0:8.2f} {r1:8.2f} {tot:6d}")


###############################################################################
# main
###############################################################################

def main():
    args = parse_args()
    log_dir = pathlib.Path(args.log_dir)
    pattern = re.compile(r"router_logs_epoch(\d+)\.jsonl")

    epoch_files = sorted(
        (p for p in log_dir.glob("router_logs_epoch*.jsonl") if pattern.match(p.name)),
        key=lambda p: int(pattern.match(p.name).group(1)),
    )
    if not epoch_files:
        raise SystemExit(f"No router_logs_epoch*.jsonl files found in {log_dir!s}.")

    # global accumulators across all epochs
    global_pos = defaultdict(lambda: [0, 0])
    global_tot = defaultdict(lambda: [0, 0])

    for ep_path in epoch_files[:3]:
        epoch_no = int(pattern.match(ep_path.name).group(1))
        recs = load_epoch(ep_path)

        # per‑epoch accumulators
        pos = defaultdict(lambda: [0, 0])
        tot = defaultdict(lambda: [0, 0])
        accumulate(recs, pos, tot)

        rates = compute_rates(pos, tot)
        print_table(f"Epoch {epoch_no}", rates, args.min_count)

        # update global
        for cot in tot:
            for s in (0, 1):
                global_pos[cot][s] += pos[cot][s]
                global_tot[cot][s] += tot[cot][s]

    # all‑epochs summary
    all_rates = compute_rates(global_pos, global_tot)
    print_table("ALL EPOCHS", all_rates, args.min_count)


if __name__ == "__main__":
    main()
