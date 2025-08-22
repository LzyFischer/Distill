#!/usr/bin/env python3
"""
filter_train_spurious_cot.py

Second-pass filter: keep only Chain-of-Thought (CoT) answers whose reasoning
is logically valid, judged by an LLM hosted on DeepInfra.

New in this version
───────────────────
* **Time-budget guard** – abort once the wall-clock budget is exceeded;
  partial results are saved and summarised.
* Removed stray ``pdb.set_trace()`` that blocked execution.
* Progress bars show dataset/file names; CLI gets ``--budget`` and
  ``--max_concurrency``.
* **Parallel ``*.invalid.jsonl`` output** collects all rejected CoTs.
* **Judge replies for the rejected CoTs are now stored** (field
  ``"judge_replies"``) so you can see why each response was marked INVALID.

Usage
-----
export DEEPINFRA_TOKEN="…"  # your key
python filter_train_spurious_cot.py \
       --root data \
       --budget 3600 \
       --batch 20 \
       --max_concurrency 8
"""

from __future__ import annotations
import argparse, asyncio, json, os, re, time
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple
from tqdm import tqdm
from openai import AsyncOpenAI

###############################################################################
# DeepInfra / OpenAI client                                                   #
###############################################################################

client = AsyncOpenAI(
    api_key="FlPfq3C7SrrpXWP2t7COkMgY3iFy0u7B",
    base_url="https://api.deepinfra.com/v1/openai",
)

###############################################################################
# Judge prompt                                                                #
###############################################################################

_JUDGE_TEMPLATE = """You are a strict logician.

Below is a problem prompt, a model’s chain-of-thought reasoning, and the final
answer the model produced. Decide whether the reasoning is logically sound and
truly supports the answer.

If it is sound **and** leads to the answer, reply with:
VALID

Otherwise, reply with:
INVALID – <ONE-SENTENCE REASON>

Reply with exactly VALID or INVALID plus **one concise reason**.

---
PROMPT:
{prompt}

MODEL REASONING & ANSWER:
{cot}
---
"""

def build_judge_prompts(prompts: Sequence[str],
                        cots:     Sequence[str]) -> List[str]:
    return [_JUDGE_TEMPLATE.format(prompt=p, cot=c)
            for p, c in zip(prompts, cots)]

_RE_VALID   = re.compile(r"^\s*valid\b",   re.I)
_RE_INVALID = re.compile(r"^\s*invalid\b", re.I)

def is_valid(txt: str) -> bool:
    """Return True ↔ reply starts with VALID; False ↔ starts with INVALID."""
    txt = txt.strip()
    if _RE_INVALID.match(txt):
        return False
    if _RE_VALID.match(txt):
        return True
    # ambiguous → default-to-valid (optimistic)
    return True

###############################################################################
# Async batch call                                                            #
###############################################################################

async def _one_call(prompt: str, model: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": ("Reply VALID if the reasoning is logically sound and "
                             "supports the answer; otherwise INVALID – <reason>. "
                             "Give exactly one short sentence after the tag.")},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        return resp.choices[0].message.content.strip()

async def _batch(prompts: List[str], model: str, max_conc: int) -> List[str]:
    sem = asyncio.Semaphore(max_conc)
    return await asyncio.gather(*[_one_call(p, model, sem) for p in prompts])

def batch_call_llm(prompts: List[str], model: str, max_conc: int) -> List[str]:
    """Blocking wrapper so the rest of the code can stay synchronous."""
    return asyncio.run(_batch(prompts, model, max_conc))

###############################################################################
# Helpers                                                                     #
###############################################################################

def _aligned_list_keys(row: Dict[str, Any], target_len: int) -> List[str]:
    """Names of list fields whose length matches `responses`."""
    return [k for k, v in row.items() if isinstance(v, list) and len(v) == target_len]

###############################################################################
# Per-file processing                                                         #
###############################################################################

def process_file(path: Path, model: str, batch_size: int, max_conc: int,
                 deadline: float) -> Tuple[int, int]:
    """Filter one ``*.correct.jsonl`` file.

    Returns (#kept, #total).  Stops early when *deadline* (epoch seconds) passes.
    Outputs:
    • ``*.filtered.jsonl`` – only logically valid CoTs
    • ``*.invalid.jsonl``  – rejected CoTs + judge explanations
    """
    rows          = [json.loads(l) for l in path.open("r", encoding="utf-8")]
    valid_rows    = []  # type: List[Dict[str, Any]]
    invalid_rows  = []  # type: List[Dict[str, Any]]
    kept = total = 0

    for start in tqdm(range(0, len(rows), batch_size), desc=path.name, leave=False):
        if time.time() > deadline:
            break

        chunk = rows[start:start + batch_size]
        flat_prompts, flat_cots = [], []
        for r in chunk:
            flat_prompts.extend(r["prompts"])
            flat_cots.extend(r["responses"])

        judge_prompts = build_judge_prompts(flat_prompts, flat_cots)
        replies       = batch_call_llm(judge_prompts, model, max_conc)

        rep_iter = iter(replies)
        for r in chunk:
            this_replies = [next(rep_iter) for _ in r["responses"]]
            flags        = [is_valid(rep) for rep in this_replies]
            total       += len(flags)

            idx_keep = [i for i, ok in enumerate(flags) if ok]
            idx_bad  = [i for i, ok in enumerate(flags) if not ok]
            kept     += len(idx_keep)

            if idx_keep:
                out = dict(r)
                for k in _aligned_list_keys(r, len(r["responses"])):
                    out[k] = [r[k][i] for i in idx_keep]
                valid_rows.append(out)

            if idx_bad:
                out = {k: [r[k][i] for i in idx_bad] for k in _aligned_list_keys(r, len(r["responses"]))}
                # Attach judge feedback
                out["judge_replies"] = [this_replies[i] for i in idx_bad]
                invalid_rows.append(out)

    # ── write results ────────────────────────────────────────────────────
    filtered_path = path.with_name(path.stem.replace(".correct", ".filtered") + ".jsonl")
    invalid_path  = path.with_name(path.stem.replace(".correct", ".invalid")  + ".jsonl")

    with filtered_path.open("w", encoding="utf-8") as fh:
        for obj in valid_rows:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with invalid_path.open("w", encoding="utf-8") as fh:
        for obj in invalid_rows:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    pct = kept / total * 100 if total else 0.0
    print(f"   {path.relative_to(path.parents[2])}: kept {kept}/{total} ({pct:.2f} % valid)")
    return kept, total

###############################################################################
# CLI                                                                         #
###############################################################################

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   default="data", help="dataset root folder")
    ap.add_argument("--dataset", default="all", help="'all' or one dataset directory")
    ap.add_argument("--judge_model", default="mistralai/Mistral-7B-Instruct-v0.3", help="LLM for judging")
    ap.add_argument("--batch", type=int, default=20, help="#CoTs per API batch")
    ap.add_argument("--max_concurrency", type=int, default=8, help="max parallel API calls")
    ap.add_argument("--budget", type=int, default=20, help="time budget in seconds")
    args = ap.parse_args()

    if not client.api_key:
        raise SystemExit("Set DEEPINFRA_TOKEN environment variable.")

    start_time = time.time()
    deadline   = start_time + args.budget

    root = Path(args.root).expanduser()
    targets = ([args.dataset] if args.dataset != "all" else sorted(d.name for d in root.iterdir() if d.is_dir()))

    grand_kept = grand_total = 0
    for ds in targets:
        for f in sorted((root / ds).rglob("*.correct.jsonl")):
            if "train" not in f.parts:
                continue
            if time.time() > deadline:
                print("\n⏰ Budget exhausted – stopping early.")
                break
            k, t = process_file(f, args.judge_model, args.batch, args.max_concurrency, deadline)
            grand_kept  += k
            grand_total += t
        else:
            continue  # inner loop finished naturally
        break         # inner loop broke → break outer

    # ── summary ──────────────────────────────────────────────────────────
    pct     = grand_kept / grand_total * 100 if grand_total else 0.0
    elapsed = time.time() - start_time
    print("\n========== SUMMARY ==========")
    print(f"Processed : {grand_total} CoTs")
    print(f"Kept      : {grand_kept} ({pct:.2f} % valid)")
    print(f"Elapsed   : {elapsed/60:.1f} min ({elapsed:.0f} s, budget {args.budget} s)")
    print("=============================")

if __name__ == "__main__":
    main()