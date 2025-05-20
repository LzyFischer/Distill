#########################################################################
# from ... import batch_call_gemini_api
#### Import code


import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
import pdb

# try:
#     import google.generativeai as genai  # type: ignore
# except ImportError as e:  # pragma: no cover
#     raise SystemExit("pip install google-generativeai  # required to call Gemini") from e

# ------------------ import provided utils ----------------------------------
from utils import (
    get_alphabet_choice,
    get_number_choice,
    get_true_false,
    get_yes_no,
    extract_answer_anli
)
from math_utils import is_math_correct, parse_math_boxed, parse_boxed

###############################################################################
# Gemini configuration                                                        #
###############################################################################

# aio_sem = asyncio.Semaphore(20)

# def configure_gemini(model_name: str = "gemini-pro", temperature: float = 0.8):
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     return genai.GenerativeModel(model_name, generation_config={"temperature": temperature, "max_output_tokens": 1024})

# async def _call(model, prompt: str) -> str:
#     async with aio_sem:
#         out = await model.generate_content_async(prompt)
#         return out.text.strip()

# async def batch_call(prompts: List[str], model_name):
#     return await asyncio.gather(*[_call(model_name, p) for p in prompts])

###############################################################################
# Reasoning-style wrappers                                                    #
###############################################################################

STYLE_WRAP = [
    lambda base: f"{base}\n\nLet's reason step by step, writing each reasoning step clearly before giving the final answer.", # CoT
    lambda base: f"Use code to solve the following problem and print the final answer.\n{base}", # Code
    lambda base: f"First retrieve some relevant facts from your knowledge, then use them to reason to the final answer.\n{base}", # Knowledge
    lambda base: f"Think in a tree of thoughts: outline multiple solution paths and choose the most promising one to derive the answer.\n{base}", # Tree of thoughts
    lambda base: f"Use forward reasoning to propose a candidate answer, then backward reasoning to verify it and provide the final verified answer.\n{base}", # Forward-backward
    lambda base: f"Reason to solve the problem:\n{base}",
]

###############################################################################
# Dataset‑specific prompt builders                                            #
###############################################################################

def _mc_prompt(question: str, choice_lines: str, n_choices: int) -> str:
    letters = ", ".join([chr(65+i) for i in range(n_choices)])
    return (
        f"Given the following question and {n_choices} candidate answers ({letters}), choose the best answer.\n"
        f"Question: {question}\n{choice_lines}\n"
        f"Please conclude with your choice. "
        f"Your response should end with \"The best answer is [LETTER]\" where LETTER is one of {letters}."
    )


dataset_prompt: Dict[str, Any] = {
    "math": lambda s: f"Question: {s['question']}\n\nPlease put your final answer within \\boxed{{}}.",
    "gsm8k": lambda s: f"Question: {s['question']}\n\nPlease put your final answer within \\boxed{{}}.",
    "table_mwp": lambda s: (
        f"Read the following table then answer the question:\n\n{s['table']}\n\nQuestion: {s['question']}\n\nPlease put your final answer within \\boxed{{}}."
    ),
    "commonsense_qa": lambda s: _mc_prompt(
        s["question"],
        "\n".join(f"{l}. {t}" for l, t in zip(s["choices"]["label"], s["choices"]["text"])),
        5,
    ),
    "date": lambda s: _mc_prompt(
        s["question"],
        "\n".join(f"{l}. {t}" for l, t in zip(s["choices"]["label"], s["choices"]["text"])),
        6,
    ),
    "arc_challenge": lambda s: _mc_prompt(
        s["question"],
        "\n".join(f"{l}. {t}" for l, t in zip(s["choices"]["label"], s["choices"]["text"])),
        4,
    ),
    "anli": lambda s: (
        f"Given that \"{s['premise']}\"\nQuestion: {s['hypothesis']} True, False, or Neither?\n\nPlease conclude with your final answer in \n\nAnswer: "
    ),
    "strategy_qa": lambda s: f"Question: Yes or No: {s['question']}\n\nPlease conclude with either \"Yes\" or \"No\".",
}

###############################################################################
# Prediction extraction helpers                                               #
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

###############################################################################
# Gold‑answer normalisation                                                   #
###############################################################################

def gold_norm(dataset: str, sample: Dict[str, Any]):
    if dataset in {"commonsense_qa", "arc_challenge", "date"}:
        return sample["answerKey"].upper()
    if dataset == "anli":
        return sample["label"].lower()
    if dataset == "strategy_qa":
        return "yes" if sample["answer"] else "no"
    if dataset in {"math", "gsm8k"}:
        return sample["answer"]  # latex / numeric string
    if dataset == "table_mwp":
        return sample["answer"]
    return "N/A"

###############################################################################
# Evaluation                                                                  #
###############################################################################

def evaluate_pred(dataset: str, pred: str, gold: str) -> bool:
    if dataset in {"math", "gsm8k", "table_mwp"}:
        return is_math_correct(pred, gold)
    return pred.lower() == gold.lower()

###############################################################################
# Core processing                                                             #
###############################################################################

def build_prompts(sample: Dict[str, Any], dataset: str, n: int) -> List[str]:
    base = dataset_prompt[dataset](sample)
    return [STYLE_WRAP[i % len(STYLE_WRAP)](base) for i in range(n)]


def process_file(path: Path, dataset: str, n_prompts: int, model):
    # pdb.set_trace()
    # print(f"→ {path.relative_to(Path.cwd())}")
    rows = [json.loads(l) for l in path.open()]
    enriched, correct_subset = [], []
    for samp in rows:
        samp["gold_answer"] = gold_norm(dataset, samp)
        prompts = build_prompts(samp, dataset, n_prompts)
        # replies = asyncio.run(batch_call_gemini_api(prompts, model))
        replies = batch_call_gemini_api(prompts, model)
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
                # keep whatever identifiers you find useful
                "id":   samp.get("id") or samp.get("uid") or samp.get("qid") or samp.get("pid"),
                "gold_answer": samp["gold_answer"],
                "prompts":   [prompts[i]  for i in idx_ok],
                "responses": [replies[i]  for i in idx_ok],
                "preds":     [preds[i]    for i in idx_ok],
            })

        enriched.append(samp)
    # overwrite original
    enriched_path = path.with_name(path.stem + ".enriched.jsonl")
    with enriched_path.open("w") as f:
        for obj in enriched:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    # write correct‑only file
    corr = path.with_name(path.stem + ".correct.jsonl")
    with corr.open("w") as f:
        for obj in correct_subset:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"   {len(correct_subset)}/{len(enriched)} examples have ≥1 correct CoT")

###############################################################################
# CLI                                                                         #
###############################################################################

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".data/")
    p.add_argument("--dataset", choices=list(dataset_prompt)+["all"], default="all")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--model", default="pro")
    p.add_argument("--temp", type=float, default=0.8)
    args = p.parse_args()

    # model = configure_gemini(args.model, args.temp)
    targets = [args.dataset] if args.dataset != "all" else list(dataset_prompt)
    for ds in targets:
        dir_path = Path(args.root) / ds
        if not dir_path.exists():
            print(f"! {ds} directory not found – skip")
            continue
        for jsonl in dir_path.rglob("cot_response.jsonl"):
            # if files contains "train/"
            if "train/" in str(jsonl):
                print(f"! {jsonl} – skip")
                continue
            process_file(jsonl, ds, args.n, args.model)

if __name__ == "__main__":
    main()
