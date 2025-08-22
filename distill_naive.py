# main.py  ── single‑student SFT (simple supervised fine‑tuning)

import os, json, torch, copy, re
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from math_utils import is_math_correct, parse_math_boxed, parse_boxed
import wandb
import pdb
from utils import get_number_choice, get_alphabet_choice, get_true_false, get_yes_no, extract_answer_anli
import argparse
import pathlib
import regex




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",    required=True,
                   choices=["gsm8k","math","arc_challenge","anli",
                            "commonsense_qa","date","strategy_qa","table_mwp"])
    p.add_argument("--model",   default="google/gemma-7b-it")
    p.add_argument("--epochs",  type=int, default=1)
    p.add_argument("--bs",      type=int, default=4)
    p.add_argument("--eval_bs",     type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lr",      type=float, default=2e-4)
    return p.parse_args()

args = parse_args()

root       = pathlib.Path("data") / args.task
DATA_PATH  = root / "train" / "train_cot_distill.jsonl"
VAL_PATH   = root / "train" / "test_cot_distill.jsonl"
TEST_PATH  = root / "test"  / "cot_response.enriched.jsonl"

BASE_MODEL = args.model
BATCH_SIZE = args.bs
GEN_BS     = args.eval_bs
EPOCHS     = args.epochs
MAX_LEN    = args.max_len
LR         = args.lr
PROJECT    = f"sft‑{args.task}"
RUN_NAME   = f"{BASE_MODEL.split('/')[-1]}_{args.task}"

# ───────────────── CONFIG ─────────────────

ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
pg  = InitProcessGroupKwargs(timeout=timedelta(hours=1))
acc = Accelerator(kwargs_handlers=[ddp, pg])
DEVICE = acc.device
print("Rank", acc.local_process_index, "| device", DEVICE)

if acc.is_main_process:
    wandb.init(
        project="multi-student-distillation",
        name="distill_naive",
        config={
            "base_model": BASE_MODEL,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_len": MAX_LEN,
            "lr": LR,
            "data_path": DATA_PATH
        },
    )

# ───────────────── DATA ─────────────────
def load_jsonl(path, limit=None):
    with open(path) as f:
        data = [json.loads(l) for l in f]
    return data[:limit] if limit else data

class CotDataset(Dataset):
    """Return (input_ids, attn_mask, labels) where labels mask the instruction."""
    def __init__(self, data, tok):
        self.data, self.tok = data, tok
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d      = self.data[idx]
        ins    = d["prompt"]
        cot    = d["messages"][1]["content"]          # teacher CoT (+ answer)

        prompt = self.tok(ins, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=MAX_LEN)
        full   = self.tok(ins + cot, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=MAX_LEN)

        L      = prompt["attention_mask"].sum()       # prompt length
        ids    = full["input_ids"].squeeze(0)
        attn   = full["attention_mask"].squeeze(0)
        lbl    = ids.clone()
        lbl[:L]      = -100                           # ignore instruction tokens
        lbl[attn==0] = -100                           # ignore padding
        return ids, attn, lbl

def collate(batch):
    ids, attn, lbl = zip(*batch)
    return dict(
        input_ids     = torch.stack(ids),
        attention_mask= torch.stack(attn),
        labels        = torch.stack(lbl)
    )

# ───────────────── TOKENIZER & MODEL ─────────────────
print("Loading model …")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)
backbone = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    quantization_config=bnb,
    # attn_implementation='flash_attention_2',
    torch_dtype=torch.float16, trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM
)
student = get_peft_model(backbone, lora_cfg)
student.print_trainable_parameters()

# ───────────────── DATA LOADERS ─────────────────
train_ds = CotDataset(load_jsonl(DATA_PATH), tok)   # small subset; remove limit for full
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, collate_fn=collate)

# ───────────────── PREPARE FOR ACCELERATE ─────────────────
# opt = torch.optim.AdamW(student.parameters(), lr=LR)
opt = bitsandbytes.optim.PagedAdamW32bit(student.parameters(), lr=LR)   # drop‑in replacement
student, opt, train_dl = acc.prepare(student, opt, train_dl)

# ───────────────── TRAIN ─────────────────
student.train()
for ep in range(1, EPOCHS + 1):
    loop = tqdm(train_dl, disable=not acc.is_main_process,
                desc=f"Epoch {ep}", dynamic_ncols=True)
    for batch in loop:
        out  = student(**batch)
        loss = out.loss
        acc.backward(loss)
        opt.step(); opt.zero_grad(set_to_none=True)
        loop.set_postfix(loss=f"{loss.item():.4f}")
        if acc.is_main_process:
            wandb.log({"loss": loss.item()})
            
        

print("✓ Training done")
# -------------------------------------------------------------------------
_GOLD_MAP = {
    "entailment":    "true",
    "neutral":       "neither",
    "contradiction": "false",
}

def convert_gold(label: str) -> str:
    try:
        return _GOLD_MAP[label.strip().lower()]
    except KeyError as e:
        raise ValueError(f"Unknown gold label: {label!r}") from e

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

def evaluate_pred(dataset: str, pred: str, gold: str) -> bool:
    if dataset in {"math", "gsm8k", "table_mwp"}:
        return is_math_correct(pred, gold)
    return pred == gold

class PromptDataset(Dataset):
    """
    Each line in the JSONL file must contain
      {
        "prompt": "<already-formatted prompt string>",
        "gold_answer": "<gold label/answer text>"
      }
    """
    def __init__(self, path):
        self.data = load_jsonl(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["prompt"], d["gold_answer"]


# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loader(
    model,
    tokenizer,
    dataloader: DataLoader,
    dataset_name: str,
    max_gen_tokens: int = 128,
):
    """
    Generic streaming evaluation using:
      • `extract_pred(dataset_name, text)`  – post-process LLM outputs
      • `evaluate_pred(dataset_name, pred, gold)` – task-specific scoring
    Works for all supported datasets in `extract_pred/evaluate_pred`.
    """
    total, correct = 0, 0
    model.eval()

    for prompts, golds in tqdm(dataloader, desc="Eval", dynamic_ncols=True):
        # 1. Encode & generate
        enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(
            **enc,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        outs = tokenizer.batch_decode(gen, skip_special_tokens=True)

        if dataset_name in {'anli'}:
            golds = [convert_gold(g) for g in golds]
        # 2. Post-process & score
        preds = [extract_pred(dataset_name, o) for o in outs]
        for p, g in zip(preds, golds):
            correct += int(evaluate_pred(dataset_name, p, g))
            total += 1

        # 3. Online logging (optional)
        wandb.log({"eval/acc": correct / max(total, 1)})

        # Fast-stop if you want a quick sanity check
        # if total >= 200:
        #     break

        print(f"Eval acc so far: {correct / max(total, 1):.4f}")

    final_acc = correct / max(total, 1)
    print(f"Final eval acc: {final_acc:.4f}")
    return final_acc


# -------------------------------------------------------------------------
# Example usage
eval_ds = PromptDataset(TEST_PATH)                # path to *.jsonl with prompt & gold_answer
eval_dl = DataLoader(eval_ds, batch_size=GEN_BS, shuffle=False)

if acc.is_main_process:
    gen_model = acc.unwrap_model(student)         # your model
    gen_model = torch.compile(gen_model, mode="reduce-overhead").eval()
    acc_val = evaluate_loader(
        gen_model,
        tok,                                      # tokenizer
        eval_dl,
        dataset_name=args.task,                   # e.g. "gsm8k", "arc_challenge", ...
        max_gen_tokens=MAX_LEN,
    )
    wandb.log({"eval/acc": acc_val})