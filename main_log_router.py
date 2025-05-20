# -*- coding: utf-8 -*-
"""
main_router_logging.py  ── two‑student MoE with CoT routing + weighted ensemble KL

🔹 Adds per‑epoch logging of the router decisions **and** the indices of the two
   randomly‑selected CoTs that were fed to the router for every training sample.
   The file for epoch *e* is written to  
       <outdir>/router_logs_epoch<e>.jsonl
   Each line is a JSON object of the form
       {"sample": <int>,
        "cot_idx_0": <int>, "route_0": <0|1>,
        "cot_idx_1": <int>, "route_1": <0|1>}
   where route_* is 0 → student‑1, 1 → student‑2.

🔹 No other training logic is changed.  Accelerate is still used; you can run the
   script exactly as before with a single‑GPU command such as
       python main_router_logging.py --task gsm8k …

Feel free to diff against your previous *main.py* – only ~40 new lines were
added and a handful were touched.
"""

import os, json, random, re, copy, torch, argparse, pathlib
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    RobertaModel, RobertaTokenizer,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import datetime, timedelta
import bitsandbytes, torch.distributed as dist, wandb, pdb

from math_utils import is_math_correct, parse_math_boxed, parse_boxed
from utils import (
    get_number_choice, get_alphabet_choice, get_true_false, get_yes_no,
    extract_answer_anli,
)

# ──────────────────────── ARGUMENTS ────────────────────────

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True,
                   choices=[
                       "gsm8k", "math", "arc_challenge", "anli",
                       "commonsense_qa", "date", "strategy_qa", "table_mwp",
                   ],
                   help="folder name under ./data")
    p.add_argument("--model1", type=str,
                   default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--model2", type=str, default="google/gemma-7b-it")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--eval_max_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_s1", type=float, default=5e-6)
    p.add_argument("--lr_s2", type=float, default=2e-4)
    p.add_argument("--lr_misc", type=float, default=1e-4)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--use_kl", type=str2bool, default=True)
    p.add_argument("--is_router", type=str2bool, default=True)
    p.add_argument("--is_quality", type=str2bool, default=True)
    return p.parse_args()


args = parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

# ──────────────────────── PATHS ────────────────────────
root = pathlib.Path("data") / args.task
DATA_PATH = root / "train" / "cot_response.correct.jsonl"
ALL_PATH = root / "train" / "cot_response.enriched.jsonl"
if not args.is_quality:
    DATA_PATH = ALL_PATH
DEV_PATH = root / "train" / "test_cot_distill.jsonl"
TEST_PATH = root / "test" / "cot_response.enriched.jsonl"

MODEL_1 = args.model1
MODEL_2 = args.model2
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.bs
EVAL_BATCH_SIZE = args.eval_bs
MAX_LEN = args.max_len
EVAL_MAX_LEN = args.eval_max_len
PROJECT = f"multi-student-{args.task}"
ENCODER = "FacebookAI/roberta-base"
LR_S1 = args.lr_s1
LR_S2 = args.lr_s2
LR_MISC = args.lr_misc

alpha = 0.5  # unused (baseline ensemble weight)
kl_weight = 0.1
temperature = 1.0
reg_weight = 0.01
USE_KL_DISTILLATION = args.use_kl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────── ACCELERATOR ────────────────────────

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True,
                                           gradient_as_bucket_view=True)
pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(1800))
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, pg_kwargs])
print("local rank =", accelerator.local_process_index, "device =", accelerator.device)

# ──────────────────────── WANDB ────────────────────────

if accelerator.is_main_process:
    wandb.init(
        project=PROJECT,
        name=f"{MODEL_1}_{MODEL_2}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=dict(
            model_1=MODEL_1, model_2=MODEL_2,
            lr_s1=LR_S1, lr_s2=LR_S2,
            encoder=ENCODER, batch_size=BATCH_SIZE,
            eval_bs=EVAL_BATCH_SIZE, num_epochs=NUM_EPOCHS,
            kl_weight=kl_weight, alpha=alpha,
            temperature=temperature, use_kl=USE_KL_DISTILLATION,
            reg_weight=reg_weight, is_router=args.is_router,
            is_quality=args.is_quality,
        ),
    )

# ──────────────────────── HELPERS ────────────────────────

def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
                # only keep when the line "level" is ""level": "Level 3""
                if "level" in data[-1]:
                    if data[-1]["level"] != "Level 1":
                        data.pop()
            except json.JSONDecodeError:
                print("Skipping bad JSON:", line.strip())
    return data

# ──────────────────────── DATASET  (+ CoT index logging) ────────────────────────

class CotDataset(Dataset):
    """Return tensors plus the indices of the two randomly‑picked CoTs."""

    def __init__(self, data, tok_shared, tok_1, tok_2):
        self.data = data
        self.tok_shared, self.tok_1, self.tok_2 = tok_shared, tok_1, tok_2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        prompts = d["prompts"]                    # list[str]
        cots = d["responses"]                    # list[str]

        # choose two distinct CoT indices (with replacement if len==1)
        if len(cots) == 1:
            rand_indices = [0, 0]
        else:
            rand_indices = random.sample(range(len(cots)), 2)

        ids, attn = [], []
        ids_1, attn_1, lbl_1 = [], [], []
        ids_2, attn_2, lbl_2 = [], [], []

        for i, cot_idx in enumerate(rand_indices):
            sel_prompt = prompts[cot_idx]
            sel_cot = cots[cot_idx]

            shared_full = self.tok_shared(sel_prompt + sel_cot,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           max_length=512)
            full_1 = self.tok_1(sel_prompt + sel_cot,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=MAX_LEN)
            full_2 = self.tok_2(sel_prompt + sel_cot,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=MAX_LEN)

            prompt_enc = self.tok_shared(sel_prompt, return_tensors="pt",
                                         padding="max_length", truncation=True,
                                         max_length=512)
            L = prompt_enc["attention_mask"].sum()

            # shared tokenizer (for router encoder)
            ids.append(shared_full["input_ids"].squeeze(0))
            attn.append(shared_full["attention_mask"].squeeze(0))

            # student‑1
            ids_1.append(full_1["input_ids"].squeeze(0))
            attn_1.append(full_1["attention_mask"].squeeze(0))
            lab1 = ids_1[-1].clone()
            lab1[:L] = -100
            lab1[attn_1[-1] == 0] = -100
            lbl_1.append(lab1)

            # student‑2
            ids_2.append(full_2["input_ids"].squeeze(0))
            attn_2.append(full_2["attention_mask"].squeeze(0))
            lab2 = ids_2[-1].clone()
            lab2[:L] = -100
            lab2[attn_2[-1] == 0] = -100
            lbl_2.append(lab2)

        # keep only two variants (0 and 1)
        return (
            ids, attn,
            lbl_1, ids_1, attn_1,
            lbl_2, ids_2, attn_2,
            tuple(rand_indices),  # ← NEW: (cot_idx_0, cot_idx_1)
        )


def collate(batch):
    (
        ids, attn,
        lbl_1, ids_1, attn_1,
        lbl_2, ids_2, attn_2,
        cot_pairs,
    ) = zip(*batch)

    def _stack(two_d_list):
        return torch.stack([torch.stack(el) for el in zip(*two_d_list)])

    return {
        "input_ids": _stack(ids),
        "attention_mask": _stack(attn),
        "input_ids_1": _stack(ids_1),
        "attention_mask_1": _stack(attn_1),
        "labels_1": _stack(lbl_1),
        "input_ids_2": _stack(ids_2),
        "attention_mask_2": _stack(attn_2),
        "labels_2": _stack(lbl_2),
        "cot_idx_0": [p[0] for p in cot_pairs],  # list[int]
        "cot_idx_1": [p[1] for p in cot_pairs],
    }

# ──────────────────────── TOKENIZERS & BACKBONES ────────────────────────

print("Loading tokenizers & frozen backbones …")

tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1)
if tokenizer_1.pad_token is None:
    tokenizer_1.pad_token = tokenizer_1.eos_token

tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2)
if tokenizer_2.pad_token is None:
    tokenizer_2.pad_token = tokenizer_2.eos_token

tokenizer_shared = RobertaTokenizer.from_pretrained(ENCODER)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_1 = AutoModelForCausalLM.from_pretrained(
    MODEL_1, quantization_config=bnb_cfg, torch_dtype=torch.float16,
    trust_remote_code=True,
)
model_1.eval();  # freeze outside LoRA
for p in model_1.parameters():
    p.requires_grad = False

model_2 = AutoModelForCausalLM.from_pretrained(
    MODEL_2, quantization_config=bnb_cfg, torch_dtype=torch.float16,
    trust_remote_code=True,
)
model_2.eval()
for p in model_2.parameters():
    p.requires_grad = False

encoder_backbone = RobertaModel.from_pretrained(
    ENCODER, torch_dtype=torch.float16, trust_remote_code=True,
)
encoder_backbone.eval()
for p in encoder_backbone.parameters():
    p.requires_grad = False

# ──────────────────────── NEW MODULES (router etc.) ────────────────────────

class CotEncoder(nn.Module):
    def __init__(self, backbone, pool="mean"):
        super().__init__()
        self.backbone, self.pool = backbone, pool

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, return_dict=True)
        h = h.last_hidden_state  # (B,T,d)
        if self.pool == "mean":
            num = (h * attention_mask.unsqueeze(-1)).sum(1)
            den = attention_mask.sum(1, keepdim=True)
            return (num / den).float()  # (B,d)
        return h[:, 0].float()


class Router(nn.Module):
    def __init__(self, d_model, tau=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2)
        )
        self.tau = tau

    def forward(self, h):
        return F.gumbel_softmax(self.mlp(h), tau=self.tau, hard=True)


class HiddenProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.float16)

    def forward(self, h):
        return self.proj(h)


class WeightLearner(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1, bias=False, dtype=torch.float16)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


# ──────────────────────── LoRA students ────────────────────────

def make_student(backbone):
    cfg = LoraConfig(r=8, lora_alpha=32,
                     target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.05, bias="none",
                     task_type=TaskType.CAUSAL_LM)
    return get_peft_model(backbone, cfg)

print("Building two LoRA students …")
student1 = make_student(model_1)
student2 = make_student(model_2)

encoder = CotEncoder(encoder_backbone)
router = Router(encoder_backbone.config.hidden_size)
weight_net = WeightLearner(student1.model.config.hidden_size)
proj1 = HiddenProjector(student1.model.config.hidden_size,
                       student1.model.config.hidden_size)
proj2 = HiddenProjector(student2.model.config.hidden_size,
                       student1.model.config.hidden_size)

# ──────────────────────── DATA LOADERS ────────────────────────

train_set = CotDataset(load_data(DATA_PATH), tokenizer_shared,
                       tokenizer_1, tokenizer_2)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)

# ──────────────────────── OPTIMIZER ────────────────────────

opt_groups = [
    {"params": [p for p in student1.parameters() if p.requires_grad], "lr": LR_S1},
    {"params": [p for p in student2.parameters() if p.requires_grad], "lr": LR_S2},
    {
        "params": list(router.parameters()) + list(weight_net.parameters()),
        "lr": LR_MISC,
        "weight_decay": 0.0,
    },
]
optimizer = bitsandbytes.optim.Adam8bit(opt_groups, betas=(0.9, 0.95))

# ──────────────────────── ACCEL PREP ────────────────────────

(student1, student2, proj1, proj2,
 router, weight_net, optimizer, train_loader) = accelerator.prepare(
    student1, student2, proj1, proj2,
    router, weight_net, optimizer, train_loader,
)
encoder = encoder.to(accelerator.device)
DEVICE = accelerator.device

# ──────────────────────── TRAIN ────────────────────────

sample_ctr = 0  # resets every epoch
os.makedirs(args.outdir, exist_ok=True)

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_logs = []  # collect router decisions for the whole epoch
    prog = tqdm(train_loader, disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch}", dynamic_ncols=True)

    for batch in prog:
        # NOTE: retain batch lists for logging before moving to device.
        cot_idx0 = batch.pop("cot_idx_0")  # list[int]
        cot_idx1 = batch.pop("cot_idx_1")  # list[int]

        # move tensors to device – skip the two index lists
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        B, T = batch["input_ids_1"].shape[1:]

        # ── Router forward ───────────────────────────────
        h1 = encoder(batch["input_ids"][0], batch["attention_mask"][0])
        h2 = encoder(batch["input_ids"][1], batch["attention_mask"][1])

        gate1 = router(h1)  # (B,2)
        gate2 = router(h2)

        if dist.is_initialized():
            dist.broadcast(gate1, src=0)
            dist.broadcast(gate2, src=0)

        m1 = gate1[:, 0].bool() | gate2[:, 0].bool()
        m2 = gate1[:, 1].bool() | gate2[:, 1].bool()

        # ── Forward & loss (identical to original) ────────────────────
        optimizer.zero_grad()
        h1_full = torch.zeros(B, T, proj1.proj.out_features,
                              dtype=torch.float16, device=DEVICE)
        h2_full = torch.zeros_like(h1_full)

        ce1 = ce2 = torch.tensor(0.0, device=DEVICE)
        if m1.any():
            out1 = student1(
                input_ids=batch["input_ids_1"][0][m1],
                attention_mask=batch["attention_mask_1"][0][m1],
                labels=batch["labels_1"][0][m1],
                output_hidden_states=True,
            )
            h1_full[m1] = proj1(out1.hidden_states[-1])
            ce1 = out1.loss
        if m2.any():
            out2 = student2(
                input_ids=batch["input_ids_2"][0][m2],
                attention_mask=batch["attention_mask_2"][0][m2],
                labels=batch["labels_2"][0][m2],
                output_hidden_states=True,
            )
            h2_full[m2] = proj2(out2.hidden_states[-1])
            ce2 = out2.loss

        # Weighted ensemble KL (unchanged)
        w_1 = weight_net(h1_full).unsqueeze(-1)
        w_2 = weight_net(h2_full).unsqueeze(-1)
        we = torch.softmax(torch.cat([w_1, w_2], dim=-1), dim=-1)
        logits_ens = (
            we[..., [0]].expand_as(h1_full) * h1_full +
            we[..., [1]].expand_as(h2_full) * h2_full
        ).detach()
        if USE_KL_DISTILLATION and h1_full.abs().sum() and h2_full.abs().sum():
            kl1 = F.mse_loss(h1_full, logits_ens)
            kl2 = F.mse_loss(h2_full, logits_ens)
        else:
            kl1 = kl2 = torch.tensor(0.0, device=DEVICE)
        loss = ce1 + ce2 + kl_weight * (kl1 + kl2) + reg_weight * (
            gate1.mean() + gate2.mean()) / 2

        accelerator.backward(loss)
        optimizer.step()

        # ── LOGGING (router decisions + CoT indices) ───────────────────
        route0 = gate1.argmax(-1).tolist()  # variant‑0 → student (0/1)
        route1 = gate2.argmax(-1).tolist()  # variant‑1 → student (0/1)
        for i in range(B):
            epoch_logs.append({
                "sample": sample_ctr + i,
                "cot_idx_0": int(cot_idx0[i]), "route_0": int(route0[i]),
                "cot_idx_1": int(cot_idx1[i]), "route_1": int(route1[i]),
            })
        sample_ctr += B

        if accelerator.is_main_process:
            wandb.log({
                "loss": loss.item(), "CE1": ce1.item(), "CE2": ce2.item(),
                "KL": (kl1 + kl2).item(), "epoch": epoch,
            })
            prog.set_postfix(loss=f"{loss.item():.4f}")

    # ─────── write epoch log (main process only) ─────────────────────
    if accelerator.is_main_process:
        out_path = pathlib.Path(args.outdir) / f"router_logs_epoch{epoch}_level_1.jsonl"
        with open(out_path, "w") as f:
            for rec in epoch_logs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n✓ Saved router log to {out_path}")

    sample_ctr = 0  # reset for next epoch

print("✓ Training complete – see router logs for details.")
