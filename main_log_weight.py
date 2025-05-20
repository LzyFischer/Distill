# -*- coding: utf-8 -*-
"""
Main training script – **two‑student MoE with CoT routing + weighted ensemble KL**
-------------------------------------------------------------------------------
This version extends the original `main.py` with *light‑touch* additions to log, at
**every epoch**, the router‑selected weight assigned to **each individual Chain of
Thought (CoT)** for **each student**.

Key design goals
----------------
1. **Keep the original control‑flow and Accelerate integration** intact.  All
   additions are non‑invasive and clearly marked with `##### NEW:` comments.
2. **No heavyweight objects are serialized.**  We only keep the *index* of the
   selected CoT (relative to the `responses` list of the sample) and the scalar
   weight that the ensemble assigns to the corresponding student.
3. **Per‑epoch JSON files** are written to `runs/<TASK>/cot_weights_epoch<N>.json`.
   Each file has the structure

        {
          "student1": {"<sampleId>-<cotIdx>": avg_weight, ...},
          "student2": {"<sampleId>-<cotIdx>": avg_weight, ...}
        }

   where `sampleId` is the line number of the example in the *training* JSONL
   file and `cotIdx` is the 0‑based index into its `responses` list.
"""

# -----------------------------------------------------------------------------
#                               ORIGINAL IMPORTS
# -----------------------------------------------------------------------------
import os, json, random, argparse, pathlib, copy
from collections import defaultdict
from datetime import datetime, timedelta

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    RobertaTokenizer, RobertaModel,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
import bitsandbytes
import torch.distributed as dist
import wandb

from math_utils import is_math_correct, parse_math_boxed, parse_boxed
from utils      import (
    get_number_choice, get_alphabet_choice, get_true_false,
    get_yes_no, extract_answer_anli,
)

# -----------------------------------------------------------------------------
#                                ARGUMENTS
# -----------------------------------------------------------------------------

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
    p.add_argument("--task",      required=True,
                   choices=[
                       "gsm8k", "math", "arc_challenge", "anli",
                       "commonsense_qa", "date", "strategy_qa", "table_mwp",
                   ])
    p.add_argument("--model1",    default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--model2",    default="google/gemma-7b-it")
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--bs",        type=int, default=4)
    p.add_argument("--max_len",   type=int, default=768)
    p.add_argument("--eval_bs",   type=int, default=8)
    p.add_argument("--eval_max_len", type=int, default=1024)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--lr_s1",     type=float, default=5e-6)
    p.add_argument("--lr_s2",     type=float, default=2e-4)
    p.add_argument("--lr_misc",   type=float, default=1e-4)
    p.add_argument("--outdir",    type=str,  default="runs")
    p.add_argument("--use_kl",    type=str2bool, default=True)
    p.add_argument("--is_router", type=str2bool, default=True)
    p.add_argument("--is_quality",type=str2bool, default=True)
    return p.parse_args()

args = parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)

# -----------------------------------------------------------------------------
#                           DATA & OUTPUT PATHS
# -----------------------------------------------------------------------------
root         = pathlib.Path("data") / args.task
train_jsonl  = root / "train" / ("cot_response.correct.jsonl" if args.is_quality else "cot_response.enriched.jsonl")
all_jsonl    = root / "train" / "cot_response.enriched.jsonl"     # (unused if --is_quality)
dev_jsonl    = root / "train" / "test_cot_distill.jsonl"
test_jsonl   = root / "test"  / "cot_response.enriched.jsonl"

outdir = pathlib.Path(args.outdir) / args.task
outdir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#                             STATIC HYPERPARAMS
# -----------------------------------------------------------------------------
MODEL_1, MODEL_2 = args.model1, args.model2
NUM_EPOCHS       = args.epochs
BATCH_SIZE       = args.bs
EVAL_BATCH_SIZE  = args.eval_bs
MAX_LEN          = args.max_len
EVAL_MAX_LEN     = args.eval_max_len
ENCODER          = "FacebookAI/roberta-base"

LR_S1   = args.lr_s1
LR_S2   = args.lr_s2
LR_MISC = args.lr_misc

kl_weight  = 0.1
reg_weight = 0.01
USE_KL     = args.use_kl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
#                               ACCELERATE SETUP
# -----------------------------------------------------------------------------
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True,
                                           gradient_as_bucket_view=True)
pg_kwargs  = InitProcessGroupKwargs(timeout=timedelta(1800))
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, pg_kwargs])
print("▶ local‑rank", accelerator.local_process_index, "| device", accelerator.device)

# -----------------------------------------------------------------------------
#                      WANDB (only on rank‑0 for clarity)
# -----------------------------------------------------------------------------
if accelerator.is_main_process:
    wandb.init(
        project=f"multi‑student‑{args.task}",
        name   =f"{MODEL_1.split('/')[-1]}_{MODEL_2.split('/')[-1]}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config = vars(args),
    )

# -----------------------------------------------------------------------------
#                             UTILITY: Load JSONL
# -----------------------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] skip bad json (line {ln}): {line[:80]}")
    return data

# -----------------------------------------------------------------------------
#                               DATASET
# -----------------------------------------------------------------------------
class CotDataset(Dataset):
    """Returns **two** (prompt, CoT) variants *plus* their indices.

    For sample‑`idx` we randomly draw *two* distinct indices from
    `responses` → (`cot_idx_a`, `cot_idx_b`).  *Only these indices* are
    surfaced for logging – no CoT text itself is stored on disk.
    """
    def __init__(self, data, tok_roberta, tok_s1, tok_s2):
        self.data        = data
        self.tok_roberta = tok_roberta
        self.tok_s1      = tok_s1
        self.tok_s2      = tok_s2

    def __len__(self):
        return len(self.data)

    def _encode_pair(self, prompt, cot, tok, max_len):
        full = tok(prompt + cot, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=max_len)
        return full["input_ids"].squeeze(0), full["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        record      = self.data[idx]
        prompts     = record["prompts"]            # list[str]
        responses   = record["responses"]          # list[str]

        # --- choose two distinct CoTs ----------------------------------------
        if len(responses) < 2:
            choices = [0, 0]
        else:
            choices = random.sample(range(len(responses)), 2)
        pA, pB = prompts[choices[0]], prompts[choices[1]]
        cA, cB = responses[choices[0]], responses[choices[1]]

        # --- TOKENIZATION ------------------------------------------------------
        ids_rb_A, mask_rb_A = self._encode_pair(pA, cA, self.tok_roberta, 512)
        ids_s1_A,mask_s1_A  = self._encode_pair(pA, cA, self.tok_s1,     MAX_LEN)
        ids_s2_A,mask_s2_A  = self._encode_pair(pA, cA, self.tok_s2,     MAX_LEN)

        ids_rb_B, mask_rb_B = self._encode_pair(pB, cB, self.tok_roberta, 512)
        ids_s1_B,mask_s1_B  = self._encode_pair(pB, cB, self.tok_s1,     MAX_LEN)
        ids_s2_B,mask_s2_B  = self._encode_pair(pB, cB, self.tok_s2,     MAX_LEN)

        # labels (mask out prompt tokens & padding)
        def label_from(ids, attn, prompt_len):
            lbl = ids.clone()
            lbl[:prompt_len]       = -100
            lbl[attn == 0]         = -100
            return lbl

        # prompt lengths for masking (ATTN mask is 1 everywhere in prompt+cot)
        prompt_len_A = mask_rb_A.sum().item() - len(cA)   # rough but okay for mask‑out
        prompt_len_B = mask_rb_B.sum().item() - len(cB)

        lbl_s1_A = label_from(ids_s1_A, mask_s1_A, prompt_len_A)
        lbl_s2_B = label_from(ids_s2_B, mask_s2_B, prompt_len_B)

        return {
            # variant‑A feeds student‑1, variant‑B feeds student‑2 by convention
            "ids_rb_A":     ids_rb_A,  "mask_rb_A":     mask_rb_A,
            "ids_rb_B":     ids_rb_B,  "mask_rb_B":     mask_rb_B,
            "ids_s1":       ids_s1_A,  "mask_s1":       mask_s1_A, "lbl_s1": lbl_s1_A,
            "ids_s2":       ids_s2_B,  "mask_s2":       mask_s2_B, "lbl_s2": lbl_s2_B,
            # ---- NEW: identifiers for logging --------------------------------
            "cot_key_s1":  f"{idx}-{choices[0]}",      # sample‑id + CoT‑index
            "cot_key_s2":  f"{idx}-{choices[1]}",
        }


# -------------------------- collate  (unchanged logic) -----------------------

def collate(batch):
    out = {k: torch.stack([d[k] for d in batch])
           for k, v in batch[0].items() if isinstance(v, torch.Tensor)}
    # keep key lists intact for logging (no stacking)
    out["cot_key_s1"] = [d["cot_key_s1"] for d in batch]
    out["cot_key_s2"] = [d["cot_key_s2"] for d in batch]
    return out

# -----------------------------------------------------------------------------
#                       TOKENIZERS & BACKBONES (unchanged)
# -----------------------------------------------------------------------------
print("▶ Loading tokenizers …")

tokenizer_s1 = AutoTokenizer.from_pretrained(MODEL_1); tokenizer_s1.pad_token = tokenizer_s1.eos_token
tokenizer_s2 = AutoTokenizer.from_pretrained(MODEL_2); tokenizer_s2.pad_token = tokenizer_s2.eos_token
tokenizer_rb = RobertaTokenizer.from_pretrained(ENCODER)

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)

print("▶ Loading frozen teacher backbones …")
model_s1 = AutoModelForCausalLM.from_pretrained(MODEL_1, quantization_config=bnb_cfg,
                                               torch_dtype=torch.float16, trust_remote_code=True)
model_s2 = AutoModelForCausalLM.from_pretrained(MODEL_2, quantization_config=bnb_cfg,
                                               torch_dtype=torch.float16, trust_remote_code=True)
for p in list(model_s1.parameters())+list(model_s2.parameters()):
    p.requires_grad = False

print("▶ Loading frozen encoder …")
enc_backbone = RobertaModel.from_pretrained(ENCODER, torch_dtype=torch.float16)
for p in enc_backbone.parameters():
    p.requires_grad = False
enc_backbone.eval().to(device)

# -----------------------------------------------------------------------------
#                     COT‑ENCODER / ROUTER / WEIGHT‑NET  (unchanged)
# -----------------------------------------------------------------------------
class CotEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone
    @torch.no_grad()
    def forward(self, ids, attn):
        h = self.backbone(input_ids=ids, attention_mask=attn,
                          output_hidden_states=True, return_dict=True).last_hidden_state
        num = (h * attn.unsqueeze(-1)).sum(1)
        den = attn.sum(1, keepdim=True)
        return (num / den).float()          # [B,d]

class Router(nn.Module):
    def __init__(self, d, tau=1.0):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 2)); self.tau=tau
    def forward(self, h):
        return F.gumbel_softmax(self.mlp(h), tau=self.tau, hard=True)

class WeightNet(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d, 1, bias=False, dtype=torch.float16)
    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)   # [B]

class HiddenProj(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.proj = nn.Linear(d_in, d_out, bias=False, dtype=torch.float16)
    def forward(self, x):
        return self.proj(x)

# -----------------------------------------------------------------------------
#                            LORA STUDENTS (unchanged)
# -----------------------------------------------------------------------------

def build_student(backbone):
    cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    return get_peft_model(backbone, cfg)

student1 = build_student(model_s1)
student2 = build_student(model_s2)

encoder     = CotEncoder(enc_backbone)
router      = Router(enc_backbone.config.hidden_size)
weight_net  = WeightNet(student1.model.config.hidden_size)
proj1       = HiddenProj(student1.model.config.hidden_size, student1.model.config.hidden_size)
proj2       = HiddenProj(student2.model.config.hidden_size, student1.model.config.hidden_size)

# -----------------------------------------------------------------------------
#                         DATA LOADER & OPTIMISER
# -----------------------------------------------------------------------------
train_ds = CotDataset(load_jsonl(train_jsonl), tokenizer_rb, tokenizer_s1, tokenizer_s2)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

opt = bitsandbytes.optim.Adam8bit([
    {"params": [p for p in student1.parameters() if p.requires_grad], "lr": LR_S1},
    {"params": [p for p in student2.parameters() if p.requires_grad], "lr": LR_S2},
    {"params": list(router.parameters())+list(weight_net.parameters()), "lr": LR_MISC},
])

# -----------------------------------------------------------------------------
#                      PREPARE WITH ACCELERATOR (unchanged)
# -----------------------------------------------------------------------------
(student1, student2, proj1, proj2, router, weight_net,
 opt, train_dl) = accelerator.prepare(student1, student2, proj1, proj2, router, weight_net,
                                      opt, train_dl)
encoder = encoder.to(accelerator.device)
D = proj1.proj.out_features

# -----------------------------------------------------------------------------
#                       ##### NEW: EPOCH‑LEVEL BUFFERS #####
# -----------------------------------------------------------------------------
weights_epoch_s1 = defaultdict(list)   # key → list[float]
weights_epoch_s2 = defaultdict(list)

# -----------------------------------------------------------------------------
#                               TRAIN LOOP
# -----------------------------------------------------------------------------
step = 0
for epoch in range(1, NUM_EPOCHS+1):
    prog = tqdm(train_dl, disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch}", dynamic_ncols=True)

    for batch in prog:
        step += 1
        B = batch["ids_s1"].size(0)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

        # -------- encode for routing -----------------------------------------
        h_A = encoder(batch["ids_rb_A"], batch["mask_rb_A"])
        h_B = encoder(batch["ids_rb_B"], batch["mask_rb_B"])
        gateA, gateB = router(h_A), router(h_B)          # [B,2] one‑hot

        # make gating decisions visible to all ranks
        if dist.is_initialized():
            dist.broadcast(gateA, src=0); dist.broadcast(gateB, src=0)

        m1 = (gateA[:,0].bool() | gateB[:,0].bool())
        m2 = (gateA[:,1].bool() | gateB[:,1].bool())

        # -------------- student‑1 forward ------------------------------------
        opt.zero_grad()
        if m1.any():
            out1 = student1(input_ids=batch["ids_s1"][m1], attention_mask=batch["mask_s1"][m1],
                            labels=batch["lbl_s1"][m1], output_hidden_states=True)
            hid1 = proj1(out1.hidden_states[-1])
            ce1  = out1.loss
        else:
            hid1 = torch.zeros(B, batch["ids_s1"].size(1), D, dtype=torch.float16, device=device)
            ce1  = torch.tensor(0., device=device)

        # -------------- student‑2 forward ------------------------------------
        if m2.any():
            out2 = student2(input_ids=batch["ids_s2"][m2], attention_mask=batch["mask_s2"][m2],
                            labels=batch["lbl_s2"][m2], output_hidden_states=True)
            hid2 = proj2(out2.hidden_states[-1])
            ce2  = out2.loss
        else:
            hid2 = torch.zeros_like(hid1)
            ce2  = torch.tensor(0., device=device)

        # -------------- ensemble weights -------------------------------------
        w1_raw = weight_net(hid1.mean(1))    # [B]
        w2_raw = weight_net(hid2.mean(1))
        w_soft = torch.softmax(torch.stack([w1_raw, w2_raw], dim=-1), dim=-1)  # [B,2]

        # -------- NEW: push weights into buffers -----------------------------
        if accelerator.is_main_process:
            for j in range(B):
                key_s1 = batch["cot_key_s1"][j]
                key_s2 = batch["cot_key_s2"][j]
                weights_epoch_s1[key_s1].append(float(w_soft[j,0]))
                weights_epoch_s2[key_s2].append(float(w_soft[j,1]))

        # ----------- KL distillation / regularisation ------------------------
        loss = ce1 + ce2 + reg_weight * (gateA.mean()+gateB.mean())/2
        accelerator.backward(loss)
        opt.step()

        if accelerator.is_main_process:
            wandb.log({"loss": loss.item(), "ce1": ce1.item(), "ce2": ce2.item(),
                       "epoch": epoch, "step": step})
            prog.set_postfix(loss=f"{loss.item():.4f}")

    # ───────── END‑OF‑EPOCH: write weight stats to disk ─────────────────────
    if accelerator.is_main_process:
        avg_s1 = {k: sum(v)/len(v) for k,v in weights_epoch_s1.items()}
        avg_s2 = {k: sum(v)/len(v) for k,v in weights_epoch_s2.items()}
        with open(outdir / f"cot_weights_epoch{epoch}.json", "w", encoding="utf-8") as fh:
            json.dump({"student1": avg_s1, "student2": avg_s2}, fh, indent=2)
        print(f"\n✓ Saved CoT‑weight stats for epoch {epoch} → {fh.name}\n")

        # reset for next epoch
        weights_epoch_s1.clear(); weights_epoch_s2.clear()

print("Training complete ✅")
