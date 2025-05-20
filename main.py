# main.py  ── two-student MoE with CoT routing + weighted ensemble KL
import os, json, random, re, copy, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import datetime, timedelta
import bitsandbytes
import torch.distributed as dist
from math_utils import is_math_correct, parse_math_boxed, parse_boxed
from utils import get_number_choice, get_alphabet_choice, get_true_false, get_yes_no, extract_answer_anli
import wandb
import pdb

import argparse, pathlib

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",      type=str, required=True,
                   choices=["gsm8k", "math", "arc_challenge", "anli",
                            "commonsense_qa", "date", "strategy_qa", "table_mwp"],
                   help="folder name under ./data")
    p.add_argument("--model1",    type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--model2",    type=str, default="google/gemma-7b-it")
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--bs",        type=int, default=4)
    p.add_argument("--max_len",   type=int, default=768)
    p.add_argument("--eval_bs",   type=int, default=8)
    p.add_argument("--eval_max_len", type=int, default=1024)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--lr_s1",     type=float, default=5e-6)
    p.add_argument("--lr_s2",     type=float, default=2e-4)
    p.add_argument("--lr_misc",   type=float, default=1e-4)
    p.add_argument("--outdir",    type=str, default="runs")
    p.add_argument("--use_kl", type=str2bool, default=True)
    p.add_argument("--is_router", type=str2bool, default=True)
    p.add_argument("--is_quality", type=str2bool, default=True)
    p.add_argument("--sample_size", type=float, default=1.0)
    p.add_argument("--is_sft", type=str2bool, default=False)
    return p.parse_args()

args = parse_args()

# build file paths from task name
if args.is_sft:
    args.use_kl = False
    args.is_router = False
root         = pathlib.Path("data") / args.task
DATA_PATH    = root / "train" / "cot_response.correct.jsonl"
ALL_PATH     = root / "train" / "cot_response.enriched.jsonl"
if not args.is_quality:
    DATA_PATH = ALL_PATH
if args.is_sft:
    DATA_PATH = root / "train" / "cot_response.correct.jsonl"
DEV_PATH     = root / "train" / "test_cot_distill.jsonl"
TEST_PATH    = root / "test"  / "cot_response.enriched.jsonl"
MODEL_1      = args.model1
MODEL_2      = args.model2
NUM_EPOCHS   = args.epochs
BATCH_SIZE   = args.bs
EVAL_BATCH_SIZE = args.eval_bs
MAX_LEN      = args.max_len
EVAL_MAX_LEN = args.eval_max_len
PROJECT      = f"multi‑student‑{args.task}"
ENCODER = "FacebookAI/roberta-base"
LR_S1   = args.lr_s1          # student‑1 LoRA
LR_S2   = args.lr_s2          # student‑2 LoRA
LR_MISC = args.lr_misc        # shared modules

alpha           = 0.5     # unused (was ensemble weight in baseline)
temperature     = 1.0
kl_weight       = 0.1
reg_weight      = 0.01
USE_KL_DISTILLATION = args.use_kl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cuda.matmul.allow_tf32 = True      # (optional) speed

ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,          # allow conditional experts
    gradient_as_bucket_view=True          # (tiny perf win)
)
pg_kwargs  = InitProcessGroupKwargs(timeout=timedelta(1800))       # long timeout for large models
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, pg_kwargs])
print("local rank =", accelerator.local_process_index, "device =", accelerator.device)

automatic_wandb_config = dict(
    model_1=MODEL_1, model_2=MODEL_2, 
    lr_s1=LR_S1, lr_s2=LR_S2,
    encoder=ENCODER, batch_size=BATCH_SIZE,
    eval_bs=EVAL_BATCH_SIZE, num_epochs=NUM_EPOCHS,
    kl_weight=kl_weight, alpha=alpha,
    temperature=temperature, use_kl=USE_KL_DISTILLATION
    , reg_weight=reg_weight, is_router=args.is_router,
    is_quality=args.is_quality, sample_size=args.sample_size,
    is_sft=args.is_sft, max_len=MAX_LEN
)
if accelerator.is_main_process:
    wandb.init(project=PROJECT,
             name=f"{MODEL_1}_{MODEL_2}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
             config=automatic_wandb_config,
    )

# ──────────────────────── HELPERS ────────────────────────
def load_data(file_path, sample_size=1.0):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    data = data[:int(len(data) * sample_size)]
    return data

# ──────────────────────── DATASETS ────────────────────────
class CotDataset(Dataset):
    """Return (input_ids, attn_mask, labels) where labels mask the instruction."""
    def __init__(self, data, tok, tok_1, tok_2):
        self.data, self.tok, self.tok_1, self.tok_2 = data, tok, tok_1, tok_2
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d      = self.data[idx]
        # pdb.set_trace()
             # teacher CoT (+ answer) list
        if args.is_sft:
            ins   = [d['prompt'], d['prompt']] # list
            cots = [d['gold_answer'], d['gold_answer']]
        else:
            ins    = d["prompts"] # list
            cots    = d['responses']     
    

        # prompt = self.tok(ins, return_tensors="pt", padding="max_length",
        #                   truncation=True, max_length=512)
        # L      = prompt["attention_mask"].sum()       # prompt lenth

        ids = []
        attn = []
        ids_1 = []
        attn_1 = []
        label_1 = []
        ids_2 = []
        attn_2 = []
        label_2 = []
        rand_indices = random.sample(range(len(cots)), len(cots))
        if len(rand_indices) < 2:
            rand_indices = [0, 0]
        for i in range(2):
            # Randomly select a CoT from the list
            try:
                selected_cot = cots[rand_indices[i]]
            except:
                pdb.set_trace()
            selected_ins = ins[rand_indices[i]]
            selected_prompt = self.tok(selected_ins, return_tensors="pt", padding="max_length",
                                      truncation=True, max_length=512)
            L = selected_prompt["attention_mask"].sum()       # prompt length
            full = (self.tok(selected_ins + selected_cot, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=512))
            full_1 = (self.tok_1(selected_ins + selected_cot, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=MAX_LEN))
            full_2 = (self.tok_2(selected_ins + selected_cot, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=MAX_LEN))
            
            ids.append(full["input_ids"].squeeze(0))
            attn.append(full["attention_mask"].squeeze(0))
            ids_1.append(full_1["input_ids"].squeeze(0))
            attn_1.append(full_1["attention_mask"].squeeze(0))
            ids_2.append(full_2["input_ids"].squeeze(0))
            attn_2.append(full_2["attention_mask"].squeeze(0))
            label_1.append(ids_1[i].clone())    
            label_1[i][:L]      = -100                           # ignore instruction tokens
            label_1[i][attn_1[i]==0] = -100                           # ignore padding
            label_2.append(ids_2[i].clone())
            label_2[i][:L]      = -100                           # ignore instruction tokens
            label_2[i][attn_2[i]==0] = -100                           # ignore padding

        return ids, attn, label_1, ids_1, attn_1, label_2, ids_2, attn_2

def collate(batch):
    ids, attn, lbl_1, ids_1, attn_1, lbl_2, ids_2, attn_2 = zip(*batch)
    return dict(
        input_ids     = torch.stack([torch.stack(ids) for ids in zip(*ids)]),
        attention_mask = torch.stack([torch.stack(attn) for attn in zip(*attn)]),
        input_ids_1 = torch.stack([torch.stack(ids_1) for ids_1 in zip(*ids_1)]),
        attention_mask_1 = torch.stack([torch.stack(attn_1) for attn_1 in zip(*attn_1)]),
        input_ids_2 = torch.stack([torch.stack(ids_2) for ids_2 in zip(*ids_2)]),
        attention_mask_2 = torch.stack([torch.stack(attn_2) for attn_2 in zip(*attn_2)]),
        labels_1 = torch.stack([torch.stack(lbl_1ins) for lbl_1ins in zip(*lbl_1)]),
        labels_2 = torch.stack([torch.stack(lbl_2ins) for lbl_2ins in zip(*lbl_2)]),
    )


# ──────────────────────── TOKENIZER & BASE MODEL ────────────────────────
print("Loading tokenizer & frozen backbone …")
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1)
tokenizer_1.pad_token = tokenizer_1.eos_token
tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2)
tokenizer_2.pad_token = tokenizer_2.eos_token
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_1 = AutoModelForCausalLM.from_pretrained(
    MODEL_1,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
for p in model_1.parameters(): p.requires_grad = False
model_1.eval()

model_2 = AutoModelForCausalLM.from_pretrained(
    MODEL_2,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
for p in model_2.parameters(): p.requires_grad = False
model_2.eval()

encoder_backbone = RobertaModel.from_pretrained(
    ENCODER,                      # "google-bert/bert-base-uncased"
    # quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,       # if it’s a custom repo
)

for p in encoder_backbone.parameters():
    p.requires_grad = False       # 冻结
encoder_backbone.eval().to(DEVICE)


# ──────────────────────── NEW MODULES ────────────────────────
class CotEncoder(nn.Module):
    """Frozen encoder that pools last hidden state."""
    def __init__(self, backbone, pool="mean"):
        super().__init__()
        self.backbone, self.pool = backbone, pool
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        h = self.backbone(
            input_ids     = input_ids,
            attention_mask= attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state

        if self.pool == "mean":
            num = (h * attention_mask.unsqueeze(-1)).sum(1)
            den = attention_mask.sum(1, keepdim=True)
            return (num / den).float()          # [B,d]
        return h[:,0].float()

class Router(nn.Module):
    """Two-way hard router with straight-through Gumbel-Softmax."""
    def __init__(self, d_model, tau=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 2)
        )
        self.tau = tau
    def forward(self, h):       # h:[B,d]
        return F.gumbel_softmax(self.mlp(h), tau=self.tau, hard=True)

class HiddenProjector(nn.Module):
    def __init__(self, in_dim, out_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.float16)
    def forward(self, h):                    # h: [B,T,d]
        return self.proj(h)                  # [B,T,out_dim]

class WeightLearner(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1, bias=False, dtype=torch.float16)
    def forward(self, h):       # h:[B,d]
        return torch.sigmoid(self.fc(h)).squeeze(-1)   # [B]

# ──────────────────────── LoRA STUDENTS ────────────────────────
def make_student(backbone):
    cfg   = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(backbone, cfg)
    model.print_trainable_parameters()
    return model

print("Building two LoRA students …")
student1, student2 = make_student(model_1), make_student(model_2)
# pdb.set_trace()

# ──────────────────────── INSTANTIATE NEW UTILS ────────────────────────
encoder     = CotEncoder(encoder_backbone)
router      = Router(encoder_backbone.config.hidden_size)
weight_net  = WeightLearner(student1.model.config.hidden_size)
proj1 = HiddenProjector(student1.model.config.hidden_size, student1.model.config.hidden_size).to(DEVICE)
proj2 = HiddenProjector(student2.model.config.hidden_size, student1.model.config.hidden_size).to(DEVICE)


# ──────────────────────── DATA LOADERS ────────────────────────
train_set = CotDataset(load_data(DATA_PATH, sample_size=args.sample_size), tokenizer, tokenizer_1, tokenizer_2)
# train_set = torch.utils.data.Subset(train_set, range(0, 10))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)

# ──────────────────────── OPTIMIZER ────────────────────────
opt_groups = [
    {   # LoRA adapters in student‑1
        "params": [p for p in student1.parameters() if p.requires_grad],
        "lr": LR_S1
    },
    {   # LoRA adapters in student‑2
        "params": [p for p in student2.parameters() if p.requires_grad],
        "lr": LR_S2
    },
    {   # shared modules
        "params": list(router.parameters()) + list(weight_net.parameters()),
        "lr": LR_MISC,
        "weight_decay": 0.0          # example: turn off WD just for these
    },
]
# optimizer = torch.optim.AdamW(trainables, lr=LR)
optimizer = bitsandbytes.optim.Adam8bit(opt_groups, betas=(0.9, 0.95))

# ──────────────────────── ACCELERATOR ────────────────────────
# accelerator.wait_for_everyone()

# if accelerator.is_local_main_process:
#     import pdb; pdb.set_trace()          # 只 rank‑0 停
# accelerator.wait_for_everyone() 

D = proj1.proj.out_features
(student1, student2, proj1, proj2,
router, weight_net, optimizer, train_loader) = accelerator.prepare(
    student1, student2, proj1, proj2,
    router, weight_net, optimizer, train_loader
)

encoder = encoder.to(accelerator.device)

DEVICE = accelerator.device
step=0
# ──────────────────────── TRAIN ────────────────────────
for epoch in range(1, NUM_EPOCHS + 1):
    prog = tqdm(train_loader, disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch}", dynamic_ncols=True)

    for batch in prog:
        # batch 结构:
        #   input_ids         [2,B,T]  (BERT tokenizer)
        #   attention_mask    [2,B,T]
        #   input_ids_1/_2    [2,B,T]  (各自 decoder tokenizer)
        #   labels_1/_2       [2,B,T]

        # ── 先移动到 device ────────────────────────────
        step += 1
        B, T = batch["input_ids_1"].shape[1:]       # 原始 batch 尺寸
        
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # 取两条 CoT 变体 (0 / 1)
        enc_ids1, enc_mask1 = batch["input_ids"][0], batch["attention_mask"][0]
        enc_ids2, enc_mask2 = batch["input_ids"][1], batch["attention_mask"][1]

        # ── 编码 + 路由门控 ─────────────────────────────
        h1 = encoder(enc_ids1, enc_mask1)          # [B,d]
        h2 = encoder(enc_ids2, enc_mask2)

        gate1 = router(h1)                         # [B,2] one‑hot
        gate2 = router(h2)

        # 主进程广播，确保多 GPU 一致
        if dist.is_initialized():
            dist.broadcast(gate1, src=0)
            dist.broadcast(gate2, src=0)

        m1 = (gate1[:, 0].bool() | gate2[:, 0].bool())  # 选给 student1 的样本
        m2 = (gate1[:, 1].bool() | gate2[:, 1].bool())  # 选给 student2 的样本
        # if not args.is_router:
        #     m1 = torch.ones_like(m1, dtype=torch.bool)
        #     m2 = torch.ones_like(m2, dtype=torch.bool)
        h1_full = torch.zeros(B, T, D, dtype=torch.float16, device=DEVICE)
        h2_full = torch.zeros_like(h1_full)

        optimizer.zero_grad()

        # ── student‑1 forward ──────────────────────────
        if m1.any():
            out1 = student1(
                input_ids     = batch["input_ids_1"][0][m1],   # 用 CoT 变体‑0
                attention_mask= batch["attention_mask_1"][0][m1],
                labels        = batch["labels_1"][0][m1],
                output_hidden_states=True
            )
            # logits1 = torch.zeros((*batch["labels_1"][0].shape,
            #                        student1.model.config.vocab_size),
            #                        dtype=out1.logits.dtype, device=DEVICE)
            # logits1[m1] = out1.logits
            hid_1  = proj1(out1.hidden_states[-1])
            h1_full[m1] = hid_1
            ce1 = out1.loss
        else:
            # logits1 = torch.zeros((*batch["labels_1"][0].shape,
            #                        student1.model.config.vocab_size),
            #                        dtype=torch.float16, device=DEVICE)
            # hid_1   = proj1(out1.hidden_states[-1])
            # h1_full[m1] = hid_1
            ce1 = torch.tensor(0., device=DEVICE)
            

        # ── student‑2 forward ──────────────────────────
        if m2.any():
            out2 = student2(
                input_ids     = batch["input_ids_2"][0][m2],
                attention_mask= batch["attention_mask_2"][0][m2],
                labels        = batch["labels_2"][0][m2],
                output_hidden_states=True
            )
            # logits2 = torch.zeros((*batch["labels_2"][0].shape,
            #                        student2.model.config.vocab_size),
            #                        dtype=out2.logits.dtype, device=DEVICE)
            # logits2[m2] = out2.logits
            hid_2   = proj2(out2.hidden_states[-1])
            h2_full[m2] = hid_2
            ce2 = out2.loss
        else:
            # logits2 = torch.zeros((*batch["labels_2"][0].shape,
            #                        student2.model.config.vocab_size),
            #                        dtype=out2.logits.dtype, device=DEVICE)
            # hid_2   = proj2(out2.hidden_states[-1])
            # h2_full[m2] = hid_2
            ce2 = torch.tensor(0., device=DEVICE)

        # ── 加权融合 + KL 蒸馏 ───────────────────────────
        tok_mask = (batch["labels_1"][0] != -100).unsqueeze(-1)
        w_1 = weight_net((h1_full)).unsqueeze(-1)       # [B,1,1]
        w_2 = weight_net((h2_full)).unsqueeze(-1)       # [B,1,1]
        # softmax over the two students
        we = torch.softmax(torch.cat([w_1, w_2], dim=-1), dim=-1)  # [B, 1, 2]

        logits_ens = (we[..., [0]].expand(h1_full.shape) * h1_full + we[..., [1]].expand(h2_full.shape) * h2_full).detach()
        # logits_ens = (w * h1_full[tok_mask] + (1 - w) * h2_full[tok_mask]).to(torch.bfloat16).detach()

        if USE_KL_DISTILLATION:
            # logp1 = F.log_softmax(logits1 / temperature, dim=-1)
            # logp2 = F.log_softmax(logits2 / temperature, dim=-1)
            # logpE = F.log_softmax(logits_ens / temperature, dim=-1)

            # tok_mask = (batch["labels_1"][0] != -100)
            # kl1 = F.kl_div(logp1, logpE, reduction="none",
            #                log_target=True).sum(-1)[tok_mask].mean()
            # kl2 = F.kl_div(logp2, logpE, reduction="none",
            #                log_target=True).sum(-1)[tok_mask].mean()
            if h1_full.sum() == 0 or h2_full.sum() == 0:
                kl1 = kl2 = torch.tensor(0., device=DEVICE)
            else:
                kl1 = F.mse_loss(h1_full, logits_ens)
                kl2 = F.mse_loss(h2_full, logits_ens)
            loss = ce1 + ce2 + kl_weight * (kl1 + kl2) + reg_weight * (gate1.mean() + gate2.mean()) / 2
        else:
            kl1 = kl2 = torch.tensor(0., device=DEVICE)
            loss = ce1 + ce2

        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_main_process:
            wandb.log({
                "loss": loss.item(), "CE1": ce1.item(),
                "CE2": ce2.item(), "KL": (kl1 + kl2).item(),
                "epoch": epoch, "step": step
            })



        prog.set_postfix(
            loss=f"{loss.item():.4f}",
            CE1=f"{ce1.item():.3f}",
            CE2=f"{ce2.item():.3f}",
            KL=f"{(kl1+kl2).item():.3f}"
        )


####### ──────────────────────── EVALUATION ──────────────────────── #######
print("✓ Training done")

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
        self.data = load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["prompt"], d["gold_answer"]


def _calc_seq_logprobs(scores, seq, start_idx: int) -> torch.Tensor:
    """
    Sum log-probs of generated tokens, batched.

    scores   – tuple(list) of length L with tensors (B,V)
    seq      – generated ids (B, prompt_len+L)
    start_idx– first newly-generated token position in `seq`
    returns  – (B,) log-prob for each example
    """
    L, B = len(scores), seq.size(0)
    lp = torch.zeros(B, device=scores[0].device, dtype=torch.float32)
    for t in range(L):
        step_lp = scores[t].log_softmax(-1)          # (B,V)
        tok     = seq[:, start_idx + t]              # (B,)
        lp     += step_lp.gather(1, tok.unsqueeze(1)).squeeze(1)
    return lp


@torch.no_grad()
def evaluate_dual_loader(
    model1, tokenizer1,
    model2, tokenizer2,
    dataloader: DataLoader,
    dataset_name: str,
    max_gen_tokens: int = 128,
):
    """
    • Generates with both students.
    • Uses log-prob of each model’s own output to pick an **ensemble** answer.
    • Logs running & final accuracies for:
          – student-1
          – student-2
          – ensemble (best-log-prob pick)

    All task-specific post-processing is funnelled through
    `extract_pred` and `evaluate_pred`, identical to the single-model flow.
    """
    model1.eval(); model2.eval()

    total           = 0
    correct_s1      = 0
    correct_s2      = 0
    correct_ensemble= 0

    for prompts, golds in tqdm(dataloader, desc="Eval-dual", dynamic_ncols=True):

        # --- 1. Encode & generate ------------------------------------------------
        enc1 = tokenizer1(list(prompts), return_tensors="pt", padding=True).to(DEVICE)
        enc2 = tokenizer2(list(prompts), return_tensors="pt", padding=True).to(DEVICE)

        p_len1 = enc1.attention_mask.sum(-1)          # prompt lengths (B,)
        p_len2 = enc2.attention_mask.sum(-1)

        out1 = model1.generate(
            **enc1,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer1.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        out2 = model2.generate(
            **enc2,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer2.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        txt1 = tokenizer1.batch_decode(out1.sequences, skip_special_tokens=True)
        txt2 = tokenizer2.batch_decode(out2.sequences, skip_special_tokens=True)

        # --- 2. Post-process predictions ----------------------------------------
        if dataset_name.lower() == "anli":
            golds = [convert_gold(g) for g in golds]

        preds1 = [extract_pred(dataset_name, t) for t in txt1]
        preds2 = [extract_pred(dataset_name, t) for t in txt2]

        # log-prob of the generated continuation (for ensemble pick)
        lp1 = _calc_seq_logprobs(out1.scores, out1.sequences, int(p_len1.min())).cpu()
        lp2 = _calc_seq_logprobs(out2.scores, out2.sequences, int(p_len2.min())).cpu()

        # --- 3. Scoring ----------------------------------------------------------
        for p1, p2, g, l1, l2 in zip(preds1, preds2, golds, lp1, lp2):

            # individual students
            correct_s1      += int(evaluate_pred(dataset_name, p1, g))
            correct_s2      += int(evaluate_pred(dataset_name, p2, g))

            # ensemble: keep the prediction with the higher own-model log-prob
            p_best          = p1 if l1 >= l2 else p2
            correct_ensemble+= int(evaluate_pred(dataset_name, p_best, g))

            total += 1

        # --- 4. Online logging ---------------------------------------------------
        wandb.log({
            "eval/acc_s1":      correct_s1 / total,
            "eval/acc_s2":      correct_s2 / total,
            "eval/acc_ensemble":correct_ensemble / total,
        })

        print(
            f"\rAcc  | S1: {correct_s1/total:.4f}  "
            f"S2: {correct_s2/total:.4f}  "
            f"Ensemble: {correct_ensemble/total:.4f}",
            end="",
        )

        # (optional) quick sanity-check cut-off
        # if total >= 200: break

    print(
        f"\nFinal acc –  S1: {correct_s1/total:.4f}  "
        f"S2: {correct_s2/total:.4f}  "
        f"Ensemble: {correct_ensemble/total:.4f}"
    )
    return {
        "acc_s1":       correct_s1      / total,
        "acc_s2":       correct_s2      / total,
        "acc_ensemble": correct_ensemble/ total,
    }


# ─────────────────────────────── usage ───────────────────────────────
if accelerator.is_main_process:
    eval_ds = PromptDataset(TEST_PATH)
    eval_dl = DataLoader(eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    gen_s1 = torch.compile(accelerator.unwrap_model(student1), mode="reduce-overhead").eval()
    gen_s2 = torch.compile(accelerator.unwrap_model(student2), mode="reduce-overhead").eval()

    res = evaluate_dual_loader(
        gen_s1, tokenizer_1,
        gen_s2, tokenizer_2,
        eval_dl,
        dataset_name=args.task,     # e.g. "gsm8k", "arc_challenge", ...
        max_gen_tokens=EVAL_MAX_LEN,
    )
    wandb.log({f"eval/{k}": v for k, v in res.items()})