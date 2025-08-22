# Distill: CoT Prompting & Student–Router Distillation

This repository contains two complementary components:

1. **Prompting pipelines (`cot_pipeline.py`, `prompt_gen.py`)** — generate Chain-of-Thought (CoT) prompts per dataset, batch-call an LLM API (Gemini/OpenAI), and write enriched logs with correctness flags.
2. **Training pipelines (`main.py` and variants)** — a two-student mixture-of-experts (MoE) setup with a CoT router and optional KL-weighted ensembling, built on 🤗 Transformers + LoRA + Accelerate.

The goal is to (a) produce diverse *reasoning traces* and (b) learn routing/ensembling policies over students using those traces.

---

## 🔧 Installation

**Python**: 3.10+ recommended.

```bash
# create env (example)
conda create -n distill python=3.10 -y
conda activate distill

# core deps (minimal)
pip install -U pip wheel
pip install "transformers>=4.41" 
pip install "accelerate>=0.28" 
pip install "peft>=0.11" 
pip install "bitsandbytes>=0.43"
pip install "torch>=2.1"

# choose ONE of the LLM API clients (or both)
pip install google-generativeai      # if you plan to call Gemini
pip install openai                   # if you plan to call OpenAI
(Others including tqdm, regex ...)
```

---

## 🔐 API Keys (for prompting)

Set exactly one of the following (depending on which client you use):

```bash
# Gemini
export GOOGLE_API_KEY="...your key..."

# OpenAI
export OPENAI_API_KEY="...your key..."
```

Inside the code, you must implement **`batch_call_gemini_api`** (or replace it with your preferred provider) to actually send batched prompts and return text responses. See **[Implement your API caller](#-implement-your-api-caller)** below.

---

## 📁 Repository Structure

```
Distill/
├── cot_pipeline.py                # Multi-style CoT prompting over datasets
├── prompt_gen.py                  # Single powerful wrapper prompt variant
├── math_utils.py                  # Robust math answer parsing & equivalence
├── utils.py                       # Choice / answer extractors & helpers
├── main.py                        # Two-student MoE + router + (optional) KL
├── main_log_router.py             # Same as main.py + per-epoch router logs
├── main_log_weight.py             # Weighted-ensemble logging variant
├── distill_naive.py               # Baselines / ablations
├── calc_posrate.py                # Compute per-CoT positive routing rates
├── regen_correct_with_flags.py    # Rebuild correct-only logs from enriched
├── test_dummy_run.py              # Offline test harness (no API calls)
├── data/                          # <you create this> dataset root
├── results/                       # example results dumps
└── plot/                          # plotting assets & notebooks
```

> The archive you received also includes `data/` with placeholders. For real runs, create your own `data/` as described below.

---

## 📦 Datasets & Expected Formats

Create a root folder (default `./data`) with one subfolder **per task**. Typical tasks covered in the code include:
- `gsm8k` (grade-school math word problems)
- `math`  (formal math; LaTeX answers)
- `arc_challenge` (multiple-choice science)
- `commonsense_qa` (MCQ)
- `date` (MCQ)
- `anli` (NLI)
- `strategy_qa` (Yes/No)
- `table_mwp` (table-based math word problems)

Each task directory can include files like `train/`, `dev/`, `test/` or flat JSONL files. The prompting scripts scan for `*.jsonl` files (excluding anything under `train/` by default).

### Minimal JSONL schema per line

Below are **examples**. Field names are what the scripts look for internally.

- **GSM8K / Math**
```json
{"id":"gsm8k-0001", "question":"...", "gold_answer":"\boxed{12}"}
```

- **Multiple-choice (ARC, CSQA, DATE)**
```json
{
  "id":"arc-123",
  "question":"Which ...?",
  "choices": { "label": ["A","B","C","D"], "text": ["opt1","opt2","opt3","opt4"] },
  "gold_answer":"C"
}
```

- **ANLI**
```json
{"id":"anli-42", "premise":"...", "hypothesis":"...", "gold_answer":"entailment"}
```

- **StrategyQA**
```json
{"id":"sqa-7", "question":"Is ...?", "gold_answer":"Yes"}
```

- **Table-MWP**
```json
{"id":"tmwp-9", "table":"<serialized table or path>", "question":"...", "gold_answer":"\boxed{37}"}
```

> The scripts will write two outputs per source file:
> - `*.enriched.jsonl` — original + prompts/responses/predictions/flags
> - `*.correct.jsonl` — only entries for which at least one style was correct

---

## ▶️ Quickstarts

### A) Prompting with multi-style CoT
```bash
python cot_pipeline.py   --root ./data   --n 6 \            # number of reasoning styles per sample
  --model flash      # string you pass into your API caller
```
This scans every `cot_response.jsonl` under each task directory (skipping `train/`), wraps the dataset-specific **core prompts** in several **reasoning-style wrappers**, calls your LLM API in **batches**, and saves `*.enriched.jsonl` and `*.correct.jsonl` alongside the originals.

### B) Prompting with a single powerful wrapper
```bash
python prompt_gen.py   --root ./data   --n 1 \            # ignored; always 1 in this variant
  --model pro
```

### C) Two-student MoE training (LoRA + Accelerate)
> Prepare GPU(s) with enough VRAM. Quantized loading is supported via bitsandbytes.
```bash
accelerate launch --num_processes=1 main.py   --task gsm8k   --model1 mistralai/Mistral-7B-Instruct-v0.3   --model2 google/gemma-7b-it   --epochs 10 --bs 4   --use_kl true --is_router true --is_quality true   --outdir runs/gsm8k
```
Variants:
- `main_log_router.py` — logs per-epoch router decisions to `outdir/router_logs_epoch<e>.jsonl`.
- `main_log_weight.py` — logs weighted-ensemble signals for analysis.

### D) Analyze router positive rates
```bash
python calc_posrate.py --log_dir runs/gsm8k
```

### E) Offline smoke test without network calls
```bash
python test_dummy_run.py
```
This uses a dummy model to patch `batch_call_gemini_api` and drives a mini run end‑to‑end for sanity checks.

---

## 🧩 Implement your API caller

Both `cot_pipeline.py` and `prompt_gen.py` expect a function named **`batch_call_gemini_api(prompts, model_name)`** that returns a list of strings (one response per prompt). A simple OpenAI-based implementation could look like:

```python
# in prompt_gen.py (or a new module you import)
from openai import OpenAI
client = OpenAI()

def batch_call_gemini_api(prompts, model_name="gpt-4o-mini"):
    out = []
    for p in prompts:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content": p}],
            temperature=0.8,
            max_tokens=1000,
        )
        out.append(resp.choices[0].message.content.strip())
    return out
```

> If you use Gemini, mirror the same signature with `google-generativeai`’s async or sync client. `cot_pipeline.py` contains commented snippets showing how it was wired previously.

---

## ✅ Answer extraction & evaluation

- Arithmetic / LaTeX answers are checked via **`math_utils.py`** using SymPy & latex2sympy2, tolerant to boxed answers, units, radicals, percentages, etc.
- Multiple-choice/boolean datasets use helper extractors in **`utils.py`** (e.g., `get_number_choice`, `get_alphabet_choice`, `get_true_false`, `get_yes_no`, `extract_answer_anli`).

Each enriched sample records:
```json
{
  "prompts": [...],
  "responses": [...],
  "preds": [...],               // normalized model answers
  "correct_flags": [true,false] // per-style correctness
}
```

---

## ⚠️ Common gotchas

- **Max output tokens:** When following reverse thinking styles you may need `max_tokens≈1000`. Trim if your provider has strict limits.
- **File scanning:** By default, training JSONLs under `train/` are skipped by the prompting scripts.
- **SymPy parsing:** If LaTeX is malformed, `latex2sympy2` may fail. Consider normalizing answers or catching exceptions.
- **Quantization:** `bitsandbytes` requires CUDA alignment with your PyTorch build. If you hit import errors, install the matching wheels or run in FP16/FP32 without bnb.
- **Accelerate config:** Run `accelerate config` once and pick a config that matches your hardware.

---

## 📊 Reproducibility

- Set `--seed` flags where available to make sampling/initialization repeatable.
- Log dirs under `runs/` capture router decisions and (optionally) ensemble weights for analysis.

---

## 📝 Citation

If you build on this codebase, please cite the repository in your work:

```
@software{distill_repo,
  title        = {Distill: CoT Prompting and Student–Router Distillation},
  year         = {2025},
}
```

---

## 🛠️ License

Unless otherwise stated in the file headers, code in this repository is provided under the Apache 2.0 License.
