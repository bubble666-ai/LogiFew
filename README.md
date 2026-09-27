# LogiFew: Neural-Symbolic Few-Shot Reasoning

LogiFew blends lightweight neural encoders with a differentiable logic module to perform few-shot deductive reasoning (<= 10 examples per rule). The model produces answers, probabilistic proof traces, and induced rules that can later be verified by external provers.

## Highlights
- **Hybrid reasoning stack**: text/video features + Transformer/T5 encoder + differentiable rule memory + probabilistic reasoner.
- **Tiny-data training**: synthetic proof-bank pretraining followed by few-shot adaptation on CLEVRER question–answer pairs.
- **Explainable outputs**: every prediction ships with a proof trace and candidate rules.
- **Hardened + packaged (v0.2.0)**: secure checkpoint loading (`weights_only`), deterministic text hashing, validated configs/metrics, `pyproject.toml` install with CLI entry points, CI.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # runtime deps only (torch, transformers, lightning, torchmetrics, tqdm, pyyaml)
pip install -e .                 # optional: exposes logifew-train/adapt/eval/backtest CLIs

# Optional capability groups
pip install -e ".[hf]"        # HF datasets helpers
pip install -e ".[logic]"     # problog / rdflib formal-logic tooling
pip install -e ".[tracking]"  # wandb experiment tracking
pip install -e ".[test]"      # pytest
```
> Default install is CPU-only PyTorch; install a CUDA wheel if you plan to use GPU.

## Quickstart

### 1. Data Preparation
```bash
# Synthetic subset with symbolic noise
python scripts/build_clevrer_beta_s.py --output_dir data/logifew

# Real CLEVRER annotations (train + validation merged, capped at 400 QA pairs)
python scripts/build_clevrer_real_subset.py \
    --train_file data/logifew/clevrer_train_real.json \
    --extra_train_files data/logifew/clevrer_validation_real.json \
    --test_file data/logifew/clevrer_test_real.json \
    --limit 400
```

### 2. Pretrain & Adapt
```bash
# Phase 1: synthetic proof-bank pretraining
python train.py --config configs/pretrain_synthetic.yaml \
                --output_checkpoint checkpoints/pretrain.ckpt
# or: logifew-train --config configs/pretrain_synthetic.yaml --output_checkpoint checkpoints/pretrain.ckpt

# Phase 2: few-shot adaptation with T5 encoder + early stopping
python adapt_real.py --config configs/adapt_real_hf.yaml \
                     --pretrained checkpoints/pretrain.ckpt \
                     --output_checkpoint checkpoints/nsml_clevrer_real_hf.ckpt
# or: logifew-adapt --config configs/adapt_real_hf.yaml --pretrained checkpoints/pretrain.ckpt ...
```

### 3. Evaluate / Backtest
```bash
python eval_fewshot.py --dataset data/logifew/clevrer_real_train.jsonl \
                       --shots 5 \
                       --metrics EDA,PVR,LCS,DER,RIF1 \
                       --checkpoint checkpoints/nsml_clevrer_real_hf.ckpt
# or: logifew-eval --dataset ... --shots 5 --metrics EDA,PVR,LCS,DER,RIF1 --checkpoint ...

python scripts/backtest_logifew.py --train_dataset data/logifew/clevrer_real_train.jsonl \
                                   --ood_dataset data/logifew/clevrer_real_test.jsonl \
                                   --shots 5 \
                                   --checkpoint checkpoints/nsml_clevrer_real_hf.ckpt
# or: logifew-backtest --train_dataset ... --ood_dataset ... --shots 5 --checkpoint ...
```

## Repository Layout
```
logifew/        Core Python package (data loaders, models, training utilities)
scripts/        CLI helpers for data prep, adaptation, backtesting
configs/        YAML configs (BOW encoder, T5 encoder, etc.)
data/           Generated JSONL datasets live here after running scripts
checkpoints/    Saved model weights + configs (gitignored)
docs/           Documentation (English & Persian summaries, real-world tips)
tests/          Pytest unit tests (incl. security regression tests)
```

## What Changed in v0.2.0 (refactor + security + packaging)
**Bug fixes**
- `configs/fewshot_clevrer.yaml`: removed stray `*** End Patch` trailer that broke `yaml.safe_load`.
- `configs/pretrain_synthetic.yaml`: `encoder.type: text` → `bow` (only `bow`/`hf_text` are implemented); both `train.py` and `adapt_real.py` now validate the value with a clear error.
- `logifew/training/module.py`: one shared `Accuracy` object accumulated train+val batches and corrupted `val_acc` (which drives early stopping/checkpointing) — now separate `train_acc`/`val_acc`.
- `train.py`: nondeterministic `random_split` (no generator) — now seeded from the experiment seed.
- `eval_fewshot.py`: `DER` divided by the full file size instead of the few-shot sample count; `RIF1` copied gold premises into "discovered" (always 1.0) — both fixed; malformed labels no longer `KeyError`; empty datasets raise a clear error.

**Security hardening** (method: Context7 PyTorch serialization docs for `weights_only`, SkillsMP `pytorch-patterns`/`python-patterns`, TypeSafe Jev for packaging/RIF1 decisions)
- Checkpoint loading (`eval_fewshot.py`, `adapt_real.py`) via `logifew/utils/checkpoints.py`: `torch.load(..., weights_only=True)` + `FileNotFoundError` instead of untrusted pickle execution.
- `TextEncoder`: `hash()` (salted per-process, nondeterministic) → SHA-256 bucketing; encodings stable across runs.
- JSONL/annotation loaders: existence checks, size caps, per-line `JSONDecodeError` with line numbers, blank-line tolerance; sample validation (required keys, label allow-list, `premises` type).
- `synthetic_rulebank.generate_rule_bank` validates ranges (counts, probabilities, non-empty vocabularies).

**Packaging**
- New `pyproject.toml` (v0.2.0): runtime deps trimmed to what the code imports (`torch`, `transformers`, `pytorch-lightning`, `torchmetrics`, `tqdm`, `pyyaml` — incl. previously missing `pyyaml`); heavy unused deps (`datasets`, `problog`, `rdflib`, `wandb`, `scikit-learn`) moved to optional extras; CLI entry points (`logifew-train/adapt/eval/backtest/build-beta-s/build-real`); pytest config.
- `requirements.txt` slimmed to match (same runtime set, unpinned for install flexibility).
- `.gitignore` rewritten (was UTF-16 with duplicated stanzas, silently failing to ignore): now covers `__pycache__`, `.pytest_cache`, `.venv`, `checkpoints/`, `*.ckpt`, `lightning_logs/`, `wandb/`, generated `data/logifew` JSON(L), local HF mirrors; all previously tracked `__pycache__/*.pyc` + `.pytest_cache` files untracked.
- New CI: `.github/workflows/tests.yml` (install runtime + pytest, run suite, validate all configs parse).

## Verification
```bash
pytest -q            # 14 passed (incl. new tests/test_security.py)
python scripts/build_clevrer_beta_s.py --output_dir /tmp/logifew_smoke
python eval_fewshot.py --dataset /tmp/logifew_smoke/clevrer_beta_s_train.jsonl --shots 3 --metrics EDA,PVR,LCS,DER,RIF1
# EDA 0.3333 / PVR 0.3333 / LCS 0.9970 / DER 0.0370 (per few-shot N=9) / RIF1 0.0 (honest: synthetic generator emits no arrow-rules)
```

## Metric Semantics (read before comparing runs)
- **DER** = EDA / (# few-shot samples actually evaluated), not / full file size.
- **RIF1** compares proof-trace rule candidates against premise arrow-rules; it is 0 when the split contains no extractable candidates — honest, not a bug.
- **PVR** gates displayed traces on premise support; OOD scores near 0 mean the heuristic prover rejects them (integrate Prover9/Lean for certified scores).

## Roadmap Ideas
1. Plug in Prover9 or Lean to replace the heuristic prover and certify induced rules.
2. Add video features (ViT/ResNet) to reason jointly over text + vision.
3. Log experiments with Weights & Biases (`wandb` is now an optional extra: `pip install -e ".[tracking]"`).
4. Explore parameter-efficient fine-tuning (LoRA, adapters) for larger Transformer backbones.

## License
Released under the MIT License (see [`LICENSE`](LICENSE)).

---

Built as a learning vehicle for neuro-symbolic few-shot reasoning research. Contributions, questions, and experiment reproductions are welcome!

### 🇮🇷 راهنمای فارسی (خلاصه)

**لاگی‌فیو چیست؟** ترکیب انکدر متنی سبک (BOW یا T5) با حافظه قوانین مشتق‌پذیر و استنتاج احتمالی؛ برای استدلال قیاسی با حداکثر ۱۰ مثال برای هر قانون. خروجی هر پیش‌بینی: پاسخ + ردّ اثبات + قوانین پیشنهادی.

**نصب سریع:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .[test]
```

**اجرا (سه گام):** ۱) ساخت داده مصنوعی/واقعی با اسکریپت‌های `scripts/` ۲) پیش‌آموزش `train.py` بعد تطبیق کم‌نمونه `adapt_real.py` ۳) ارزیابی `eval_fewshot.py` و بک‌تست OOD.

**تغییرات نسخه ۰٫۲٫۰:** رفع خرابی YAML، اصلاح نوع انکدر، تفکیک دقت train/val، امن‌سازی بارگذاری چک‌پوینت (`weights_only`)، هش قطعی SHA-256، اعتبارسنجی ورودی‌ها، پکیج `pyproject` با دستورات CLI، تست‌های امنیتی جدید، CI گیت‌هاب. جزئیات کامل در بخش انگلیسی بالا.
