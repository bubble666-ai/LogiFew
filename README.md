# LogiFew: Neural-Symbolic Few-Shot Reasoning

LogiFew blends lightweight neural encoders with a differentiable logic module to perform few-shot deductive reasoning (<= 10 examples per rule). The model produces answers, probabilistic proof traces, and induced rules that can later be verified by external provers.

## Highlights
- **Hybrid reasoning stack**: text/video features + Transformer/T5 encoder + differentiable rule memory + probabilistic reasoner.
- **Tiny-data training**: synthetic proof-bank pretraining followed by few-shot adaptation on CLEVRER question–answer pairs.
- **Explainable outputs**: every prediction ships with a proof trace and candidate rules.

## Quickstart

### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> Default install is CPU-only PyTorch; install a CUDA wheel if you plan to use GPU.

### 2. Data Preparation
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

### 3. Pretrain & Adapt
```bash
# Phase 1: synthetic proof-bank pretraining
python train.py --config configs/pretrain_synthetic.yaml \
                --output_checkpoint checkpoints/pretrain.ckpt

# Phase 2: few-shot adaptation with T5 encoder + early stopping
python adapt_real.py --config configs/adapt_real_hf.yaml \
                     --pretrained checkpoints/pretrain.ckpt \
                     --output_checkpoint checkpoints/nsml_clevrer_real_hf.ckpt
```

### 4. Evaluate / Backtest
```bash
python eval_fewshot.py --dataset data/logifew/clevrer_real_train.jsonl \
                       --shots 5 \
                       --metrics EDA,PVR,LCS,DER,RIF1 \
                       --checkpoint checkpoints/nsml_clevrer_real_hf.ckpt

python scripts/backtest_logifew.py --train_dataset data/logifew/clevrer_real_train.jsonl \
                                   --ood_dataset data/logifew/clevrer_real_test.jsonl \
                                   --shots 5 \
                                   --checkpoint checkpoints/nsml_clevrer_real_hf.ckpt
```

## Repository Layout
```
logifew/        Core Python package (data loaders, models, training utilities)
scripts/        CLI helpers for data prep, adaptation, backtesting
configs/        YAML configs (BOW encoder, T5 encoder, etc.)
data/           Generated JSONL datasets live here after running scripts
checkpoints/    Saved model weights + configs
docs/           Documentation (English & Persian summaries, real-world tips)
tests/          Pytest unit tests
```

## What's New in the Latest Iteration?
- **Early stopping + best-checkpoint saving** to avoid overfitting when validation accuracy drops.
- **Practical hyperparameters**: LR defaults to `1e-4` and dropout to `0.1/0.3` for the T5 setup (tune around these values as needed).
- **Shuffled train/validation split** for more reliable validation statistics.
- **Simple proof validator** (`logifew/utils/prover.py`) that filters out rules/proofs unsupported by the given premises.

## Current Status
- With the T5-small encoder and the current CLEVRER subset, LogiFew reaches ~0.33 EDA on few-shot train and 1.0 on the small OOD split (proof validity is still heuristic on OOD).
- Real visual features (ViT/ResNet) are not yet integrated; only textual annotations are used.

## Roadmap Ideas
1. Plug in Prover9 or Lean to replace the heuristic prover and certify induced rules.
2. Add video features (ViT/ResNet) to reason jointly over text + vision.
3. Log experiments with Weights & Biases or similar tooling.
4. Explore parameter-efficient fine-tuning (LoRA, adapters) for larger Transformer backbones.

## Latest Evaluation Snapshot
```
Few-shot train (5-shot):
  EDA = 0.3333
  PVR = 1.0000
  LCS = 0.9932
  DER = 0.0003
  RIF1 = 1.0000

OOD split (5-shot):
  EDA = 1.0000
  PVR = 0.0000  # heuristic prover rejects proofs; integrate a formal prover for reliable scores
  LCS = 0.9931
  DER = 0.0008
  RIF1 = 1.0000
```

## License
Released under the MIT License (see [`LICENSE`](LICENSE)).

---

Built as a learning vehicle for neuro-symbolic few-shot reasoning research. Contributions, questions, and experiment reproductions are welcome!
