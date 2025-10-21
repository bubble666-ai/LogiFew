# LogiFew Documentation

LogiFew is a neuro-symbolic framework for deductive reasoning in few-shot regimes (<= 10 examples per rule). It combines lightweight neural encoders, a differentiable rule memory, and a probabilistic inference engine to produce answers, proof traces, and induced rule candidates.

## Architecture and Components
- **Synthetic Rule Bank:** `logifew/data/synthetic_rulebank.py` fabricates first-order proofs for pre-training.
- **NSML Model:** Core modules (encoder, memory, reasoner) live in `logifew/models/`. The encoder can be a classic bag-of-words projection or a HuggingFace Transformer (T5, etc.).
- **Training:** `train.py` handles synthetic pre-training via PyTorch Lightning and emits a base checkpoint.
- **Real CLEVRER Bridge:** `scripts/build_clevrer_real_subset.py` converts the official CLEVRER annotations into LogiFew-compatible JSONL files.
- **Evaluation & Backtesting:** `eval_fewshot.py` and `scripts/backtest_logifew.py` report metrics (EDA, PVR, LCS, DER, RIF1).

## Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Python 3.10+ and a suitable PyTorch build are available.

## Data Preparation
- **Synthetic subset with symbolic noise**
  ```bash
  python scripts/build_clevrer_beta_s.py --output_dir data/logifew
  ```
  Produces `clevrer_beta_s_train.jsonl` and `clevrer_beta_s_ood.jsonl`.

- **Real CLEVRER annotations (train + validation merged)**
  ```bash
  python scripts/build_clevrer_real_subset.py \
      --train_file data/logifew/clevrer_train_real.json \
      --extra_train_files data/logifew/clevrer_validation_real.json \
      --test_file data/logifew/clevrer_test_real.json \
      --limit 400
  ```
  Generates `clevrer_real_train.jsonl` and `clevrer_real_test.jsonl` with up to 400 QA pairs (increase the limit if you have more balanced data).

## Pre-Training on Synthetic Proofs
1. Inspect `configs/pretrain_synthetic.yaml`.
2. Launch training and save the checkpoint:
   ```bash
   python train.py --config configs/pretrain_synthetic.yaml \
                   --output_checkpoint checkpoints/pretrain.ckpt
   ```

## Few-Shot Adaptation on Real CLEVRER
1. Review `configs/adapt_real.yaml`.
2. Fine-tune using the synthetic checkpoint:
   ```bash
   python adapt_real.py --config configs/adapt_real.yaml \
                        --pretrained checkpoints/pretrain.ckpt \
                        --output_checkpoint checkpoints/nsml_clevrer_real.ckpt
   ```

## Switching to Transformer Encoders (T5/Vit-style)
1. Ensure the desired HuggingFace model weights are available locally (e.g., `t5-small`).
2. Use the provided config template `configs/adapt_real_hf.yaml` (note the `encoder.type: hf_text` block).
3. Adapt with the Transformer encoder:
   ```bash
   python adapt_real.py --config configs/adapt_real_hf.yaml \
                        --pretrained checkpoints/pretrain.ckpt \
                        --output_checkpoint checkpoints/nsml_clevrer_real_hf.ckpt
   ```
   This routes raw text through the specified Transformer instead of the bag-of-words projection. For ViT-style inputs, extend the dataset to supply frame tensors/features and set `encoder.type` accordingly.

## Evaluation Workflows
- **Direct evaluation**
  ```bash
  python eval_fewshot.py --dataset data/logifew/clevrer_real_train.jsonl \
                         --shots 5 \
                         --metrics EDA,PVR,LCS,DER,RIF1 \
                         --checkpoint checkpoints/nsml_clevrer_real.ckpt
  ```

- **IID vs. OOD backtest**
  ```bash
  python scripts/backtest_logifew.py --shots 5 \
                                     --checkpoint checkpoints/nsml_clevrer_real.ckpt
  ```
  Override `--train_dataset` / `--ood_dataset` to test other splits.

## Automated Tests
```bash
pytest
```
Validates dataset builders, metrics, evaluation helpers, and the NSML forward/loss path.

## Project Structure
```
logifew/           Core package: data, models, training helpers
scripts/           Data builders, adaptation runner, backtest driver
configs/           YAML configs for synthetic and real pipelines
tests/             Unit tests
data/logifew/      Generated datasets (after running scripts)
checkpoints/       Saved model weights
```

## Next Steps
- Swap the simple encoder with stronger text/vision backbones (e.g., T5, ViT) using features from the CLEVRER_VQA_ResNet_Bert release.
- Replace the heuristic prover in `logifew/utils/prover.py` with a full Prover9/Lean integration to verify induced rules.
- Log experiments with Weights & Biases (`wandb.init(...)`) once credentials are configured.

## Tips for Real-World Reliability
- **Early stopping + best checkpoint:** The adaptation script includes early stopping (patience = 3 by default) and saves the best `val_acc` checkpoint. Tweak patience if your validation set is noisy.
- **Learning rate & dropout:** `configs/adapt_real_hf.yaml` currently sets LR to `1e-4` with encoder/rule-memory dropout `0.1/0.3`. If learning is unstable, adjust LR by a factor of ~2 and compensate with dropout or weight decay.
- **Balanced splits:** Training/validation sets are reshuffled using the experiment seed and sampled with a deterministic generator to avoid repeated samples in validation.
- **Visual features:** If you have access to CLEVRER frames, extract visual embeddings (ViT/ResNet) and feed them through a multimodal encoder for stronger generalisation.
- **Formal verification:** The bundled prover stub is heuristic; integrate Prover9 or Lean for certified proofs and trustworthy RIF1 scores.
- **Experiment tracking:** Keep track of seeds, configs, and metrics with W&B or similar tools, especially when comparing encoder types or data mixes.
