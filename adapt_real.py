#!/usr/bin/env python
"""Few-shot adaptation of NSML on real CLEVRER data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from logifew.data.datasets import ClevrerBetaSDataset, TextEncoder, collate_fn, load_jsonl_dataset
from logifew.models.nsml import NSMLConfig
from logifew.training.module import NSMLLightningModule, TrainingConfig
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt NSML model on CLEVRER real subset.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained checkpoint.")
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        default="checkpoints/nsml_clevrer_real.ckpt",
        help="Where to store adapted model weights.",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Override dataset size limit.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloaders(cfg: dict, limit_override: int | None = None) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    items = load_jsonl_dataset(Path(data_cfg["path"]))
    limit = limit_override if limit_override is not None else data_cfg.get("limit")
    if limit is not None:
        items = items[:limit]
    seed = cfg["experiment"].get("seed", 0)
    rng = random.Random(seed)
    rng.shuffle(items)
    dataset = ClevrerBetaSDataset(items, encoder=TextEncoder())
    val_split = data_cfg.get("val_split", 0.2)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    batch_size = data_cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    seed_everything(cfg["experiment"].get("seed", 0), workers=True)

    train_loader, val_loader = build_dataloaders(cfg, args.limit)

    training_cfg = TrainingConfig(
        learning_rate=cfg["training"].get("learning_rate", 2e-4),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    model_cfg = cfg.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    rule_cfg = model_cfg.get("rule_memory", {})
    reasoner_cfg = model_cfg.get("reasoner", {})
    nsml_config = NSMLConfig(
        input_dim=encoder_cfg.get("input_dim", NSMLConfig.input_dim),
        hidden_size=encoder_cfg.get("hidden_size", NSMLConfig.hidden_size),
        dropout=encoder_cfg.get("dropout", NSMLConfig.dropout),
        encoder_type=encoder_cfg.get("type", NSMLConfig.encoder_type),
        hf_model_name=encoder_cfg.get("hf_model_name", NSMLConfig.hf_model_name),
        hf_max_length=encoder_cfg.get("hf_max_length", NSMLConfig.hf_max_length),
        hf_local_files_only=encoder_cfg.get("hf_local_files_only", NSMLConfig.hf_local_files_only),
        num_rules=rule_cfg.get("num_rules", NSMLConfig.num_rules),
        key_dim=rule_cfg.get("key_dim", NSMLConfig.key_dim),
        value_dim=rule_cfg.get("value_dim", NSMLConfig.value_dim),
        rule_dropout=rule_cfg.get("dropout", NSMLConfig.rule_dropout),
        temperature=reasoner_cfg.get("temperature", NSMLConfig.temperature),
        proof_depth=reasoner_cfg.get("proof_depth", NSMLConfig.proof_depth),
    )
    module = NSMLLightningModule(nsml_config, training_cfg)

    pretrained_path = args.pretrained or cfg.get("model", {}).get("checkpoint", "")
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        state_to_load = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        load_result = module.model.load_state_dict(state_to_load, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
        missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
        if missing:
            print(f"Missing keys ignored ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"Unexpected keys ignored ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=3, verbose=True)
    checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="nsml-best")

    trainer = Trainer(
        accelerator="cpu",
        devices=args.devices,
        max_epochs=args.max_epochs or cfg["training"].get("max_epochs", 5),
        log_every_n_steps=cfg["training"].get("log_every_n_steps", 10),
        callbacks=[early_stop, checkpoint_cb],
    )
    trainer.fit(module, train_loader, val_loader)

    if args.output_checkpoint:
        output_path = Path(args.output_checkpoint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict

        torch.save(
            {
                "state_dict": module.model.state_dict(),
                "model_config": asdict(module.model.config),
            },
            output_path,
        )
        print(f"Saved adapted checkpoint to {output_path}")


if __name__ == "__main__":
    main()
