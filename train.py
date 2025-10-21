#!/usr/bin/env python
"""Training entry-point for LogiFew."""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader, random_split

from logifew.data.datasets import ClevrerBetaSDataset, TextEncoder, collate_fn
from logifew.data.synthetic_rulebank import generate_rule_bank, to_training_examples
from logifew.models.nsml import NSMLConfig
from logifew.training.module import NSMLLightningModule, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NSML model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override maximum epochs.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices.")
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        default="checkpoints/pretrain.ckpt",
        help="Path to save the trained model weights.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config["data"]["synthetic_rulebank"]
    clauses = generate_rule_bank(
        num_proofs=dataset_cfg.get("num_proofs", 500),
        predicates=dataset_cfg.get("predicates", []),
        variables=dataset_cfg.get("variables", []),
        max_body_literals=dataset_cfg.get("max_body_literals", 4),
        noise_probability=dataset_cfg.get("noise_probability", 0.1),
        negative_ratio=dataset_cfg.get("negative_ratio", 0.5),
    )
    items = to_training_examples(clauses)

    dataset = ClevrerBetaSDataset(items, encoder=TextEncoder())
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    batch_size = config["training"].get("batch_size", 16)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    seed_everything(cfg["experiment"].get("seed", 42), workers=True)

    train_loader, val_loader = build_dataloaders(cfg)

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
    training_cfg = TrainingConfig(learning_rate=cfg["training"].get("learning_rate", 3e-4))

    lightning_module = NSMLLightningModule(nsml_config, training_cfg)

    max_epochs = args.max_epochs or cfg["training"].get("max_epochs", 3)
    trainer = Trainer(
        accelerator="cpu",
        devices=args.devices,
        max_epochs=max_epochs,
        log_every_n_steps=cfg["training"].get("log_every_n_steps", 10),
    )

    trainer.fit(lightning_module, train_loader, val_loader)

    if args.output_checkpoint:
        checkpoint_path = Path(args.output_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict

        torch.save(
            {
                "state_dict": lightning_module.model.state_dict(),
                "model_config": asdict(lightning_module.model.config),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
