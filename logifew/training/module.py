"""PyTorch Lightning module wrapping NSML."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from logifew.models.nsml import NSMLConfig, NSMLModel


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    label_smoothing: float = 0.0


class NSMLLightningModule(LightningModule):
    """High-level training harness for NSML."""

    def __init__(self, model_config: Optional[NSMLConfig] = None, training_config: Optional[TrainingConfig] = None) -> None:
        super().__init__()
        self.model = NSMLModel(model_config)
        self.training_config = training_config or TrainingConfig()
        self.metric_acc = Accuracy(task="binary")

    def forward(
        self,
        premises: torch.Tensor,
        queries: torch.Tensor,
        premises_text: Optional[Sequence[Sequence[str]]] = None,
        queries_text: Optional[Sequence[str]] = None,
    ) -> dict:
        return self.model(
            premises,
            queries,
            premises_text=premises_text,
            queries_text=queries_text,
        )

    def _calc_step(self, batch: dict, stage: str) -> torch.Tensor:
        premises, queries, labels = batch["premises"], batch["queries"], batch["labels"]
        premises_text = batch.get("premises_text")
        queries_text = batch.get("queries_text")
        outputs = self.forward(premises, queries, premises_text=premises_text, queries_text=queries_text)
        losses = self.model.compute_losses(
            premises,
            queries,
            premises_text=premises_text or [],
            queries_text=queries_text or [],
            outputs=outputs,
            labels=labels,
        )
        self.log(f"{stage}_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        if stage != "test":
            predictions = (outputs["probabilities"] > 0.5).long()
            target = (labels == 1).long()
            acc = self.metric_acc(predictions, target)
            self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return losses["total_loss"]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._calc_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._calc_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._calc_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        return optimizer
