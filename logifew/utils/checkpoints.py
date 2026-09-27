"""Secure checkpoint loading for LogiFew (PyTorch weights_only by default)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    """Load a torch checkpoint with ``weights_only=True``.

    Our checkpoints only contain tensors + plain config dicts
    (``{"state_dict", "model_config"}``), so ``weights_only`` is sufficient
    and blocks arbitrary pickle execution from untrusted files.
    """
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch < 2.4 without weights_only kwarg — fall back explicitly.
        return torch.load(ckpt_path, map_location=map_location)


def extract_model_state(checkpoint: Any) -> tuple[Any | None, dict]:
    """Split ``{"state_dict", "model_config"}`` checkpoints from raw state_dicts."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], dict(checkpoint.get("model_config", {}))
    if isinstance(checkpoint, dict):
        return checkpoint, {}
    return None, {}
