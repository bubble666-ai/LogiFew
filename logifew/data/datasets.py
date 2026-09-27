"""Dataset utilities for LogiFew."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

LABEL_TO_INDEX = {"yes": 1, "no": 0, "unknown": 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def _read_jsonl(path: Path, max_lines: int = 100_000) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if lineno > max_lines:
                raise ValueError(f"Dataset {path} exceeds {max_lines} lines — refusing to load fully into memory")
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {lineno}: {exc}") from exc
    return rows


def load_jsonl_dataset(path: Path, limit: int | None = None) -> List[dict]:
    """Load a JSONL dataset into memory."""
    data = _read_jsonl(path)
    if limit is not None:
        data = data[:limit]
    return data


@dataclass
class EncodedExample:
    premises: torch.Tensor
    query: torch.Tensor
    label: torch.Tensor
    premises_text: List[str]
    query_text: str
    metadata: dict


class TextEncoder:
    """Simple whitespace tokenizer with hashing to a fixed vocabulary.

    Uses SHA-256 (not builtin ``hash()``) so encodings are stable across
    runs regardless of ``PYTHONHASHSEED`` — required for reproducible
    few-shot splits and cached datasets.
    """

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len: int = 64) -> torch.Tensor:
        tokens = text.lower().split()
        vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        for token in tokens[:max_len]:
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.vocab_size
            vector[idx] += 1.0
        if vector.norm(p=2) > 0:
            vector = vector / vector.norm(p=2)
        return vector


class ClevrerBetaSDataset(Dataset):
    """Thin wrapper that converts CLEVRER-beta_s JSON entries into tensors."""

    def __init__(
        self,
        items: Sequence[dict],
        encoder: TextEncoder | None = None,
        max_premises: int = 4,
    ) -> None:
        self.items = list(items)
        self.encoder = encoder or TextEncoder()
        self.max_premises = max_premises

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedExample:
        sample = self.items[index]
        try:
            query_text = str(sample["query"])
            label = LABEL_TO_INDEX[sample["label"]]
        except KeyError as exc:
            raise KeyError(f"Sample {index} missing required key {exc}; keys={sorted(sample)})") from exc
        if sample["label"] not in LABEL_TO_INDEX:
            raise ValueError(f"Sample {index} has unknown label {sample['label']!r}")
        premises = sample.get("premises", [])
        if not isinstance(premises, list):
            raise ValueError(f"Sample {index} 'premises' must be a list, got {type(premises).__name__}")
        encoded_premises = sum(
            (self.encoder.encode(str(p)) for p in premises[: self.max_premises]),
            torch.zeros(self.encoder.vocab_size, dtype=torch.float32),
        )
        query = self.encoder.encode(query_text)
        return EncodedExample(
            premises=encoded_premises,
            query=query,
            label=torch.tensor(label, dtype=torch.long),
            premises_text=[str(p) for p in premises],
            query_text=query_text,
            metadata=sample.get("meta", {}),
        )


def collate_fn(batch: Sequence[EncodedExample]) -> dict:
    premises = torch.stack([item.premises for item in batch], dim=0)
    queries = torch.stack([item.query for item in batch], dim=0)
    labels = torch.stack([item.label for item in batch], dim=0)
    premises_text = [item.premises_text for item in batch]
    queries_text = [item.query_text for item in batch]
    metadata = [item.metadata for item in batch]
    return {
        "premises": premises,
        "queries": queries,
        "labels": labels,
        "premises_text": premises_text,
        "queries_text": queries_text,
        "metadata": metadata,
    }
