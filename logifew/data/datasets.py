"""Dataset utilities for LogiFew."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

LABEL_TO_INDEX = {"yes": 1, "no": 0, "unknown": 2}


def _read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


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
    """Simple whitespace tokenizer with hashing to a fixed vocabulary."""

    def __init__(self, vocab_size: int = 2048, hash_seed: int = 17) -> None:
        self.vocab_size = vocab_size
        self.hash_seed = hash_seed

    def encode(self, text: str, max_len: int = 64) -> torch.Tensor:
        tokens = text.lower().split()
        vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        for token in tokens[:max_len]:
            idx = (hash((token, self.hash_seed)) % self.vocab_size)
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
        premises = sample.get("premises", [])
        encoded_premises = sum(
            (self.encoder.encode(p) for p in premises[: self.max_premises]),
            torch.zeros(self.encoder.vocab_size, dtype=torch.float32),
        )
        query = self.encoder.encode(sample["query"])
        label = LABEL_TO_INDEX[sample["label"]]
        return EncodedExample(
            premises=encoded_premises,
            query=query,
            label=torch.tensor(label, dtype=torch.long),
            premises_text=[str(p) for p in premises],
            query_text=str(sample["query"]),
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
