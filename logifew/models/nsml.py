"""Top-level Neuro-Symbolic Meta-Learner (NSML) model definition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .encoder import ProjectionEncoder
from .reasoner import ProbabilisticReasoner
from .rule_memory import RuleMemoryBank


@dataclass
class NSMLConfig:
    input_dim: int = 2048
    hidden_size: int = 256
    dropout: float = 0.1
    encoder_type: str = "bow"  # bow | hf_text
    hf_model_name: str = "t5-small"
    hf_max_length: int = 128
    hf_local_files_only: bool = False
    num_rules: int = 64
    key_dim: int = 128
    value_dim: int = 128
    rule_dropout: float = 0.3
    temperature: float = 0.67
    proof_depth: int = 3
    beta_contrastive: float = 0.3
    lambda_consistency: float = 0.7


class NSMLModel(nn.Module):
    """Minimal NSML instantiation for experimentation and testing."""

    def __init__(self, config: NSMLConfig | None = None) -> None:
        super().__init__()
        self.config = config or NSMLConfig()
        self.encoder_type = self.config.encoder_type
        if self.encoder_type == "bow":
            self.encoder = ProjectionEncoder(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_size,
                dropout=self.config.dropout,
            )
            self.hf_tokenizer = None
            self.hf_model = None
            self.hf_projection = None
        elif self.encoder_type == "hf_text":
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError("transformers is required for hf_text encoder") from exc
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.config.hf_model_name, local_files_only=self.config.hf_local_files_only
            )
            self.hf_model = AutoModel.from_pretrained(
                self.config.hf_model_name, local_files_only=self.config.hf_local_files_only
            )
            hidden_dim = getattr(self.hf_model.config, "hidden_size", self.config.hidden_size)
            self.hf_projection = nn.Sequential(
                nn.Linear(hidden_dim, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
        else:
            raise ValueError(f"Unsupported encoder_type '{self.encoder_type}'")
        self.rule_memory = RuleMemoryBank(
            num_rules=self.config.num_rules,
            key_dim=self.config.key_dim,
            value_dim=self.config.value_dim,
            dropout=self.config.rule_dropout,
        )
        self.query_adapter = nn.Linear(self.config.hidden_size, self.config.key_dim)
        self.value_adapter = nn.Linear(self.config.value_dim, self.config.hidden_size)
        self.reasoner = ProbabilisticReasoner(
            hidden_dim=self.config.hidden_size,
            temperature=self.config.temperature,
            proof_depth=self.config.proof_depth,
        )

    def _encode_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        if self.hf_tokenizer is None or self.hf_model is None or self.hf_projection is None:
            raise RuntimeError("HF tokenizer/model not initialised")
        tokenized = self.hf_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.hf_max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        if hasattr(self.hf_model, "encoder") and not hasattr(self.hf_model, "pooler"):
            encoder_kwargs = {"input_ids": tokenized["input_ids"], "return_dict": True}
            if "attention_mask" in tokenized:
                encoder_kwargs["attention_mask"] = tokenized["attention_mask"]
            encoder_outputs = self.hf_model.encoder(**encoder_kwargs)
            hidden_states = encoder_outputs.last_hidden_state
        else:
            outputs = self.hf_model(**tokenized, return_dict=True)
            hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.mean(dim=1)
        return self.hf_projection(hidden_states)

    def encode_inputs(
        self,
        premises: torch.Tensor,
        queries: torch.Tensor,
        premises_text: Optional[Sequence[Sequence[str]]] = None,
        queries_text: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == "bow":
            encoded_premises = self.encoder(premises)
            encoded_queries = self.encoder(queries)
        elif self.encoder_type == "hf_text":
            if premises_text is None or queries_text is None:
                raise ValueError("premises_text and queries_text required for hf_text encoder")
            device = premises.device
            flat_premises = [" ".join(p) if isinstance(p, (list, tuple)) else str(p) for p in premises_text]
            encoded_premises = self._encode_texts(flat_premises, device)
            encoded_queries = self._encode_texts(queries_text, device)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        return encoded_premises, encoded_queries

    def forward(
        self,
        premises: torch.Tensor,
        queries: torch.Tensor,
        premises_text: Optional[Sequence[Sequence[str]]] = None,
        queries_text: Optional[Sequence[str]] = None,
    ) -> dict:
        encoded_premises, encoded_queries = self.encode_inputs(
            premises,
            queries,
            premises_text=premises_text,
            queries_text=queries_text,
        )
        memory_values = self.rule_memory(self.query_adapter(encoded_queries))
        memory_values = self.value_adapter(memory_values)
        result = self.reasoner(encoded_premises, memory_values)
        return {
            **result,
            "encoded_premises": encoded_premises,
            "encoded_queries": encoded_queries,
        }

    def compute_losses(
        self,
        premises: torch.Tensor,
        queries: torch.Tensor,
        premises_text: Sequence[Sequence[str]],
        queries_text: Sequence[str],
        outputs: dict,
        labels: torch.Tensor,
        corrupted_queries: Optional[torch.Tensor] = None,
        corrupted_queries_text: Optional[Sequence[str]] = None,
    ) -> dict:
        target = torch.where(labels == 1, 1.0, torch.where(labels == 0, 0.0, 0.5)).to(outputs["probabilities"].device)
        entailment_loss = F.mse_loss(outputs["probabilities"], target)

        consistency_penalty = torch.mean(torch.clamp(outputs["probabilities"] - 1.0, min=0) ** 2)
        consistency_penalty += torch.mean(torch.clamp(-outputs["probabilities"], min=0) ** 2)
        consistency_loss = self.config.lambda_consistency * consistency_penalty

        contrastive_loss = torch.tensor(0.0, device=entailment_loss.device)
        if corrupted_queries is not None:
            with torch.no_grad():
                corrupted_outputs = self.forward(
                    premises,
                    corrupted_queries,
                    premises_text=premises_text,
                    queries_text=corrupted_queries_text,
                )
            similarities = F.cosine_similarity(
                outputs["encoded_queries"], outputs["encoded_premises"], dim=-1
            )
            corrupted_sim = F.cosine_similarity(
                corrupted_outputs["encoded_queries"], outputs["encoded_premises"], dim=-1
            )
            contrastive_loss = -torch.log(torch.sigmoid(similarities - corrupted_sim) + 1e-6).mean()
            contrastive_loss = self.config.beta_contrastive * contrastive_loss

        total = entailment_loss + consistency_loss + contrastive_loss
        return {
            "total_loss": total,
            "entailment_loss": entailment_loss,
            "consistency_loss": consistency_loss,
            "contrastive_loss": contrastive_loss,
        }
