"""Utility helpers for constructing proof traces."""
from __future__ import annotations

from typing import List


def assemble_proof(query: str, rules: List[str], facts: List[str], confidence: float) -> List[str]:
    """Create a human-readable proof skeleton."""
    proof = [f"Query: {query}"]
    for idx, rule in enumerate(rules):
        proof.append(f"├─ Rule {idx + 1}: {rule}")
    for fact in facts:
        proof.append(f"├─ Fact: {fact}")
    proof.append(f"└─ ∴ {query} [prob: {confidence:.2f}]")
    return proof
