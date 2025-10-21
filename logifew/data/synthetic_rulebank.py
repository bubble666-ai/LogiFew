"""Synthetic first-order proof generation for LogiFew pretraining."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class SyntheticClause:
    head: str
    body: List[str]
    label: int
    reasoning_chain: List[str]


def _sample_predicate(rng: random.Random, predicates: Sequence[str], variables: Sequence[str]) -> str:
    predicate = rng.choice(predicates)
    arity = rng.randint(1, min(3, len(variables)))
    args = rng.sample(variables, k=arity)
    return f"{predicate}({', '.join(args)})"


def _corrupt_literal(rng: random.Random, literal: str, predicates: Sequence[str]) -> str:
    parts = literal.split("(")
    if len(parts) != 2:
        return literal
    predicate = parts[0]
    replacement = rng.choice([p for p in predicates if p != predicate]) if len(predicates) > 1 else predicate
    return replacement + "(" + parts[1]


def generate_rule_bank(
    num_proofs: int,
    predicates: Sequence[str],
    variables: Sequence[str],
    max_body_literals: int,
    noise_probability: float,
    negative_ratio: float,
    seed: int = 23,
) -> List[SyntheticClause]:
    """Create a set of labelled clauses with reasoning chains."""
    rng = random.Random(seed)
    clauses: List[SyntheticClause] = []

    for proof_idx in range(num_proofs):
        body_len = rng.randint(1, max(1, max_body_literals))
        body_literals = [_sample_predicate(rng, predicates, variables) for _ in range(body_len)]
        head_literal = _sample_predicate(rng, predicates, variables)
        label = 1
        reasoning = [f"fact:{literal}" for literal in body_literals]

        if rng.random() < noise_probability:
            corrupt_idx = rng.randrange(len(body_literals))
            body_literals[corrupt_idx] = _corrupt_literal(rng, body_literals[corrupt_idx], predicates)
            reasoning.append(f"noise:introduced@{corrupt_idx}")

        if rng.random() < negative_ratio:
            label = 0
            reasoning.append("label:negative")
        else:
            reasoning.append("label:positive")

        clauses.append(
            SyntheticClause(
                head=head_literal,
                body=body_literals,
                label=label,
                reasoning_chain=reasoning,
            )
        )

    return clauses


def to_training_examples(clauses: Iterable[SyntheticClause]) -> List[Dict[str, object]]:
    """Convert clauses into serialisable dictionaries."""
    return [
        {
            "premises": clause.body,
            "query": clause.head,
            "label": "yes" if clause.label == 1 else "no",
            "proof_trace": clause.reasoning_chain,
        }
        for clause in clauses
    ]
