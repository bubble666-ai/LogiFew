"""Lightweight proof validation helpers (stand-ins for Prover9/Lean)."""
from __future__ import annotations

from typing import Iterable, Sequence


def _normalize(token: str) -> str:
    return token.strip().lower().replace(" ", "")


def validate_rule(rule: str, premises: Sequence[str]) -> bool:
    """Heuristic check that a rule's antecedent appears in the given premises."""
    rule = rule.strip()
    if not rule:
        return False
    if "->" not in rule:
        return any(_normalize(rule) in _normalize(p) for p in premises)
    left, _, right = rule.partition("->")
    left = _normalize(left)
    right = _normalize(right)
    return any(left in _normalize(p) for p in premises) and bool(right)


def validate_proof(proof_trace: Sequence[str], premises: Sequence[str]) -> bool:
    """Return True if each rule-like step is supported by the premises."""
    if not proof_trace:
        return False
    normalized_premises = list(premises)
    for step in proof_trace:
        step = step.strip()
        if not step:
            continue
        if "rule" in step.lower() or "->" in step:
            if not validate_rule(step.split(":", 1)[-1] if ":" in step else step, normalized_premises):
                return False
    return True


def filter_valid_rules(candidates: Iterable[str], premises: Sequence[str]) -> list[str]:
    """Return only those rules that pass validation."""
    return [rule for rule in candidates if validate_rule(rule, premises)]
