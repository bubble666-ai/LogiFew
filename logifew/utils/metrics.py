"""Evaluation metrics for LogiFew."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def exact_deduction_accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for pred, gold in zip(predictions, labels) if pred == gold)
    return matches / len(predictions)


def proof_validity_rate(proof_traces: Iterable[Sequence[str]]) -> float:
    traces = list(proof_traces)
    if not traces:
        return 0.0
    valid = sum(1 for trace in traces if trace and all(isinstance(step, str) for step in trace))
    return valid / len(traces)


def logical_consistency_score(predicted_prob: Sequence[float], symbolic_prob: Sequence[float]) -> float:
    if not predicted_prob:
        return 0.0
    diffs = [abs(p - s) for p, s in zip(predicted_prob, symbolic_prob)]
    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff)


def rule_induction_f1(discovered: Iterable[str], gold: Iterable[str]) -> float:
    discovered_set = set(discovered)
    gold_set = set(gold)
    if not discovered_set and not gold_set:
        return 1.0
    if not discovered_set or not gold_set:
        return 0.0
    precision = len(discovered_set & gold_set) / len(discovered_set)
    recall = len(discovered_set & gold_set) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def data_efficiency_ratio(eda: float, num_samples: int) -> float:
    if num_samples <= 0:
        return 0.0
    return eda / num_samples
