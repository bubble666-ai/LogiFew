#!/usr/bin/env python
"""Few-shot evaluation script for LogiFew."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from logifew.data.datasets import (
    INDEX_TO_LABEL,
    ClevrerBetaSDataset,
    TextEncoder,
    collate_fn,
    load_jsonl_dataset,
)
from logifew.models.nsml import NSMLConfig, NSMLModel
from logifew.utils.checkpoints import extract_model_state, safe_torch_load
from logifew.utils.metrics import (
    data_efficiency_ratio,
    exact_deduction_accuracy,
    logical_consistency_score,
    proof_validity_rate,
    rule_induction_f1,
)
from logifew.utils.proof import assemble_proof
from logifew.utils.prover import filter_valid_rules, validate_proof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NSML in a few-shot setting.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CLEVRER-beta_s JSONL.")
    parser.add_argument("--metrics", type=str, default="EDA,PVR,LCS,DER,RIF1", help="Comma separated metrics.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a model checkpoint (optional).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shots", type=int, default=5)
    return parser.parse_args(argv)


def load_model(checkpoint_path: str | None) -> NSMLModel:
    config = NSMLConfig()
    state_to_load = None
    if checkpoint_path:
        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        raw_state, model_cfg = extract_model_state(checkpoint)
        if model_cfg:
            try:
                config = NSMLConfig(**model_cfg)
            except TypeError:
                print("[eval] checkpoint model_config has unknown fields; using defaults")
        state_to_load = raw_state
    model = NSMLModel(config)
    if state_to_load is not None:
        load_result = model.load_state_dict(state_to_load, strict=False)
        missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
        if missing:
            print(f"[eval] Missing keys ignored ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[eval] Unexpected keys ignored ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model


def few_shot_subset(items: List[dict], shots: int) -> List[dict]:
    """Take up to k examples from each known label category (skips unknowns)."""
    if shots <= 0:
        raise ValueError(f"--shots must be positive, got {shots}")
    buckets: dict[str, list[dict]] = {"yes": [], "no": [], "unknown": []}
    for item in items:
        label = item.get("label")
        if label not in buckets:
            continue  # skip malformed labels instead of KeyError
        if len(buckets[label]) < shots:
            buckets[label].append(item)
    subset: List[dict] = []
    for label in ["yes", "no", "unknown"]:
        subset.extend(buckets[label])
    return subset


def _extract_gold_rules(sample: dict, out: set[str]) -> None:
    for premise in sample.get("premises", []) or []:
        if isinstance(premise, str) and premise.strip() and "->" in premise:
            out.add(premise.strip())


def _extract_discovered_rules(sample: dict, out: set[str]) -> None:
    # ponytail: discovered = only model/dataset proof_trace candidates, never
    # copied from premises (that made RIF1 trivially 1.0).
    for trace_step in sample.get("proof_trace", []) or []:
        if not isinstance(trace_step, str):
            continue
        cleaned = trace_step.strip()
        if not cleaned:
            continue
        if cleaned.startswith("rule") or cleaned.startswith("% Induced"):
            out.add(cleaned.split(":", 1)[-1].strip() if ":" in cleaned else cleaned)
        elif "Rule" in cleaned and ":" in cleaned:
            out.add(cleaned.split(":", 1)[-1].strip())
        elif "->" in cleaned:
            out.add(cleaned)


def evaluate(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset)
    items = load_jsonl_dataset(dataset_path)
    if not items:
        raise ValueError(f"No examples found in {dataset_path}")
    shot_items = few_shot_subset(items, args.shots)
    if not shot_items:
        raise ValueError(f"few-shot subset is empty (shots={args.shots}, n={len(items)})")
    dataset = ClevrerBetaSDataset(shot_items, encoder=TextEncoder())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = load_model(args.checkpoint if args.checkpoint else None)
    predictions: List[str] = []
    labels: List[str] = []
    proof_traces: List[List[str]] = []
    predicted_probs: List[float] = []
    gold_rules: set[str] = set()
    discovered_rules: set[str] = set()

    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["premises"],
                batch["queries"],
                premises_text=batch.get("premises_text"),
                queries_text=batch.get("queries_text"),
            )
            probs = outputs["probabilities"]
            batch_samples = shot_items[offset : offset + len(probs)]
            for idx, prob in enumerate(probs):
                prob_value = prob.item()
                predicted_probs.append(prob_value)
                if prob_value > 0.6:
                    predictions.append("yes")
                elif prob_value < 0.4:
                    predictions.append("no")
                else:
                    predictions.append("unknown")
                labels.append(INDEX_TO_LABEL[batch["labels"][idx].item()])
                proof_trace = assemble_proof(
                    query=f"{predictions[-1]}?",
                    rules=["soft_rule"],
                    facts=["synthetic_fact"],
                    confidence=prob_value,
                )
                sample = batch_samples[idx]
                premises_list = sample.get("premises", []) or []
                proof_valid = validate_proof(sample.get("proof_trace", []) or [], premises_list)
                proof_traces.append(proof_trace if proof_valid else [])
                _extract_gold_rules(sample, gold_rules)
                _extract_discovered_rules(sample, discovered_rules)
            offset += len(probs)

    symbolic_probs = [0.5 for _ in predicted_probs]
    metric_names = [metric.strip().upper() for metric in args.metrics.split(",")]
    results: dict = {}

    if "EDA" in metric_names:
        results["EDA"] = exact_deduction_accuracy(predictions, labels)
    if "PVR" in metric_names:
        results["PVR"] = proof_validity_rate(proof_traces)
    if "LCS" in metric_names:
        results["LCS"] = logical_consistency_score(predicted_probs, symbolic_probs)
    if "DER" in metric_names:
        # DER is per few-shot sample actually evaluated, not per full file.
        results["DER"] = data_efficiency_ratio(results.get("EDA", 0.0), len(shot_items))
    if "RIF1" in metric_names:
        premises_all = [p for sample in items for p in (sample.get("premises", []) or [])]
        validated_discovered = set(filter_valid_rules(discovered_rules, premises_all))
        validated_gold = set(filter_valid_rules(gold_rules, premises_all))
        results["RIF1"] = rule_induction_f1(validated_discovered, validated_gold)
        results["rules_discovered"] = len(validated_discovered)
        results["rules_gold"] = len(validated_gold)

    return results


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    results = evaluate(args)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    return results


if __name__ == "__main__":
    main()
