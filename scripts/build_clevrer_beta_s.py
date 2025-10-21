#!/usr/bin/env python
"""Generate the CLEVRER-beta_s subset with symbolic noise for LogiFew.

The script intentionally fabricates a lightweight proxy of the original dataset
so that unit tests and dry-runs can execute locally without the full video
corpus. The structure matches the JSONL schema outlined in the project brief.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

EXAMPLE_COUNT = 120
OOD_COUNT = 60


def _random_object(seed: random.Random, allow_ood: bool = False) -> str:
    colors = ["red", "green", "blue", "yellow"]
    shapes = ["cube", "sphere", "cylinder"]
    materials = ["rubber", "metal"]
    if allow_ood:
        materials.extend(["metallic", "elastic"])
    color = seed.choice(colors)
    shape = seed.choice(shapes)
    material = seed.choice(materials)
    return f"{material}_{color}_{shape}"


def _create_premise(seed: random.Random) -> Tuple[List[str], str]:
    event_templates = [
        "{o1} collides_with {o2} at t={t}",
        "{o1} passes_behind {o2} at t={t}",
        "{o1} accelerates {dir} at t={t}",
    ]
    rule_templates = [
        "collide(X,Y) -> move(Y, opposite_dir) & move(X, dir)",
        "accelerate(X,dir) -> move(X, dir)",
        "support(X,Y) & heavy(Y) -> move(X, down)",
    ]
    directions = ["left", "right", "up", "down"]
    o1 = _random_object(seed)
    o2 = _random_object(seed)
    t = seed.randint(1, 7)
    event = seed.choice(event_templates).format(o1=o1, o2=o2, t=t, dir=seed.choice(directions))
    rule = seed.choice(rule_templates)
    return [event, rule], o2


def _maybe_corrupt_literal(seed: random.Random, premise: str) -> str:
    replacements = {
        "collides_with": "passes_through",
        "passes_behind": "occludes",
        "accelerates": "decelerates",
        "move": "rotate",
    }
    for token, replacement in replacements.items():
        if token in premise and seed.random() < 0.5:
            return premise.replace(token, replacement)
    return premise


def _build_example(seed: random.Random, idx: int, is_causal: bool, add_noise: bool) -> dict:
    premises, focus_object = _create_premise(seed)
    if add_noise:
        noise_idx = seed.randrange(len(premises))
        premises[noise_idx] = _maybe_corrupt_literal(seed, premises[noise_idx])

    query_type = seed.choice(["Entailment", "Contradiction", "Neutral"])
    question_time = seed.randint(2, 8)
    hypothesis_dir = seed.choice(["left", "right", "up", "down"])

    label_map = {
        "Entailment": "yes",
        "Contradiction": "no",
        "Neutral": "unknown",
    }
    label = label_map[query_type]

    proof_trace = []
    if query_type == "Entailment":
        proof_trace.append(f"rule_applied({premises[1]})")
        proof_trace.append(f"derived({focus_object}, move_{hypothesis_dir}, t={question_time})")
    elif query_type == "Contradiction":
        proof_trace.append("conflict_detected")

    video_prefix = "causal" if is_causal else "physical"
    video_id = f"{video_prefix}_{idx:03d}"

    return {
        "video_id": video_id,
        "premises": premises,
        "query": f"Does {focus_object} move {hypothesis_dir} at t={question_time}?",
        "label": label,
        "proof_trace": proof_trace,
        "meta": {
            "query_type": query_type,
            "rule_complexity": seed.choice(["1-hop", "3-hop"]),
            "symbolic_noise": add_noise,
        },
    }


def _iter_examples(seed: random.Random, total: int) -> Iterable[dict]:
    for idx in range(total):
        is_causal = idx % 2 == 1
        add_noise = seed.random() < 0.3
        yield _build_example(seed, idx, is_causal, add_noise)


def generate_dataset(output_dir: Path, seed_value: int = 7) -> Tuple[Path, Path]:
    """Create train and OOD JSONL files containing fabricated CLEVRER-beta_s samples."""
    rng = random.Random(seed_value)
    train_path = output_dir / "clevrer_beta_s_train.jsonl"
    ood_path = output_dir / "clevrer_beta_s_ood.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, examples: Iterable[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example) + "\n")

    write_jsonl(train_path, _iter_examples(rng, EXAMPLE_COUNT))
    write_jsonl(
        ood_path,
        (
            _build_example(random.Random(seed_value + idx + 1000), idx, idx % 2 == 0, True)
            for idx in range(OOD_COUNT)
        ),
    )

    return train_path, ood_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLEVRER-beta_s synthetic subset.")
    parser.add_argument("--output_dir", type=str, default="data/logifew", help="Destination directory.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_path, ood_path = generate_dataset(output_dir, args.seed)
    print(f"Wrote {train_path}")
    print(f"Wrote {ood_path}")


if __name__ == "__main__":
    main()
