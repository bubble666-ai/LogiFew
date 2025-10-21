#!/usr/bin/env python
"""Convert CLEVRER real annotations into LogiFew-compatible JSONL subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

SUPPORTED_ANSWERS = {"yes", "no"}


def load_annotations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def translate_question(video_id: str, question: dict) -> dict | None:
    answer = question.get("answer")
    if isinstance(answer, dict):
        answer = answer.get("answer")
    label = None
    if isinstance(answer, str) and answer.lower() in SUPPORTED_ANSWERS:
        label = answer.lower()
    else:
        label = "unknown"
    program = question.get("program", [])
    if isinstance(program, list):
        proof_trace = [f"step_{idx}: {token}" for idx, token in enumerate(program)]
    else:
        proof_trace = []
    premises = []
    if isinstance(program, list):
        chunk = []
        for token in program:
            chunk.append(str(token))
            if len(chunk) >= 4:
                premises.append(" -> ".join(chunk))
                chunk = []
        if chunk:
            premises.append(" -> ".join(chunk))
    question_text = question.get("question", "")
    return {
        "video_id": video_id,
        "premises": premises or ["program_tokens"],
        "query": question_text,
        "label": label,
        "proof_trace": proof_trace,
    }


def convert_annotations(data: list[dict], limit: int | None = None) -> list[dict]:
    examples: List[dict] = []
    for idx, sample in enumerate(data):
        video_id = sample.get("scene_index", f"video_{idx:05d}")
        questions = sample.get("questions", [])
        for q in questions:
            example = translate_question(str(video_id), q)
            if example is not None:
                examples.append(example)
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_subset(
    train_paths: list[Path],
    test_path: Path,
    output_dir: Path,
    limit: int | None,
) -> tuple[Path, Path]:
    train_data: list[dict] = []
    for path in train_paths:
        train_data.extend(load_annotations(path))
    test_data = load_annotations(test_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = convert_annotations(train_data, limit)
    test_examples = convert_annotations(test_data, limit)

    train_out = output_dir / "clevrer_real_train.jsonl"
    test_out = output_dir / "clevrer_real_test.jsonl"
    write_jsonl(train_out, train_examples)
    write_jsonl(test_out, test_examples)
    return train_out, test_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real CLEVRER subset in LogiFew format.")
    parser.add_argument("--train_file", type=str, default="data/logifew/clevrer_train_real.json")
    parser.add_argument(
        "--extra_train_files",
        type=str,
        nargs="*",
        default=[],
        help="Additional train-like annotation files (e.g., validation) to concatenate.",
    )
    parser.add_argument("--test_file", type=str, default="data/logifew/clevrer_test_real.json")
    parser.add_argument("--output_dir", type=str, default="data/logifew")
    parser.add_argument("--limit", type=int, default=300, help="Maximum number of questions per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = [Path(args.train_file)] + [Path(path) for path in args.extra_train_files]
    train_out, test_out = build_subset(
        train_paths,
        Path(args.test_file),
        Path(args.output_dir),
        args.limit if args.limit > 0 else None,
    )
    print(f"Wrote {train_out}")
    print(f"Wrote {test_out}")


if __name__ == "__main__":
    main()
