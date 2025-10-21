#!/usr/bin/env python
"""Simple backtesting routine for LogiFew."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_fewshot import main as eval_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest LogiFew on OOD data.")
    parser.add_argument("--train_dataset", type=str, default="data/logifew/clevrer_beta_s_train.jsonl")
    parser.add_argument("--ood_dataset", type=str, default="data/logifew/clevrer_beta_s_ood.jsonl")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--metrics", type=str, default="EDA,PVR,LCS,DER,RIF1")
    return parser.parse_args()


def run_backtest(dataset_path: Path, shots: int, checkpoint: str, metrics: str) -> None:
    args = [
        "--dataset",
        str(dataset_path),
        "--shots",
        str(shots),
    ]
    if checkpoint:
        args.extend(["--checkpoint", checkpoint])
    if metrics:
        args.extend(["--metrics", metrics])
    eval_main(args)


def main() -> None:
    args = parse_args()
    print("Evaluating on few-shot train subset")
    run_backtest(Path(args.train_dataset), args.shots, args.checkpoint, args.metrics)
    print("Evaluating on OOD subset")
    run_backtest(Path(args.ood_dataset), args.shots, args.checkpoint, args.metrics)


if __name__ == "__main__":
    main()
