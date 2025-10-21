from argparse import Namespace
from pathlib import Path

from eval_fewshot import evaluate
from scripts.build_clevrer_beta_s import generate_dataset


def test_evaluate_returns_metrics(tmp_path: Path):
    train_path, _ = generate_dataset(tmp_path, seed_value=5)
    args = Namespace(
        dataset=str(train_path),
        metrics="EDA,PVR",
        checkpoint="",
        batch_size=4,
        shots=3,
    )
    results = evaluate(args)
    assert "EDA" in results
    assert "PVR" in results
