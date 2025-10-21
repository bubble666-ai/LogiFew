import json
from pathlib import Path

from logifew.data.datasets import ClevrerBetaSDataset, TextEncoder
from logifew.data.datasets import collate_fn
from logifew.data.synthetic_rulebank import generate_rule_bank, to_training_examples
from scripts.build_clevrer_beta_s import generate_dataset


def test_generate_dataset(tmp_path: Path):
    train_path, ood_path = generate_dataset(tmp_path, seed_value=1)
    train_data = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    assert len(train_data) == 120
    assert all("premises" in item and "query" in item for item in train_data)

    encoder = TextEncoder()
    dataset = ClevrerBetaSDataset(train_data, encoder=encoder)
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["premises"].shape[0] == 2
    assert len(batch["premises_text"]) == 2
    assert isinstance(batch["queries_text"][0], str)


def test_synthetic_rulebank():
    clauses = generate_rule_bank(
        num_proofs=10,
        predicates=["LeftOf", "RightOf"],
        variables=["A", "B", "C"],
        max_body_literals=3,
        noise_probability=0.2,
        negative_ratio=0.3,
        seed=2,
    )
    examples = to_training_examples(clauses)
    assert len(examples) == 10
    labels = {example["label"] for example in examples}
    assert labels.issubset({"yes", "no"})
