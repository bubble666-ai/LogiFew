"""Security regression tests: encodings, loaders, config validation."""
import json

import pytest

from logifew.data.datasets import TextEncoder
from logifew.data.synthetic_rulebank import generate_rule_bank


def test_text_encoder_stable_across_instances():
    a, b = TextEncoder(), TextEncoder()
    assert (a.encode("red cube collides") == b.encode("red cube collides")).all()


def test_generate_rule_bank_rejects_empty_predicates():
    with pytest.raises(ValueError):
        generate_rule_bank(4, [], ["A"], 2, 0.1, 0.5)


def test_generate_rule_bank_rejects_bad_probability():
    with pytest.raises(ValueError):
        generate_rule_bank(4, ["P"], ["A"], 2, 9.9, 0.5)


def test_jsonl_loader_rejects_bad_json(tmp_path):
    from logifew.data.datasets import load_jsonl_dataset

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl_dataset(bad)


def test_checkpoint_helper_missing_file(tmp_path):
    from logifew.utils.checkpoints import safe_torch_load

    with pytest.raises(FileNotFoundError):
        safe_torch_load(tmp_path / "nope.ckpt")


def test_few_shot_subset_skips_unknown_labels():
    from eval_fewshot import few_shot_subset

    items = [
        {"label": "yes"},
        {"label": "bogus"},
        {"label": "no"},
    ]
    assert few_shot_subset(items, 5) == [{"label": "yes"}, {"label": "no"}]


def test_few_shot_subset_rejects_nonpositive_shots():
    from eval_fewshot import few_shot_subset

    with pytest.raises(ValueError):
        few_shot_subset([{"label": "yes"}], 0)


def test_eval_rejects_empty_dataset(tmp_path):
    from argparse import Namespace

    from eval_fewshot import evaluate

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate(Namespace(dataset=str(empty), metrics="EDA", checkpoint="", batch_size=4, shots=3))
