from logifew.utils.metrics import (
    data_efficiency_ratio,
    exact_deduction_accuracy,
    logical_consistency_score,
    proof_validity_rate,
    rule_induction_f1,
)


def test_metrics_behaviour():
    assert exact_deduction_accuracy(["yes", "no"], ["yes", "unknown"]) == 0.5
    assert 0 <= proof_validity_rate([["a"], []]) <= 1
    assert logical_consistency_score([0.5, 0.6], [0.5, 0.5]) <= 1
    assert rule_induction_f1(["a"], ["a", "b"]) > 0
    assert abs(data_efficiency_ratio(0.7, 100) - 0.007) < 1e-6
