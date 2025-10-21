import torch

from logifew.models.nsml import NSMLModel, NSMLConfig


def test_nsml_forward_pass():
    config = NSMLConfig(hidden_size=32, num_rules=8, key_dim=16, value_dim=16)
    model = NSMLModel(config)
    premises = torch.randn(4, config.input_dim)
    queries = torch.randn(4, config.input_dim)
    outputs = model(premises, queries)
    assert "probabilities" in outputs
    assert outputs["probabilities"].shape == (4,)

    premises_text = [["p1"], ["p2"], ["p3"], ["p4"]]
    queries_text = ["q1", "q2", "q3", "q4"]
    losses = model.compute_losses(
        premises,
        queries,
        premises_text=premises_text,
        queries_text=queries_text,
        outputs=outputs,
        labels=torch.tensor([1, 0, 2, 1]),
    )
    assert "total_loss" in losses
    assert losses["total_loss"].requires_grad
