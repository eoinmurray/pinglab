import importlib.util
from pathlib import Path

import pytest
import torch


def _load_exp065():
    path = Path(__file__).resolve().parents[1] / "exp065.py"
    spec = importlib.util.spec_from_file_location("exp065", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_binarize_foreground_uses_strict_threshold():
    exp = _load_exp065()
    x = torch.tensor([[0.0, 0.1, 1.0]])
    assert torch.equal(exp.binarize_foreground(x), torch.tensor([[0.0, 1.0, 1.0]]))


def test_retention_endpoints_and_background():
    exp = _load_exp065()
    binary = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    g = torch.Generator().manual_seed(1)
    assert torch.equal(exp.retain_foreground(binary, 1.0, generator=g), binary)
    assert torch.count_nonzero(exp.retain_foreground(binary, 0.0, generator=g)) == 0
    masked = exp.retain_foreground(binary, 0.5, generator=g)
    assert masked[0, 0] == 0 and masked[0, 2] == 0


def test_retention_is_deterministic_and_tracks_q():
    exp = _load_exp065()
    binary = torch.ones((1000, 100))
    a = exp.retain_foreground(binary, 0.3, generator=torch.Generator().manual_seed(7))
    b = exp.retain_foreground(binary, 0.3, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert float(a.mean()) == pytest.approx(0.3, abs=0.005)


def test_retention_rejects_invalid_probability():
    exp = _load_exp065()
    with pytest.raises(ValueError):
        exp.retain_foreground(torch.ones(2), 1.1, generator=torch.Generator())


def test_ann_architecture_matches_ping_excitatory_width():
    exp = _load_exp065()
    model = exp.MatchedANN()
    assert model.layers[0].in_features == 784
    assert model.layers[0].out_features == 1024
    assert model.layers[2].out_features == 10


def test_matched_stimuli_use_same_examples_and_masks():
    exp = _load_exp065()
    binary = torch.ones((200, 784))
    labels = torch.arange(200) % 10
    a, ya, ia = exp.matched_stimuli(binary, labels, 0.5)
    b, yb, ib = exp.matched_stimuli(binary, labels, 0.5)
    assert torch.equal(a, b)
    assert torch.equal(ya, yb)
    assert (ia == ib).all()


def test_confusion_rows_are_true_classes_and_columns_predictions():
    exp = _load_exp065()
    matrix = exp.confusion([0, 0, 1], [0, 1, 1])
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1
