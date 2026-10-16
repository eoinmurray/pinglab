"""Recording must preserve the classifier's numerical and gradient behavior."""

import numpy as np
import torch
from experiments.exp125.compute import Classifier
from experiments.exp125.recipe import CONFIG


def test_recording_preserves_scores_gradients_and_parameters():
    torch.manual_seed(42)
    model = Classifier(CONFIG)
    images = torch.linspace(0, 1, 2 * CONFIG["inputs"]).reshape(2, -1)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    baseline = model(images, torch.Generator().manual_seed(46))
    baseline.sum().backward()
    gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
    model.zero_grad()
    recording = {name: [] for name in ("input", "hidden", "output")}
    recorded = model(images, torch.Generator().manual_seed(46), recording)
    recorded.sum().backward()
    torch.testing.assert_close(recorded, baseline, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    np.testing.assert_array_equal(np.stack(recording["output"]).sum(axis=0), recorded.detach().numpy())
    for name, size in (("input", 64), ("hidden", 128), ("output", 10)):
        values = np.stack(recording[name])
        assert values.shape == (50, 2, size)
        assert np.isin(values, [0, 1]).all()


def test_repeated_presentations_reset_neuronal_state():
    torch.manual_seed(42)
    model = Classifier(CONFIG)
    image = torch.full((1, CONFIG["inputs"]), 0.5)
    first = model(image, torch.Generator().manual_seed(46))
    model(torch.ones_like(image), torch.Generator().manual_seed(99))
    repeated = model(image, torch.Generator().manual_seed(46))
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
