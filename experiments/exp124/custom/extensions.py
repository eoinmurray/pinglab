"""Register scientific operations not supplied by SNNLab's native language.

Equal-class loss, source-centered readout initialization and the rate-normalized
score preserve the calibrated SHD protocol. Native CrossEntropy averages examples,
so substituting it would change the training objective on unequal class counts.
The identity gain retains the historical graph and parameter inventory.

Import this module before authoring an exp124 bundle. Training uses native
zero-decay AdamW, whose update is equivalent to Adam when weight decay is zero.
"""

from snnlab import extensions


def equal_class_loss(losses, target, classes):
    import torch

    counts = torch.zeros(classes, device=losses.device, dtype=losses.dtype)
    counts.scatter_add_(0, target, torch.ones_like(target, dtype=losses.dtype))
    return (losses / counts[target]).sum() / (counts > 0).sum()


def class_balanced_cross_entropy(prediction, target, config):
    import torch

    losses = torch.nn.functional.cross_entropy(
        prediction, target.long(), reduction="none"
    )
    return equal_class_loss(losses, target.long(), config["classes"])


extensions.register_objective(
    "exp124.equal_class_cross_entropy/v1",
    class_balanced_cross_entropy,
    classification=True,
)


def rate_normalized_scores(sources, parameters, config):
    """Rescale each presentation by its mean final-layer spike rate."""
    scores, counts = sources
    rate = counts.mean(dim=-1, keepdim=True) / config["steps"]
    return scores * config["reference_rate"] / rate.clamp_min(config["minimum_rate"])


extensions.register_operation("exp124.rate_normalize/v1", rate_normalized_scores)


def centered_readout(shape, config, *, device, dtype):
    import torch

    value = (
        torch.randn(shape, device=device, dtype=dtype) * config["std"] + config["mean"]
    )
    return value - value.mean(dim=0, keepdim=True) if config["center"] else value


def identity_gain(shape, config, *, device, dtype):
    import torch

    return torch.eye(*shape, device=device, dtype=dtype) * config["gain"]


extensions.register_initializer("exp124.centered_readout/v1", centered_readout)
extensions.register_initializer("exp124.identity_gain/v1", identity_gain)
