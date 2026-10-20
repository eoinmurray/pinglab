"""Load the fixed official MNIST training image and encode its input drive.

The image is original training sample zero (digit five), scaled from uint8 to
[0, 1] without cropping or thresholding. MNIST must already exist under data/;
this module never downloads it. Importing the module performs no data access.

load_sample() returns pixels and file-checksum provenance. encode_drive() uses
SNNLab's native rate encoder for image and background spikes. Reusing the
background encoding seed at each rate preserves nested private I spike trains.
"""

import torch
from pingstore.contracts import file_sha256
from snnlab.sim import encode_images_poisson
from torchvision.datasets import MNIST


def load_sample(repo, cfg):
    root = repo / cfg["dataset_directory"]
    images = MNIST(str(root), train=True, download=False)
    index = cfg["sample_index"]
    label = int(images.targets[index])
    if label != cfg["sample_label"]:
        raise ValueError("unexpected original MNIST sample label")
    pixels = images.data[index].float().reshape(1, cfg["n_input"]) / 255.0
    files = [
        root / "MNIST/raw" / name
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    ]
    provenance = dict(
        source="torchvision.datasets.MNIST",
        partition="train",
        sample_index=index,
        sample_label=label,
        preprocessing="uint8 / 255; flatten in row-major order",
        files={str(path.relative_to(repo)): file_sha256(path) for path in files},
    )
    return pixels, provenance


def encode_drive(pixels, cfg, seed):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    image = encode_images_poisson(
        pixels,
        steps,
        cfg["dt_ms"],
        cfg["max_pixel_rate_hz"],
        generator=torch.Generator().manual_seed(seed + cfg["image_seed_offset"]),
    )
    return image, {
        rate: encode_images_poisson(
            torch.ones(1, cfg["n_i"]),
            steps,
            cfg["dt_ms"],
            rate,
            # The native encoder draws the same uniforms at every rate.
            generator=torch.Generator().manual_seed(
                seed + cfg["background_seed_offset"]
            ),
        )
        for rate in sorted({case["i_rate_hz"] for case in cfg["cases"]})
    }
