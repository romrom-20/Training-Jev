"""Small reusable vector transforms for frozen activation interventions."""

import torch


def decompose_shared_residual(native, seed=20261001):
    """Project unit directions onto their shared axis and sample norm-matched residuals."""
    axis = native.mean(0)
    axis = axis / axis.norm().clamp_min(1e-12)
    shared, residual, random_residual = [], [], []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for direction in native:
        projection = torch.dot(direction, axis) * axis
        remainder = direction - projection
        random = torch.randn(direction.shape, generator=generator)
        random -= torch.dot(random, axis) * axis
        random = random / random.norm().clamp_min(1e-12) * remainder.norm()
        shared.append(projection)
        residual.append(remainder)
        random_residual.append(random)
    return {
        "native": native,
        "shared": torch.stack(shared),
        "residual": torch.stack(residual),
        "random_residual": torch.stack(random_residual),
    }
