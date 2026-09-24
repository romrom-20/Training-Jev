import torch

from latent_decisions.steering import decompose_shared_residual


def test_decomposition_reconstructs_directions_and_matches_random_residual_norms():
    native = torch.tensor(
        [[1.0, 0.1, 0.0], [1.0, 0.0, 0.2], [1.0, -0.1, -0.1]],
        dtype=torch.float32,
    )
    native = native / native.norm(dim=1, keepdim=True)

    parts = decompose_shared_residual(native)
    axis = native.mean(0)
    axis = axis / axis.norm()

    assert torch.allclose(parts["shared"] + parts["residual"], native, atol=1e-6)
    assert torch.allclose(parts["residual"] @ axis, torch.zeros(3), atol=1e-6)
    assert torch.allclose(parts["random_residual"] @ axis, torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        parts["random_residual"].norm(dim=1), parts["residual"].norm(dim=1), atol=1e-6
    )
    assert torch.equal(
        parts["random_residual"], decompose_shared_residual(native)["random_residual"]
    )
