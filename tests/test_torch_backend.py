import numpy as np
import pytest
import torch

from MHMDS import torch_backend as backend


def test_lorentz_distance_is_symmetric_and_matches_radial_distance():
    coords = torch.tensor([[0.0, 0.0], [np.sinh(0.7), 0.0]], dtype=torch.float64)
    distances = backend.lorentz_distances(coords)

    assert torch.allclose(distances, distances.T)
    assert distances[0, 1].item() == pytest.approx(0.7, abs=1e-12)
    # Diagonal values are a tiny positive stabilization of exact zero.
    assert torch.max(torch.diag(distances)).item() < 2e-6


def test_relaxation_optimizer_reduces_loss():
    reference = np.array([[0.0], [np.sinh(1.0)]])
    initial = np.array([[np.sinh(0.2)]])
    mutual = np.array([[0.7, 0.3]])
    local = np.zeros((1, 1))

    result = backend.optimize_relaxation(
        reference, mutual, local, 1.0, initial, max_iter=100
    )

    assert result["euc"].shape == (1, 1)
    assert result["optimization"].final_loss < result["optimization"].initial_loss
    assert result["optimization"].final_loss < 1e-10


def test_global_optimizer_returns_finite_compatible_outputs():
    delta = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.0],
        ]
    )

    result = backend.optimize_global(delta, 2, seed=4, max_iter=100)

    assert result["euc"].shape == (3, 2)
    assert result["sig"].shape == (3,)
    assert np.isfinite(result["euc"]).all()
    assert np.all(result["sig"] > 0.0)
    assert np.isfinite(result["lambda"]) and result["lambda"] > 0.0
    assert result["optimization"].final_loss <= result["optimization"].initial_loss


def test_global_objective_matches_stan_proportional_target():
    delta = torch.tensor([[0.0, 0.8], [0.8, 0.0]], dtype=torch.float64)
    coords = torch.tensor([[0.0], [np.sinh(0.5)]], dtype=torch.float64)
    sigma = torch.tensor([0.2, 0.3], dtype=torch.float64)
    curvature = torch.tensor(0.9, dtype=torch.float64)
    distance = torch.tensor(0.5, dtype=torch.float64)
    scale = torch.sqrt(torch.tensor(0.2**2 + 0.3**2, dtype=torch.float64))
    expected = (
        0.5 * ((0.8 - distance / curvature) / scale) ** 2
        + torch.log(scale)
        + (3.0 * torch.log(sigma) + 0.5 / sigma).sum()
        + 0.5 * curvature**2 / 100.0
    )

    actual = backend.global_negative_log_posterior(
        coords, sigma, curvature, delta
    )

    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)


def test_global_objective_gradient_matches_finite_differences():
    delta = torch.tensor(
        [[0.0, 0.5, 1.0], [0.5, 0.0, 0.7], [1.0, 0.7, 0.0]],
        dtype=torch.float64,
    )
    coords = torch.tensor(
        [[0.1, -0.2], [0.5, 0.3], [-0.4, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    sigma = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
    curvature = torch.tensor(0.9, dtype=torch.float64)

    def objective(candidate):
        return backend.global_negative_log_posterior(
            candidate, sigma, curvature, delta
        )

    assert torch.autograd.gradcheck(objective, (coords,), eps=1e-6, atol=1e-5)


def test_center_of_mass_for_symmetric_points_is_origin():
    value = np.sinh(0.4)
    coords = np.array([[-value, 0.0], [value, 0.0]])

    result = backend.optimize_center_of_mass(coords, max_iter=100)

    np.testing.assert_allclose(result["CM"], np.zeros(2), atol=1e-10)
    assert result["CM_t"] == pytest.approx(1.0, abs=1e-12)


def test_transform_optimizer_returns_positive_uncertainty():
    reference = np.array([[0.0], [np.sinh(1.0)]])
    result = backend.optimize_transform(
        reference_coords=reference,
        reference_sigma=np.ones(2),
        mutual_delta=np.array([[0.7, 0.3]]),
        local_delta=np.zeros((1, 1)),
        curvature=1.0,
        initial_coords=np.array([[np.sinh(0.2)]]),
        max_iter=100,
    )

    assert result["euc"].shape == (1, 1)
    assert result["sig"].shape == (1,)
    assert result["sig"][0] > 0.0
    assert result["optimization"].final_loss <= result["optimization"].initial_loss
