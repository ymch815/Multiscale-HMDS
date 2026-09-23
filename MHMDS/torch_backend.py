"""PyTorch optimization backend for MuH-MDS.

The objectives in this module are direct translations of the four Stan models
shipped with the paper implementation.  Public functions accept and return
NumPy arrays so PyTorch remains an implementation detail.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "MuH-MDS requires PyTorch. Install the project with `pip install -e .`."
    ) from exc


DTYPE = torch.float64
ACOSH_EPS = 1e-12
POSITIVE_EPS = 1e-10


@dataclass(frozen=True)
class OptimizationInfo:
    """Diagnostics from one L-BFGS run."""

    initial_loss: float
    final_loss: float
    function_evaluations: int


def _tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE, device="cpu")


def _positive(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw) + POSITIVE_EPS


def _inverse_positive(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value - POSITIVE_EPS, min=POSITIVE_EPS)
    return value + torch.log(-torch.expm1(-value))


def lorentz_time(coords: torch.Tensor) -> torch.Tensor:
    """Return the time-like coordinate for spatial Lorentz coordinates."""

    return torch.sqrt(1.0 + torch.sum(coords.square(), dim=-1))


def lorentz_distances(
    left: torch.Tensor, right: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return all Lorentz-model hyperbolic distances between two point sets."""

    if right is None:
        right = left
    left_t = lorentz_time(left)
    right_t = lorentz_time(right)
    inner = left_t[:, None] * right_t[None, :] - left @ right.T
    return torch.acosh(torch.clamp(inner, min=1.0 + ACOSH_EPS))


def _normal_nll(observed, mean, scale):
    # CmdStan optimization evaluates the target with propto=True, so terms
    # constant with respect to parameters are intentionally omitted.
    return 0.5 * ((observed - mean) / scale).square() + torch.log(scale)


def _inverse_gamma_nll(value, concentration=2.0, rate=0.5):
    # Stan's inv_gamma(alpha, beta) uses beta as the scale/rate parameter in
    # exp(-beta / x), matching this expression.
    beta = torch.as_tensor(rate, dtype=value.dtype)
    return (concentration + 1.0) * torch.log(value) + beta / value


def global_negative_log_posterior(coords, sigma, curvature, delta):
    """Negative log-posterior represented by ``lorentz2.stan``."""

    n_points = delta.shape[0]
    row, col = torch.triu_indices(n_points, n_points, offset=1)
    observed = delta[row, col]
    keep = observed > 0.0
    distances = lorentz_distances(coords)[row, col][keep]
    observed = observed[keep]
    scale = torch.sqrt(sigma[row[keep]].square() + sigma[col[keep]].square())
    likelihood = _normal_nll(observed, distances / curvature, scale).sum()
    sigma_prior = _inverse_gamma_nll(sigma).sum()
    n_terms = 0.5 * n_points * (n_points - 1)
    curvature_prior = n_terms * 0.5 * curvature.square() / 100.0
    return likelihood + sigma_prior + curvature_prior


def relaxation_loss(new_coords, reference_coords, mutual_delta, local_delta, curvature):
    """Negative log-likelihood represented by ``relax.stan`` (up to constants)."""

    n_new = local_delta.shape[0]
    row, col = torch.triu_indices(n_new, n_new, offset=1)
    local_observed = local_delta[row, col]
    local_keep = local_observed > 0.0
    local_distance = lorentz_distances(new_coords)[row, col]
    loss = 0.5 * (
        local_observed[local_keep] - local_distance[local_keep] / curvature
    ).square().sum()

    mutual_keep = mutual_delta > 0.0
    mutual_distance = lorentz_distances(new_coords, reference_coords)
    loss = loss + 0.5 * (
        mutual_delta[mutual_keep] - mutual_distance[mutual_keep] / curvature
    ).square().sum()
    return loss


def transform_negative_log_posterior(
    new_coords,
    new_sigma,
    reference_coords,
    reference_sigma,
    mutual_delta,
    local_delta,
    curvature,
):
    """Negative log-posterior represented by ``transform.stan``."""

    n_new = local_delta.shape[0]
    row, col = torch.triu_indices(n_new, n_new, offset=1)
    observed = local_delta[row, col]
    keep = observed > 0.0
    distance = lorentz_distances(new_coords)[row, col][keep]
    scale = torch.sqrt(
        new_sigma[row[keep]].square() + new_sigma[col[keep]].square()
    )
    loss = _normal_nll(observed[keep], distance / curvature, scale).sum()

    mutual_keep = mutual_delta > 0.0
    mutual_distance = lorentz_distances(new_coords, reference_coords)
    mutual_scale = torch.sqrt(
        new_sigma[:, None].square() + reference_sigma[None, :].square()
    )
    loss = loss + _normal_nll(
        mutual_delta[mutual_keep],
        mutual_distance[mutual_keep] / curvature,
        mutual_scale[mutual_keep],
    ).sum()
    return loss + _inverse_gamma_nll(new_sigma).sum()


def center_of_mass_loss(center, coords):
    """Distance-squared objective represented by ``CM2.stan``."""

    return lorentz_distances(center[None, :], coords).square().sum()


def _run_lbfgs(
    parameters: Iterable[torch.Tensor],
    objective,
    *,
    max_iter=1000,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
):
    parameters = list(parameters)
    with torch.no_grad():
        initial_loss = float(objective())
    if not np.isfinite(initial_loss):
        raise FloatingPointError("Initial MuH-MDS objective is not finite")

    evaluations = 0
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("MuH-MDS objective became non-finite")
        loss.backward()
        for parameter in parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("MuH-MDS gradient became non-finite")
        evaluations += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_loss = float(objective())
    if not np.isfinite(final_loss):
        raise FloatingPointError("Final MuH-MDS objective is not finite")
    return OptimizationInfo(initial_loss, final_loss, evaluations)


def optimize_global(delta, dimension, *, seed=0, max_iter=1000):
    delta_t = _tensor(delta)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    coords = (
        1e-2 * torch.randn((delta_t.shape[0], dimension), generator=generator, dtype=DTYPE)
    ).requires_grad_()
    raw_sigma = _inverse_positive(torch.full((delta_t.shape[0],), 1.0 / 6.0, dtype=DTYPE))
    raw_sigma.requires_grad_()
    raw_curvature = _inverse_positive(torch.tensor(1.0, dtype=DTYPE))
    raw_curvature.requires_grad_()

    def objective():
        return global_negative_log_posterior(
            coords, _positive(raw_sigma), _positive(raw_curvature), delta_t
        )

    info = _run_lbfgs(
        [coords, raw_sigma, raw_curvature], objective, max_iter=max_iter
    )
    return {
        "euc": coords.detach().numpy(),
        "sig": _positive(raw_sigma).detach().numpy(),
        "lambda": float(_positive(raw_curvature).detach()),
        "optimization": info,
    }


def optimize_relaxation(
    reference_coords,
    mutual_delta,
    local_delta,
    curvature,
    initial_coords,
    *,
    max_iter=1000,
):
    reference_t = _tensor(reference_coords)
    mutual_t = _tensor(mutual_delta)
    local_t = _tensor(local_delta)
    curvature_t = _tensor(curvature)
    coords = _tensor(initial_coords).clone().requires_grad_()

    def objective():
        return relaxation_loss(coords, reference_t, mutual_t, local_t, curvature_t)

    info = _run_lbfgs([coords], objective, max_iter=max_iter)
    return {"euc": coords.detach().numpy(), "optimization": info}


def optimize_transform(
    reference_coords,
    reference_sigma,
    mutual_delta,
    local_delta,
    curvature,
    initial_coords,
    *,
    max_iter=1000,
):
    reference_t = _tensor(reference_coords)
    reference_sigma_t = _tensor(reference_sigma)
    mutual_t = _tensor(mutual_delta)
    local_t = _tensor(local_delta)
    curvature_t = _tensor(curvature)
    coords = _tensor(initial_coords).clone().requires_grad_()
    raw_sigma = _inverse_positive(
        torch.full((coords.shape[0],), 1.0 / 6.0, dtype=DTYPE)
    ).requires_grad_()

    def objective():
        return transform_negative_log_posterior(
            coords,
            _positive(raw_sigma),
            reference_t,
            reference_sigma_t,
            mutual_t,
            local_t,
            curvature_t,
        )

    info = _run_lbfgs([coords, raw_sigma], objective, max_iter=max_iter)
    return {
        "euc": coords.detach().numpy(),
        "sig": _positive(raw_sigma).detach().numpy(),
        "optimization": info,
    }


def optimize_center_of_mass(coords, *, max_iter=1000):
    coords_t = _tensor(coords)
    center = coords_t.mean(dim=0).clone().requires_grad_()

    def objective():
        return center_of_mass_loss(center, coords_t)

    info = _run_lbfgs([center], objective, max_iter=max_iter)
    center_t = lorentz_time(center)
    return {
        "CM": center.detach().numpy(),
        "CM_t": float(center_t.detach()),
        "optimization": info,
    }
