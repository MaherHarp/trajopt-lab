"""Derivative-free sampling planners: MPPI and the cross-entropy method.

Both use the same batched rollout as the line search in ``ilqr.py`` and see
only the cost of each rollout, never derivatives.

MPPI here uses the common form: path weights exp(-S/lam) on the cost of the
perturbed controls, followed by a weighted mean of the perturbations. With an
explicit control cost in S, this matches Williams et al. (2017) with the
control-cost coefficient set to 1.
"""

from __future__ import annotations

import numpy as np


def mppi_update(system, cost, x0, U, rng, samples=256, sigma=1.0, lam=1.0):
    """One MPPI iteration. Returns the new mean controls and the sample costs."""
    eps = sigma * rng.standard_normal((samples,) + U.shape)
    V = U[None] + eps
    S = cost.total(system.rollout(np.broadcast_to(x0, (samples, system.n)), V), V)
    w = np.exp(-(S - S.min()) / lam)
    w /= w.sum()
    return U + np.tensordot(w, eps, axes=1), S


def cem_update(system, cost, x0, mean, std, rng, samples=256, elite_frac=0.1,
               smoothing=0.1, min_std=1e-3):
    """One CEM iteration: refit a diagonal Gaussian to the elite rollouts."""
    V = mean[None] + std[None] * rng.standard_normal((samples,) + mean.shape)
    S = cost.total(system.rollout(np.broadcast_to(x0, (samples, system.n)), V), V)
    elite = V[np.argsort(S)[: max(2, int(elite_frac * samples))]]
    new_mean = smoothing * mean + (1 - smoothing) * elite.mean(0)
    new_std = np.maximum(smoothing * std + (1 - smoothing) * elite.std(0), min_std)
    return new_mean, new_std, S


def optimise(system, cost, x0, U0, method, iters, rng, samples=256, sigma=1.0, lam=1.0,
             elite_frac=0.1):
    """Open-loop optimisation. Returns U and the true cost of the mean after each iteration."""
    U = np.array(U0, float)
    std = np.full_like(U, sigma)
    trace = [float(cost.total(system.rollout(x0, U), U))]
    for _ in range(iters):
        if method == "mppi":
            U, _ = mppi_update(system, cost, x0, U, rng, samples, sigma, lam)
        elif method == "cem":
            U, std, _ = cem_update(system, cost, x0, U, std, rng, samples, elite_frac)
        else:
            raise ValueError(method)
        trace.append(float(cost.total(system.rollout(x0, U), U)))
    return U, np.array(trace)
