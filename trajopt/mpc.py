"""Closed-loop evaluation: a plant (which may differ from the controller's
model) driven by one of four controllers.

- OpenLoop: replay an offline plan's feedforward controls.
- TVLQR:    replay the offline plan with its iLQR feedback gains
            u = u* + K (x - x*).
- ILQRMPC:  receding-horizon iLQR, warm-started from the shifted previous
            plan and capped at a fixed iteration budget per control step.
- MPPIMPC:  receding-horizon MPPI with a fixed sample/iteration budget.
"""

from __future__ import annotations

import time

import numpy as np

from . import ilqr
from .sampling import mppi_update


def _shift(U):
    return np.concatenate([U[1:], U[-1:]], axis=0)


class OpenLoop:
    def __init__(self, plan):
        self.plan = plan

    def act(self, t, x):
        return self.plan.U[min(t, len(self.plan.U) - 1)]


class TVLQR:
    def __init__(self, plan):
        self.plan = plan

    def act(self, t, x):
        t = min(t, len(self.plan.U) - 1)
        return self.plan.U[t] + self.plan.K[t] @ (x - self.plan.X[t])


class ILQRMPC:
    def __init__(self, model, cost, U_init, iters=2, method="ilqr"):
        self.model, self.cost, self.U = model, cost, np.array(U_init, float)
        self.iters, self.method = iters, method

    def act(self, t, x):
        res = ilqr.solve(self.model, self.cost, x, self.U, method=self.method,
                         max_iter=self.iters, mu0=1e-3)
        self.U = _shift(res.U)
        return res.U[0]


class MPPIMPC:
    def __init__(self, model, cost, U_init, rng, samples=256, sigma=2.0, lam=1.0, iters=1):
        self.model, self.cost, self.U, self.rng = model, cost, np.array(U_init, float), rng
        self.samples, self.sigma, self.lam, self.iters = samples, sigma, lam, iters

    def act(self, t, x):
        for _ in range(self.iters):
            self.U, _ = mppi_update(self.model, self.cost, x, self.U, self.rng,
                                    self.samples, self.sigma, self.lam)
        u = self.U[0].copy()
        self.U = _shift(self.U)
        return u


def run_closed_loop(plant, controller, x0, steps, rng=None, noise_std=0.0, noise_idx=()):
    """Simulate ``steps`` control intervals. Gaussian process noise of std
    noise_std * sqrt(dt) is added to the state components in noise_idx after
    each step. Returns (X, U, mean wall-clock seconds per control decision)."""
    X = np.empty((steps + 1, plant.n))
    U = np.empty((steps, plant.m))
    X[0] = x0
    elapsed = 0.0
    for t in range(steps):
        t0 = time.perf_counter()
        U[t] = controller.act(t, X[t])
        elapsed += time.perf_counter() - t0
        X[t + 1] = plant.step(X[t], U[t])
        if noise_std > 0:
            X[t + 1, list(noise_idx)] += noise_std * np.sqrt(plant.dt) * rng.standard_normal(len(noise_idx))
    return X, U, elapsed / steps


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
