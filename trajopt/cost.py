"""Quadratic tracking cost with analytic derivatives and optional periodic
angle residuals."""

from __future__ import annotations

import numpy as np


class QuadraticCost:
    """J = sum_t 1/2 e(x_t)'Q e(x_t) + 1/2 u_t'R u_t  +  1/2 e(x_T)'Qf e(x_T).

    The residual is e_i = x_i - g_i, except for components in ``angle_idx``,
    where e_i = 2 sin((x_i - g_i) / 2). Then 1/2 q e_i^2 = q (1 - cos(x_i - g_i)):
    this matches the quadratic near the goal, is 2*pi-periodic, and is smooth
    everywhere. That matters for receding-horizon control. Any wrapped
    quadratic has a branch cut, and a planner whose state crosses that cut
    suddenly sees a different goal.
    """

    def __init__(self, Q, R, Qf, goal, angle_idx=()):
        self.Q = np.atleast_2d(np.asarray(Q, float))
        self.R = np.atleast_2d(np.asarray(R, float))
        self.Qf = np.atleast_2d(np.asarray(Qf, float))
        self.goal = np.asarray(goal, float)
        self.angle = np.zeros(len(self.goal), bool)
        self.angle[list(angle_idx)] = True

    def residual(self, X):
        """Residual e, its derivative de/dx, and second derivative (all elementwise)."""
        d = X - self.goal
        e = np.where(self.angle, 2 * np.sin(d / 2), d)
        de = np.where(self.angle, np.cos(d / 2), 1.0)
        d2e = np.where(self.angle, -0.5 * np.sin(d / 2), 0.0)
        return e, de, d2e

    def running(self, X, U):
        """X: (..., T, n), U: (..., T, m) -> (..., T)."""
        e = self.residual(X)[0]
        return 0.5 * (np.einsum("...i,ij,...j->...", e, self.Q, e)
                      + np.einsum("...i,ij,...j->...", U, self.R, U))

    def terminal(self, x):
        e = self.residual(x)[0]
        return 0.5 * np.einsum("...i,ij,...j->...", e, self.Qf, e)

    def total(self, X, U):
        """X: (..., T+1, n), U: (..., T, m) -> (...)."""
        return self.running(X[..., :-1, :], U).sum(-1) + self.terminal(X[..., -1, :])

    def _state_derivs(self, X, W):
        e, de, d2e = self.residual(X)
        We = e @ W
        lx = We * de
        lxx = de[..., :, None] * W * de[..., None, :]
        idx = np.arange(X.shape[-1])
        lxx[..., idx, idx] += We * d2e
        return lx, lxx

    def derivatives(self, X, U):
        """Gradients and Hessians along a trajectory (no batch axis).

        Returns (lx (T,n), lu (T,m), lxx (T,n,n), luu (m,m), lfxx (n,n), lfx (n,)).
        """
        lx, lxx = self._state_derivs(X[:-1], self.Q)
        lfx, lfxx = self._state_derivs(X[-1], self.Qf)
        return lx, U @ self.R, lxx, self.R, lfxx, lfx
