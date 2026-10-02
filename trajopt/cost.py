"""Quadratic tracking cost with analytic derivatives."""

from __future__ import annotations

import numpy as np


class QuadraticCost:
    """J = sum_t 1/2 (x_t-g)'Q(x_t-g) + 1/2 u_t'R u_t  +  1/2 (x_T-g)'Qf(x_T-g)."""

    def __init__(self, Q, R, Qf, goal):
        self.Q = np.atleast_2d(np.asarray(Q, float))
        self.R = np.atleast_2d(np.asarray(R, float))
        self.Qf = np.atleast_2d(np.asarray(Qf, float))
        self.goal = np.asarray(goal, float)

    def running(self, X, U):
        """X: (..., T, n), U: (..., T, m) -> (..., T)."""
        e = X - self.goal
        return 0.5 * (np.einsum("...i,ij,...j->...", e, self.Q, e)
                      + np.einsum("...i,ij,...j->...", U, self.R, U))

    def terminal(self, x):
        e = x - self.goal
        return 0.5 * np.einsum("...i,ij,...j->...", e, self.Qf, e)

    def total(self, X, U):
        """X: (..., T+1, n), U: (..., T, m) -> (...)."""
        return self.running(X[..., :-1, :], U).sum(-1) + self.terminal(X[..., -1, :])

    def derivatives(self, X, U):
        """Gradients and Hessians along a trajectory (no batch axis)."""
        e = X[:-1] - self.goal
        lx = e @ self.Q
        lu = U @ self.R
        lfx = self.Qf @ (X[-1] - self.goal)
        return lx, lu, self.Q, self.R, self.Qf, lfx
