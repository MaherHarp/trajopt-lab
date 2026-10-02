"""Finite-horizon discrete-time LQR by the Riccati recursion (reference)."""

from __future__ import annotations

import numpy as np


def riccati(A, B, Q, R, Qf, T):
    """Gains K_t (u_t = -K_t x_t) for x' = A x + B u and cost
    sum 1/2 x'Qx + 1/2 u'Ru + 1/2 x_T'Qf x_T. Returns K: (T, m, n), P_0."""
    P = Qf.copy()
    K = np.empty((T, B.shape[1], A.shape[0]))
    for t in range(T - 1, -1, -1):
        S = R + B.T @ P @ B
        K[t] = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K[t])
        P = 0.5 * (P + P.T)
    return K, P
