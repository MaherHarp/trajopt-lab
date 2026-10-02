"""Derivatives of the discrete dynamics x' = step(x, u), batched over time.

Complex step: Im f(z + i h e_j) / h = df/dz_j + O(h^2). There is no
subtraction, so h can be 1e-30 and the result is exact to machine precision.
Finite differences, by contrast, trade truncation error against cancellation
error.

Second derivatives (needed by full DDP) come from central real differences of
complex-step Jacobians, which gives O(h^2) error with no cancellation in the
inner derivative.
"""

from __future__ import annotations

import numpy as np


def _stack(X, U):
    return np.concatenate([X, U], axis=-1)


def jacobians_cs(step, X, U, h=1e-30):
    """X: (T, n), U: (T, m) -> fx: (T, n, n), fu: (T, n, m)."""
    n = X.shape[-1]
    Z = _stack(X, U).astype(complex)
    k = Z.shape[-1]
    Zp = Z[None] + (1j * h) * np.eye(k)[:, None, :]           # (k, T, k)
    F = step(Zp[..., :n], Zp[..., n:])                         # (k, T, n)
    J = np.moveaxis(F.imag / h, 0, -1)                         # (T, n, k)
    return J[..., :n], J[..., n:]


def jacobians_fd(step, X, U, h=1e-6, scheme="forward"):
    """Finite-difference baseline: 'forward' (O(h)) or 'central' (O(h^2))."""
    n = X.shape[-1]
    Z = _stack(X, U)
    k = Z.shape[-1]
    E = h * np.eye(k)[:, None, :]
    Fp = step((Z[None] + E)[..., :n], (Z[None] + E)[..., n:])
    if scheme == "forward":
        F0 = step(X, U)[None]
        D = (Fp - F0) / h
    elif scheme == "central":
        Fm = step((Z[None] - E)[..., :n], (Z[None] - E)[..., n:])
        D = (Fp - Fm) / (2 * h)
    else:
        raise ValueError(scheme)
    J = np.moveaxis(D, 0, -1)
    return J[..., :n], J[..., n:]


def hessians_cs(step, X, U, h=1e-4, hc=1e-30):
    """Second-derivative tensor H: (T, n, k, k), H[t, i, a, b] = d2 f_i / dz_a dz_b.

    Returns (fxx, fux, fuu) with shapes (T,n,n,n), (T,n,m,n), (T,n,m,m).
    """
    n = X.shape[-1]
    Z = _stack(X, U).astype(complex)
    k = Z.shape[-1]
    I = np.eye(k)
    real_off = np.concatenate([h * I, -h * I])[:, None, None, :]   # (2k, 1, 1, k)
    imag_off = (1j * hc) * I[None, :, None, :]                     # (1, k, 1, k)
    Zp = Z[None, None] + real_off + imag_off                       # (2k, k, T, k)
    F = step(Zp[..., :n], Zp[..., n:]).imag / hc                   # (2k, k, T, n)
    D = (F[:k] - F[k:]) / (2 * h)                                  # [a, b, t, i]
    H = np.transpose(D, (2, 3, 0, 1))                              # (T, n, a, b)
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    return H[..., :n, :n], H[..., n:, :n], H[..., n:, n:]


def rk4_tangent(system, x, u):
    """Exact Jacobian of one RK4 step by the chain rule through the stages.

    Requires an analytic continuous Jacobian ``system.fjac(x, u)``. Used as an
    independent reference for the complex-step and finite-difference
    Jacobians.
    """
    n, m, dt = system.n, system.m, system.dt
    Dx = np.hstack([np.eye(n), np.zeros((n, m))])
    Du = np.hstack([np.zeros((m, n)), np.eye(m)])

    def stage(xs, Dxs):
        A, B = system.fjac(xs, u)
        return system.f(xs, u), A @ Dxs + B @ Du

    k1, d1 = stage(x, Dx)
    k2, d2 = stage(x + 0.5 * dt * k1, Dx + 0.5 * dt * d1)
    k3, d3 = stage(x + 0.5 * dt * k2, Dx + 0.5 * dt * d2)
    _, d4 = stage(x + dt * k3, Dx + dt * d3)
    J = Dx + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
    return J[:, :n], J[:, n:]
