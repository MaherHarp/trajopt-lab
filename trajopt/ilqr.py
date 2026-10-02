"""iLQR and full second-order DDP with Levenberg-Marquardt regularisation.

The backward pass follows Tassa, Erez & Todorov (IROS 2012):
- state-space regularisation Quu + mu fu'fu, Qux + mu fu'fx
- the adaptive mu schedule
- a backtracking line search accepted on the ratio of actual to predicted
  reduction

``method='ddp'`` adds the dynamics-curvature terms Vx . f_zz to Qxx, Qux, and
Quu. That makes the local model an exact second-order expansion of the
cost-to-go, so convergence near a minimum becomes quadratic rather than
linear.

The forward pass rolls out every line-search step size as one batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .derivatives import hessians_cs, jacobians_cs

ALPHAS = 10.0 ** np.linspace(0, -3, 11)


@dataclass
class Result:
    X: np.ndarray
    U: np.ndarray
    K: np.ndarray
    k: np.ndarray
    cost: float
    converged: bool
    history: list = field(default_factory=list)


def backward_pass(fx, fu, lx, lu, lxx, luu, lfxx, lfx, mu, second=None):
    """Returns (K, k, dV1, dV2), or None if the regularised Quu is not PD."""
    T, n, m = fu.shape
    K = np.zeros((T, m, n))
    kff = np.zeros((T, m))
    Vx, Vxx = lfx.copy(), lfxx.copy()
    dV1 = dV2 = 0.0
    for t in range(T - 1, -1, -1):
        A, B = fx[t], fu[t]
        VA, VB = Vxx @ A, Vxx @ B
        Qx = lx[t] + A.T @ Vx
        Qu = lu[t] + B.T @ Vx
        Qxx = lxx + A.T @ VA
        Quu = luu + B.T @ VB
        Qux = B.T @ VA
        if second is not None:
            fxx, fux, fuu = second
            Qxx = Qxx + np.tensordot(Vx, fxx[t], axes=1)
            Qux = Qux + np.tensordot(Vx, fux[t], axes=1)
            Quu = Quu + np.tensordot(Vx, fuu[t], axes=1)
        Quu_r = Quu + mu * (B.T @ B)
        Qux_r = Qux + mu * (B.T @ A)
        try:
            np.linalg.cholesky(Quu_r)
        except np.linalg.LinAlgError:
            return None
        sol = np.linalg.solve(Quu_r, np.column_stack([Qu, Qux_r]))
        k_t, K_t = -sol[:, 0], -sol[:, 1:]
        kff[t], K[t] = k_t, K_t
        dV1 += k_t @ Qu
        dV2 += 0.5 * k_t @ Quu @ k_t
        Vx = Qx + K_t.T @ (Quu @ k_t) + K_t.T @ Qu + Qux.T @ k_t
        Vxx = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
        Vxx = 0.5 * (Vxx + Vxx.T)
    return K, kff, dV1, dV2


def forward_pass(system, cost, x0, Xbar, Ubar, K, kff, alphas):
    """Closed-loop rollouts u = ubar + a k + K (x - xbar) for every a at once."""
    A = len(alphas)
    T = Ubar.shape[0]
    X = np.empty((A, T + 1, system.n))
    U = np.empty((A, T, system.m))
    X[:, 0] = x0
    # Large steps can diverge to inf/nan. Those rollouts simply fail the
    # acceptance test.
    with np.errstate(all="ignore"):
        for t in range(T):
            U[:, t] = Ubar[t] + alphas[:, None] * kff[t] + (X[:, t] - Xbar[t]) @ K[t].T
            X[:, t + 1] = system.step(X[:, t], U[:, t])
        J = cost.total(X, U)
    return X, U, np.where(np.isfinite(J), J, np.inf)


def cost_gradient(system, cost, X, U):
    """Exact open-loop gradient dJ/dU by the adjoint (costate) recursion."""
    fx, fu = jacobians_cs(system.step, X[:-1], U)
    lx, lu, _, _, _, lfx = cost.derivatives(X, U)
    lam = lfx
    g = np.empty_like(U)
    for t in range(U.shape[0] - 1, -1, -1):
        g[t] = lu[t] + fu[t].T @ lam
        lam = lx[t] + fx[t].T @ lam
    return g


def solve(system, cost, x0, U0, method="ilqr", max_iter=200, tol=1e-10,
          mu0=1.0, mu_min=1e-6, mu_max=1e10, delta0=2.0, alphas=ALPHAS,
          accept_ratio=0.1, hess_h=1e-4):
    """Optimise an open-loop control sequence from initial state x0.

    Stops when the predicted reduction from a full step falls below
    tol * (1 + |J|), when regularisation exceeds mu_max, or after max_iter
    iterations. ``history`` has one record per iteration.
    """
    if method not in ("ilqr", "ddp"):
        raise ValueError(method)
    x0 = np.asarray(x0, float)
    U = np.array(U0, float)
    X = system.rollout(x0, U)
    J = float(cost.total(X, U))
    mu, delta = mu0, 1.0
    K = np.zeros((U.shape[0], system.m, system.n))
    kff = np.zeros_like(U)
    history = [dict(iter=0, cost=J, mu=mu, alpha=0.0, time=0.0, grad=np.nan)]
    converged = False

    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()
        fx, fu = jacobians_cs(system.step, X[:-1], U)
        second = hessians_cs(system.step, X[:-1], U, h=hess_h) if method == "ddp" else None
        lx, lu, lxx, luu, lfxx, lfx = cost.derivatives(X, U)

        accepted = False
        while True:
            bp = backward_pass(fx, fu, lx, lu, lxx, luu, lfxx, lfx, mu, second)
            if bp is None:
                delta = max(delta0, delta * delta0)
                mu = max(mu_min, mu * delta)
                if mu > mu_max:
                    break
                continue
            K_new, k_new, dV1, dV2 = bp
            if -(dV1 + dV2) < tol * (1.0 + abs(J)):
                converged = True
                K, kff = K_new, k_new
                break
            Xs, Us, Js = forward_pass(system, cost, x0, X, U, K_new, k_new, alphas)
            expected = -(alphas * dV1 + alphas**2 * dV2)
            ok = np.flatnonzero((Js < J) & ((J - Js) > accept_ratio * expected))
            if ok.size:
                a = ok[0]
                X, U, J = Xs[a], Us[a], float(Js[a])
                K, kff = K_new, k_new
                delta = min(1.0 / delta0, delta / delta0)
                mu = mu * delta if mu * delta > mu_min else 0.0
                accepted = True
                alpha = alphas[a]
                break
            delta = max(delta0, delta * delta0)
            mu = max(mu_min, mu * delta)
            if mu > mu_max:
                break

        grad = float(np.max(np.abs(kff)))
        history.append(dict(iter=it, cost=J, mu=mu, alpha=alpha if accepted else 0.0,
                            time=time.perf_counter() - t0, grad=grad))
        if not accepted:
            break
    return Result(X=X, U=U, K=K, k=kff, cost=J, converged=converged, history=history)
