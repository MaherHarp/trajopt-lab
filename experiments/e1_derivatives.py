"""E1: derivative accuracy, LQR equivalence, and conditioning of the swing-up cost."""

from __future__ import annotations

import numpy as np

from experiments.common import C, plt, save, table
from trajopt import ilqr
from trajopt.cost import QuadraticCost
from trajopt.derivatives import jacobians_cs, jacobians_fd, rk4_tangent
from trajopt.lqr import riccati
from trajopt.problems import cartpole_swingup, pendulum_swingup
from trajopt.systems import DoubleIntegrator, Pendulum


def jacobian_error_sweep():
    sys_ = Pendulum(dt=0.05)
    rng = np.random.default_rng(0)
    X, U = rng.uniform(-3, 3, (50, 2)), rng.uniform(-2, 2, (50, 1))
    ref = [rk4_tangent(sys_, X[t], U[t]) for t in range(50)]
    RA = np.stack([r[0] for r in ref])
    RB = np.stack([r[1] for r in ref])

    def err(A, B):
        return max(np.abs(A - RA).max(), np.abs(B - RB).max()) / np.abs(RA).max()

    hs = 10.0 ** np.arange(-1, -16, -0.5)
    out = {"forward FD": [], "central FD": [], "complex step": []}
    for h in hs:
        out["forward FD"].append(err(*jacobians_fd(sys_.step, X, U, h, "forward")))
        out["central FD"].append(err(*jacobians_fd(sys_.step, X, U, h, "central")))
        out["complex step"].append(err(*jacobians_cs(sys_.step, X, U, h)))
    hs_cs = 10.0 ** np.array([-20.0, -50, -100, -200, -300])
    cs_tiny = [err(*jacobians_cs(sys_.step, X, U, h)) for h in hs_cs]
    return hs, {k: np.maximum(np.array(v), 1e-17) for k, v in out.items()}, hs_cs, cs_tiny


def lqr_equivalence():
    sys_ = DoubleIntegrator(dt=0.1)
    Q, R, Qf = np.diag([1.0, 0.1]), np.array([[0.05]]), np.diag([50.0, 5.0])
    cost = QuadraticCost(Q, R, Qf, goal=[0.0, 0.0])
    T, x0 = 50, np.array([1.0, -0.5])
    A, B = jacobians_cs(sys_.step, np.zeros((1, 2)), np.zeros((1, 1)))
    K, P0 = riccati(A[0], B[0], Q, R, Qf, T)
    rows = []
    for method in ("ilqr", "ddp"):
        res = ilqr.solve(sys_, cost, x0, np.random.default_rng(1).normal(size=(T, 1)), method=method, mu0=0.0)
        rows.append([method, len(res.history) - 2, f"{np.abs(res.K + K).max():.1e}",
                     f"{abs(res.cost - 0.5 * x0 @ P0 @ x0):.1e}"])
    return rows


def fd_jacobians_in_ilqr():
    sys_, cost, x0, U0 = cartpole_swingup()
    rows = []
    variants = [("complex step", None), ("forward FD h=1e-8", 1e-8), ("forward FD h=1e-6", 1e-6),
                ("forward FD h=1e-4", 1e-4), ("forward FD h=1e-2", 1e-2)]
    orig = ilqr.jacobians_cs
    for name, h in variants:
        if h is not None:
            ilqr.jacobians_cs = lambda step, X, U, h=h: jacobians_fd(step, X, U, h, "forward")
        try:
            res = ilqr.solve(sys_, cost, x0, U0, method="ilqr", max_iter=400)
        finally:
            ilqr.jacobians_cs = orig
        g = ilqr.cost_gradient(sys_, cost, res.X, res.U)
        rows.append([name, f"{res.cost:.6f}", len(res.history) - 1, res.converged, f"{np.abs(g).max():.1e}"])
    return rows


def conditioning():
    """Exact gradient and Hessian of J(U) at the pendulum optimum."""
    sys_, cost, x0, U0 = pendulum_swingup()
    res = ilqr.solve(sys_, cost, x0, U0, method="ddp", tol=1e-13)
    U = res.U
    g = ilqr.cost_gradient(sys_, cost, res.X, U)
    T = len(U)
    H = np.empty((T, T))
    h = 1e-7
    for t in range(T):
        E = np.zeros_like(U); E[t, 0] = h
        gp = ilqr.cost_gradient(sys_, cost, sys_.rollout(x0, U + E), U + E)
        gm = ilqr.cost_gradient(sys_, cost, sys_.rollout(x0, U - E), U - E)
        H[:, t] = ((gp - gm) / (2 * h))[:, 0]
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    J = lambda V: cost.total(sys_.rollout(x0, V), V)
    t = int(np.argmax(np.abs(g)))
    fds = []
    for hh in (1e-3, 1e-5, 1e-7):
        E = np.zeros_like(U); E[t, 0] = hh
        fds.append((hh, (J(U + E) - J(U - E)) / (2 * hh)))
    return dict(g=g[t, 0], t=t, fds=fds, ev_min=ev.min(), ev_max=ev.max())


def run():
    hs, errs, hs_cs, cs_tiny = jacobian_error_sweep()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (name, e), c in zip(errs.items(), C):
        ax.loglog(hs, e, "o-", color=c, label=name, markersize=4)
    ax.set_xlabel("step size h")
    ax.set_ylabel("max relative Jacobian error")
    ax.set_title("RK4 pendulum Jacobian vs exact chain-rule tangent")
    ax.invert_xaxis()
    ax.legend()
    fig1 = save(fig, "e1_jacobian_error.png")

    lqr_rows = lqr_equivalence()
    fd_rows = fd_jacobians_in_ilqr()
    cond = conditioning()
    best = {k: (hs[np.argmin(v)], np.min(v)) for k, v in errs.items()}

    md = ["## E1: Derivatives, LQR equivalence, conditioning", "",
          f"![jacobian error]({fig1})", "",
          "Best achievable relative error over h (pendulum, 50 random states, reference = exact RK4 tangent):", "",
          table(["method", "best h", "best error"],
                [[k, f"{h:.0e}", f"{e:.1e}"] for k, (h, e) in best.items()]), "",
          "Complex step at h = " + ", ".join(f"{h:.0e}" for h in hs_cs) + ": errors "
          + ", ".join(f"{e:.1e}" for e in cs_tiny) + " (no cancellation, so h can be arbitrarily small).", "",
          "**LQR equivalence** (double integrator, T=50, random initial controls, mu=0):", "",
          table(["method", "iterations to optimum", "max abs(K - K_riccati)", "abs(J - x0'P0x0/2)"], lqr_rows), "",
          "**Jacobian source inside iLQR** (cart-pole swing-up, T=80, max 400 iterations):", "",
          table(["Jacobians", "final cost", "iterations", "converged", "max abs(dJ/dU), exact adjoint"], fd_rows), "",
          "**Conditioning of the pendulum swing-up cost J(U)** at the DDP optimum (T=100):", "",
          f"- Hessian eigenvalues (FD of the exact adjoint gradient) span {cond['ev_min']:.2e} .. {cond['ev_max']:.2e}.",
          f"- Exact gradient at t={cond['t']}: {cond['g']:.3e}. Central FD of J at the same point gives "
          + ", ".join(f"{fd:.3e} (h={h:.0e})" for h, fd in cond["fds"]) + ".", ""]
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
