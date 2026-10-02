"""E2: iLQR (Gauss-Newton) vs full DDP (Newton): local convergence order,
per-iteration cost, and basins of attraction from random initial controls."""

from __future__ import annotations

import time

import numpy as np

from experiments.common import C, plt, save, table
from trajopt import ilqr
from trajopt.problems import PROBLEMS

SEEDS = 8


def order_estimates(err, floor):
    """Successive-ratio estimates p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1}) on the
    tail of the error sequence above the floor."""
    e = np.array([x for x in err if x > floor])
    if len(e) < 4:
        return np.nan
    p = np.log(e[2:] / e[1:-1]) / np.log(e[1:-1] / e[:-2])
    return float(np.median(p[-3:]))


def run():
    curves, rows, basin_rows = {}, [], []
    for name, make in PROBLEMS.items():
        sys_, cost, x0, U0 = make()
        runs = {}
        for method in ("ilqr", "ddp"):
            t0 = time.perf_counter()
            res = ilqr.solve(sys_, cost, x0, U0, method=method, max_iter=600, tol=1e-15)
            runs[method] = (res, time.perf_counter() - t0)
        J_star = min(r.cost for r, _ in runs.values())
        curves[name] = {}
        for method, (res, wall) in runs.items():
            J = np.array([h["cost"] for h in res.history])
            err = J - J_star
            curves[name][method] = err
            floor = 1e-9 * (1 + J_star)
            reach = np.flatnonzero(err <= 1e-6 * (1 + J_star))
            times = np.array([h["time"] for h in res.history[1:]])
            rows.append([name, method, f"{res.cost:.6f}", len(res.history) - 1,
                         reach[0] if reach.size else "never",
                         f"{order_estimates(err, floor):.2f}",
                         f"{1e3 * np.median(times):.1f}", f"{wall:.2f}"])

        # Basins: random initial controls, same problem.
        finals = {"ilqr": [], "ddp": []}
        for seed in range(SEEDS):
            Ur = np.random.default_rng(seed).normal(scale=2.0, size=U0.shape)
            for method in finals:
                finals[method].append(ilqr.solve(sys_, cost, x0, Ur, method=method, max_iter=600).cost)
        best = min(min(v) for v in finals.values())
        for method, v in finals.items():
            v = np.array(v)
            basin_rows.append([name, method, f"{np.median(v):.3f}", f"{v.min():.3f}", f"{v.max():.3f}",
                               f"{np.mean(v <= best * (1 + 1e-6)):.2f}"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, cv) in zip(axes, curves.items()):
        for (method, err), c in zip(cv.items(), C):
            e = np.maximum(err, 1e-16)
            ax.semilogy(np.arange(len(e)), e, color=c, label=method.upper() if method == "ddp" else "iLQR")
        ax.set_title(name)
        ax.set_xlabel("iteration")
        ax.set_ylim(1e-14, None)
    axes[0].set_ylabel("J - J*  (J* = best final cost)")
    axes[0].legend()
    fig1 = save(fig, "e2_convergence.png")

    md = ["## E2: iLQR vs DDP convergence", "", f"![convergence]({fig1})", "",
          "Initial controls are zero, tol = 1e-15. The order estimate is the median of the last three successive-ratio "
          "estimates of J - J*, above a floor of 1e-9(1+J*): 1 means linear, 2 means quadratic. Times are Python wall-clock on one machine.", "",
          table(["system", "method", "final J", "iterations", "iters to 1e-6 rel", "order est.",
                 "ms / iter (median)", "total s"], rows), "",
          f"**Basins of attraction:** {SEEDS} random initial control sequences (N(0, 2^2)) per system. "
          "The last column is the fraction of runs that reached the best cost found by either method.", "",
          table(["system", "method", "median J", "best J", "worst J", "frac. at best"], basin_rows), ""]
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
