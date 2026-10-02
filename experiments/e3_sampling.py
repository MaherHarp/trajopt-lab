"""E3: derivative-free planners (MPPI, CEM) vs the iLQR optimum on open-loop
cart-pole swing-up: sample budget, temperature, and final cost gap."""

from __future__ import annotations

import time

import numpy as np

from experiments.common import C, plt, save, table
from trajopt import ilqr, sampling
from trajopt.problems import cartpole_swingup

ITERS, SEEDS, SIGMA = 150, 4, 2.0


def run():
    sys_, cost, x0, U0 = cartpole_swingup()
    t0 = time.perf_counter()
    ref = ilqr.solve(sys_, cost, x0, U0, method="ddp", max_iter=400)
    t_ref = time.perf_counter() - t0
    J_star = ref.cost

    configs = [("mppi", K, lam) for K in (64, 256, 1024) for lam in (0.1, 1.0, 10.0, 100.0)]
    configs += [("cem", K, None) for K in (64, 256, 1024)]
    rows, traces = [], {}
    for method, K, lam in configs:
        finals, tr, wall = [], [], 0.0
        for seed in range(SEEDS):
            t0 = time.perf_counter()
            _, trace = sampling.optimise(sys_, cost, x0, U0, method, ITERS, np.random.default_rng(seed),
                                         samples=K, sigma=SIGMA, lam=lam if lam else 1.0)
            wall += time.perf_counter() - t0
            finals.append(trace[-1])
            tr.append(trace)
        finals = np.array(finals)
        traces[(method, K, lam)] = np.median(np.array(tr), axis=0)
        label = f"lambda={lam:g}" if lam else "elite 10%"
        rows.append([method.upper(), K, label, f"{np.median(finals):.2f}", f"{finals.min():.2f}",
                     f"{np.median(finals) / J_star:.3f}", f"{np.mean(finals < 1.1 * J_star):.2f}",
                     K * ITERS, f"{wall / SEEDS:.2f}"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, method in zip(axes, ("mppi", "cem")):
        for K, c in zip((64, 256, 1024), C):
            lam = 1.0 if method == "mppi" else None
            tr = traces[(method, K, lam)]
            ax.semilogy(np.arange(len(tr)) * K, tr, color=c, label=f"K={K}")
        ax.axhline(J_star, color="#52514e", linestyle="--", linewidth=1.2)
        ax.text(ITERS * 1024 * 0.98, J_star * 1.08, "iLQR/DDP optimum", ha="right", color="#52514e", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("rollouts (samples x iterations)")
        ax.set_title(("MPPI, lambda = 1" if method == "mppi" else "CEM") + f", sigma = {SIGMA:g}")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("cost of mean control sequence (median of seeds)")
    fig1 = save(fig, "e3_sampling.png")

    md = ["## E3: Sampling planners vs the gradient-based optimum", "", f"![sampling]({fig1})", "",
          f"Cart-pole swing-up, T=80, zero initial controls, {ITERS} iterations, {SEEDS} seeds, sigma={SIGMA:g}. "
          f"DDP reaches J* = {J_star:.3f} in {len(ref.history) - 1} iterations ({t_ref:.2f} s). "
          "Each DDP iteration costs about (n+m)^2 = 25 batched rollouts for derivatives plus 11 line-search rollouts.", "",
          table(["method", "K", "setting", "median final J", "best J", "median / J*", "frac. within 10%",
                 "rollouts", "s / run"], rows), ""]
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
