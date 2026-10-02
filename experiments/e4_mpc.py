"""E4: closed-loop cart-pole swing-up under model mismatch and process noise,
plus an ablation of the two receding-horizon design choices that decided
success: the warm-start policy and the angle cost."""

from __future__ import annotations

import numpy as np

from experiments.common import C, plt, save, table
from trajopt import ilqr, mpc
from trajopt.problems import cartpole_swingup

STEPS, SEEDS, NOISE, H = 120, 6, 0.3, 40
FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)


def stabilised(X):
    """Upright (mod 2*pi) for the whole final second."""
    return bool(np.all(np.abs(mpc.wrap(X[-20:, 1] - np.pi)) < 0.3))


def episode(plant, ctrl, seed):
    X, U, per = mpc.run_closed_loop(plant, ctrl, np.zeros(4), STEPS, np.random.default_rng(100 + seed),
                                    NOISE, (2, 3))
    return X, stabilised(X), per


def make_controllers(plan, model, seed, mcost):
    return {
        "open-loop plan": mpc.OpenLoop(plan),
        "TVLQR": mpc.TVLQR(plan),
        "iLQR-MPC": mpc.ILQRMPC(model, mcost, plan.U[:H], iters=2),
        "MPPI-MPC": mpc.MPPIMPC(model, mcost, plan.U[:H], np.random.default_rng(seed), samples=256, sigma=2.0),
    }


def run():
    model, cost, x0, U0 = cartpole_swingup()
    plan = ilqr.solve(model, cost, x0, U0, method="ddp", max_iter=400)
    _, mcost, _, _ = cartpole_swingup(running=1.0, periodic=True)

    names = list(make_controllers(plan, model, 0, mcost))
    succ = {n: [] for n in names}
    ms = {n: [] for n in names}
    example = {}
    for f in FACTORS:
        plant = cartpole_swingup(pole_mass=0.5 * f)[0]
        hits = {n: 0 for n in names}
        for seed in range(SEEDS):
            for n, ctrl in make_controllers(plan, model, seed, mcost).items():
                X, ok, per = episode(plant, ctrl, seed)
                hits[n] += ok
                ms[n].append(per * 1e3)
                if f == 1.5 and seed == 0:
                    example[n] = X
        for n in names:
            succ[n].append(hits[n] / SEEDS)

    # Ablation on the nominal plant.
    _, raw_cost, _, _ = cartpole_swingup(running=1.0, periodic=False)
    _, mcost10, _, _ = cartpole_swingup(running=10.0, periodic=True)
    variants = [
        ("iLQR-MPC", "feedback", "periodic, running x10 (default)", lambda s: mpc.ILQRMPC(model, mcost, plan.U[:H], 2)),
        ("iLQR-MPC", "open-loop", "periodic", lambda s: mpc.ILQRMPC(model, mcost, plan.U[:H], 2, warm_start="open_loop")),
        ("iLQR-MPC", "feedback", "raw quadratic", lambda s: mpc.ILQRMPC(model, raw_cost, plan.U[:H], 2)),
        ("iLQR-MPC", "open-loop", "raw quadratic", lambda s: mpc.ILQRMPC(model, raw_cost, plan.U[:H], 2, warm_start="open_loop")),
        ("iLQR-MPC", "feedback", "periodic, offline weights", lambda s: mpc.ILQRMPC(model, cartpole_swingup(periodic=True)[1], plan.U[:H], 2)),
        ("MPPI-MPC", "shifted mean", "periodic, running x10 (default)", lambda s: mpc.MPPIMPC(model, mcost, plan.U[:H], np.random.default_rng(s))),
        ("MPPI-MPC", "shifted mean", "periodic, running x100", lambda s: mpc.MPPIMPC(model, mcost10, plan.U[:H], np.random.default_rng(s))),
        ("MPPI-MPC, K=1024", "shifted mean", "periodic, running x100", lambda s: mpc.MPPIMPC(model, mcost10, plan.U[:H], np.random.default_rng(s), samples=1024)),
        ("MPPI-MPC, H=30", "shifted mean", "periodic, running x100", lambda s: mpc.MPPIMPC(model, mcost10, plan.U[:30], np.random.default_rng(s))),
        ("MPPI-MPC, H=30, K=1024", "shifted mean", "periodic, running x100", lambda s: mpc.MPPIMPC(model, mcost10, plan.U[:30], np.random.default_rng(s), samples=1024)),
    ]
    abl_rows = []
    for ctrl_name, warm, cost_name, mk in variants:
        oks = [episode(model, mk(s), s)[1] for s in range(SEEDS)]
        abl_rows.append([ctrl_name, warm, cost_name, f"{np.mean(oks):.2f}"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for (n, c) in zip(names, C):
        axes[0].plot(FACTORS, succ[n], "o-", color=c, label=n)
    axes[0].set_xlabel("plant pole mass / model pole mass")
    axes[0].set_ylabel(f"success rate ({SEEDS} noise seeds)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks(FACTORS)
    axes[0].set_title(f"Swing-up and hold, process noise sigma = {NOISE:g}")
    axes[0].legend(loc="lower left")
    t = np.arange(STEPS + 1) * model.dt
    for (n, c) in zip(names, C):
        axes[1].plot(t, np.abs(mpc.wrap(example[n][:, 1] - np.pi)), color=c, label=n)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("angle from upright [rad]")
    axes[1].set_title("Example episode, pole mass x1.5")
    fig1 = save(fig, "e4_mpc.png")

    rows = [[n] + [f"{s:.2f}" for s in succ[n]] + [f"{np.median(ms[n]):.2f}"] for n in names]
    md = ["## E4: Closed-loop robustness", "", f"![mpc]({fig1})", "",
          f"Cart-pole, {STEPS} control steps ({STEPS * model.dt:g} s), Gaussian velocity noise of std {NOISE:g} sqrt(dt) "
          f"per step, {SEEDS} seeds per cell. The offline DDP plan uses T=80 and the nominal model. Both MPC "
          f"controllers use H={H}, the nominal model, and the periodic cost with 10x the offline running weight. "
          "iLQR-MPC runs 2 iterations per step; MPPI-MPC draws 256 samples per step. "
          "Success means within 0.3 rad of upright (mod 2 pi) for the entire final second.", "",
          table(["controller"] + [f"mass x{f:g}" for f in FACTORS] + ["median ms / step"], rows), "",
          "**Ablation** (nominal plant, same noise and seeds):", "",
          table(["controller", "warm start", "angle cost", "success rate"], abl_rows), ""]
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
