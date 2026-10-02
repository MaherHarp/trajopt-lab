import unittest

import numpy as np

from trajopt import ilqr, mpc, sampling
from trajopt.cost import QuadraticCost
from trajopt.derivatives import jacobians_cs
from trajopt.lqr import riccati
from trajopt.problems import cartpole_swingup, pendulum_swingup
from trajopt.systems import DoubleIntegrator


class TestLQREquivalence(unittest.TestCase):
    """On linear dynamics with quadratic cost, one unregularised iLQR/DDP
    iteration from any initial guess must land on the Riccati solution."""

    def setUp(self):
        self.sys = DoubleIntegrator(dt=0.1)
        Q, R, Qf = np.diag([1.0, 0.1]), np.array([[0.05]]), np.diag([50.0, 5.0])
        self.cost = QuadraticCost(Q, R, Qf, goal=[0.0, 0.0])
        self.T, self.x0 = 40, np.array([1.0, -0.5])
        A, B = jacobians_cs(self.sys.step, np.zeros((1, 2)), np.zeros((1, 1)))
        self.Klqr, self.P0 = riccati(A[0], B[0], Q, R, Qf, self.T)

    def check(self, method):
        U0 = np.random.default_rng(3).normal(size=(self.T, 1))
        res = ilqr.solve(self.sys, self.cost, self.x0, U0, method=method, mu0=0.0, max_iter=5)
        self.assertTrue(res.converged)
        self.assertEqual(len(res.history), 3)       # init, one step, convergence check
        self.assertEqual(res.history[1]["alpha"], 1.0)
        np.testing.assert_allclose(res.K, -self.Klqr, atol=1e-9)
        X = np.empty((self.T + 1, 2)); X[0] = self.x0
        U = np.empty((self.T, 1))
        for t in range(self.T):
            U[t] = -self.Klqr[t] @ X[t]
            X[t + 1] = self.sys.step(X[t], U[t])
        np.testing.assert_allclose(res.U, U, atol=1e-9)
        self.assertAlmostEqual(res.cost, 0.5 * self.x0 @ self.P0 @ self.x0, places=9)

    def test_ilqr(self):
        self.check("ilqr")

    def test_ddp(self):
        self.check("ddp")


class TestNonlinear(unittest.TestCase):
    def test_adjoint_gradient_matches_complex_step_of_total_cost(self):
        """Complex-step on J itself is an exact reference. Finite differences are
        not: swing-up costs are so ill-conditioned that central FD is wrong in
        the first digit near the optimum."""
        for make in (cartpole_swingup, pendulum_swingup):
            sys_, cost, x0, U0 = make()
            U = np.random.default_rng(0).normal(size=U0.shape)
            g = ilqr.cost_gradient(sys_, cost, sys_.rollout(x0, U), U)
            J = lambda V: cost.total(sys_.rollout(x0, V), V)
            for t in (0, 7, len(U) - 1):
                E = np.zeros(U.shape, complex); E[t, 0] = 1e-30j
                self.assertAlmostEqual(g[t, 0], J(U + E).imag / 1e-30, delta=1e-12 * (1 + abs(g[t, 0])))

    def test_monotone_descent_and_stationarity(self):
        sys_, cost, x0, U0 = pendulum_swingup()
        for method in ("ilqr", "ddp"):
            res = ilqr.solve(sys_, cost, x0, U0, method=method)
            costs = [h["cost"] for h in res.history]
            self.assertTrue(res.converged, method)
            self.assertTrue(np.all(np.diff(costs) <= 0), method)
            # Newton decrement: the predicted reduction of a full unregularised
            # step at the solution. Gradient norm is a poor test here (the
            # Hessian is extremely ill-conditioned).
            fx, fu = jacobians_cs(sys_.step, res.X[:-1], res.U)
            lx, lu, lxx, luu, lfxx, lfx = cost.derivatives(res.X, res.U)
            _, _, dV1, dV2 = ilqr.backward_pass(fx, fu, lx, lu, lxx, luu, lfxx, lfx, 0.0)
            self.assertLess(-(dV1 + dV2), 1e-8, method)
            self.assertLess(abs(res.X[-1, 0] - np.pi), 0.01)

    def test_ilqr_and_ddp_reach_same_pendulum_optimum(self):
        sys_, cost, x0, U0 = pendulum_swingup()
        a = ilqr.solve(sys_, cost, x0, U0, method="ilqr", tol=1e-13)
        b = ilqr.solve(sys_, cost, x0, U0, method="ddp", tol=1e-13)
        self.assertAlmostEqual(a.cost, b.cost, places=7)
        np.testing.assert_allclose(a.U, b.U, atol=1e-3)


class TestSampling(unittest.TestCase):
    def test_mppi_zero_noise_is_identity(self):
        sys_, cost, x0, U0 = pendulum_swingup(T=20)
        U = np.random.default_rng(0).normal(size=U0.shape)
        Un, _ = sampling.mppi_update(sys_, cost, x0, U, np.random.default_rng(1), sigma=0.0)
        np.testing.assert_allclose(Un, U, atol=0)

    def test_mppi_low_temperature_selects_best_sample(self):
        sys_, cost, x0, U0 = pendulum_swingup(T=20)
        Un, S = sampling.mppi_update(sys_, cost, x0, U0, np.random.default_rng(5), samples=64, lam=1e-9)
        eps = 1.0 * np.random.default_rng(5).standard_normal((64,) + U0.shape)
        np.testing.assert_allclose(Un, U0 + eps[np.argmin(S)], atol=1e-12)

    def test_samplers_reduce_cost(self):
        sys_, cost, x0, U0 = pendulum_swingup()
        for method in ("mppi", "cem"):
            _, trace = sampling.optimise(sys_, cost, x0, U0, method, 30, np.random.default_rng(0))
            self.assertLess(trace[-1], 0.5 * trace[0], method)


class TestClosedLoop(unittest.TestCase):
    def test_tvlqr_reproduces_plan_on_exact_model(self):
        sys_, cost, x0, U0 = cartpole_swingup()
        plan = ilqr.solve(sys_, cost, x0, U0, method="ddp")
        X, U, _ = mpc.run_closed_loop(sys_, mpc.TVLQR(plan), x0, len(U0))
        np.testing.assert_allclose(X, plan.X, atol=1e-10)

    def test_feedback_warm_start_holds_unstable_equilibrium(self):
        """Regression test for the E4 finding. Replaying shifted open-loop
        controls from a perturbed upright state makes iLQR start from a fallen
        trajectory. Rolling out the previous local policy does not."""
        model, cost, _, _ = cartpole_swingup(running=1.0, periodic=True)
        top = np.array([0.0, np.pi, 0.0, 0.0])
        dev = {}
        for warm in ("feedback", "open_loop"):
            ctrl = mpc.ILQRMPC(model, cost, np.zeros((40, 1)), iters=2, warm_start=warm)
            with np.errstate(all="ignore"):
                X, _, _ = mpc.run_closed_loop(model, ctrl, top, 60, np.random.default_rng(3), 0.3, (2, 3))
            dev[warm] = np.abs(mpc.wrap(X[:, 1] - np.pi)).max()
        self.assertLess(dev["feedback"], 0.3)
        self.assertGreater(dev["open_loop"], 1.0)

    def test_periodic_angle_cost_derivatives(self):
        _, cost, _, _ = cartpole_swingup(running=1.0, periodic=True)
        rng = np.random.default_rng(2)
        X, U = rng.normal(scale=3, size=(6, 4)), rng.normal(size=(5, 1))
        lx, lu, lxx, _, lfxx, lfx = cost.derivatives(X, U)
        h = 1e-30
        for t in range(5):
            for i in range(4):
                E = np.zeros(4, complex); E[i] = 1j * h
                self.assertAlmostEqual(lx[t, i], cost.running(X[t] + E, U[t]).imag / h, places=12)
                g = lambda x: cost._state_derivs(x, cost.Q)[0]
                np.testing.assert_allclose(lxx[t][:, i], g((X[t] + E)[None])[0].imag / h, atol=1e-12)
        # 2*pi periodicity in the angle.
        x = X[0].copy(); y = x.copy(); y[1] += 2 * np.pi
        self.assertAlmostEqual(float(cost.terminal(x)), float(cost.terminal(y)), places=10)

    def test_ilqr_mpc_swings_up_pendulum(self):
        sys_, cost, x0, U0 = pendulum_swingup(T=60)
        ctrl = mpc.ILQRMPC(sys_, cost, U0, iters=3)
        X, _, _ = mpc.run_closed_loop(sys_, ctrl, x0, 100)
        self.assertLess(abs(mpc.wrap(X[-1, 0] - np.pi)), 0.1)


if __name__ == "__main__":
    unittest.main()
