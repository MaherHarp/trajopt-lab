import unittest

import numpy as np

from trajopt.derivatives import hessians_cs, jacobians_cs, jacobians_fd, rk4_tangent
from trajopt.systems import Acrobot, CartPole, Pendulum


class TestDynamics(unittest.TestCase):
    def test_rk4_fourth_order(self):
        """Halving dt cuts the global error by ~2^4 (vs a dt/64 reference)."""
        x0, T_end = np.array([2.5, 0.0]), 2.0
        U = lambda N: np.full((N, 1), 0.3)
        errs = []
        for dt in (0.02, 0.01, 0.005):
            sys_ = Pendulum(dt=dt)
            N = int(round(T_end / dt))
            ref = Pendulum(dt=dt / 64).rollout(x0, U(N * 64))[-1]
            errs.append(np.linalg.norm(sys_.rollout(x0, U(N))[-1] - ref))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        np.testing.assert_allclose(orders, 4.0, atol=0.1)

    def test_energy_conserved_without_input(self):
        """A sign error in any inertia or Coriolis term shows up as O(1) drift."""
        cases = [(Pendulum(damping=0.0, dt=0.01), [2.0, 0.0]),
                 (CartPole(dt=0.01), [0.0, 2.0, 0.5, 0.0]),
                 (Acrobot(dt=0.005), [1.5, -0.7, 0.0, 0.3])]
        for sys_, x0 in cases:
            X = sys_.rollout(np.array(x0), np.zeros((1000, 1)))
            E = sys_.energy(X)
            drift = np.max(np.abs(E - E[0])) / (1 + abs(E[0]))
            self.assertLess(drift, 1e-5, sys_.name)

    def test_cartpole_upright_is_equilibrium(self):
        sys_ = CartPole()
        np.testing.assert_allclose(sys_.f(np.array([0.3, np.pi, 0, 0]), np.zeros(1)), 0, atol=1e-12)


class TestDerivatives(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.sys = Pendulum(dt=0.05)
        self.X = rng.uniform(-3, 3, (20, 2))
        self.U = rng.uniform(-2, 2, (20, 1))

    def test_complex_step_matches_exact_rk4_tangent(self):
        fx, fu = jacobians_cs(self.sys.step, self.X, self.U)
        for t in range(len(self.U)):
            A, B = rk4_tangent(self.sys, self.X[t], self.U[t])
            np.testing.assert_allclose(fx[t], A, atol=1e-14)
            np.testing.assert_allclose(fu[t], B, atol=1e-14)

    def test_finite_differences_are_less_accurate(self):
        fx, _ = jacobians_cs(self.sys.step, self.X, self.U)
        fwd, _ = jacobians_fd(self.sys.step, self.X, self.U, h=1e-6, scheme="forward")
        cen, _ = jacobians_fd(self.sys.step, self.X, self.U, h=1e-5, scheme="central")
        e_fwd, e_cen = np.abs(fwd - fx).max(), np.abs(cen - fx).max()
        self.assertGreater(e_fwd, 1e-9)
        self.assertLess(e_cen, e_fwd)
        self.assertLess(e_cen, 1e-8)

    def test_hessian_tensor(self):
        """d2f vs central differences of the exact RK4 tangent; symmetric."""
        fxx, fux, fuu = hessians_cs(self.sys.step, self.X, self.U)
        h = 1e-5
        for t in range(5):
            x, u = self.X[t], self.U[t]
            for a in range(2):
                e = np.eye(2)[a] * h
                Ap, Bp = rk4_tangent(self.sys, x + e, u)
                Am, Bm = rk4_tangent(self.sys, x - e, u)
                np.testing.assert_allclose(fxx[t][:, :, a], (Ap - Am) / (2 * h), atol=1e-7)
            Ap, Bp = rk4_tangent(self.sys, x, u + h)
            Am, Bm = rk4_tangent(self.sys, x, u - h)
            np.testing.assert_allclose(fux[t][:, 0, :], (Ap - Am) / (2 * h), atol=1e-7)
            np.testing.assert_allclose(fuu[t][:, 0, 0], ((Bp - Bm) / (2 * h))[:, 0], atol=1e-7)
        np.testing.assert_allclose(fxx, np.swapaxes(fxx, -1, -2), atol=0)

    def test_batched_jacobians_on_4d_system(self):
        """The CS result is insensitive to h (no cancellation) and agrees with central FD."""
        sys_ = Acrobot()
        rng = np.random.default_rng(1)
        X, U = rng.normal(size=(7, 4)), rng.normal(size=(7, 1))
        a1, b1 = jacobians_cs(sys_.step, X, U, h=1e-30)
        a2, b2 = jacobians_cs(sys_.step, X, U, h=1e-200)
        np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-14)
        np.testing.assert_allclose(b1, b2, rtol=0, atol=1e-14)
        ac, _ = jacobians_fd(sys_.step, X, U, h=1e-5, scheme="central")
        np.testing.assert_allclose(a1, ac, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
