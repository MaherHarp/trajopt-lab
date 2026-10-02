"""Underactuated benchmark systems with RK4 zero-order-hold discretisation.

Every dynamics function is vectorised over leading axes and uses only
analytic operations (sin, cos, +, *, /), so it accepts complex inputs. That
is what makes complex-step differentiation in ``derivatives.py`` exact to
machine precision.

Angle conventions: theta = 0 is hanging straight down, theta = pi is upright.
"""

from __future__ import annotations

import numpy as np


class System:
    """Continuous dynamics xdot = f(x, u) integrated with classical RK4."""

    name = "system"
    n = 0
    m = 0

    def __init__(self, dt: float):
        self.dt = dt

    def f(self, x, u):
        raise NotImplementedError

    def step(self, x, u):
        """One RK4 step with the control held constant over the interval."""
        dt = self.dt
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout(self, x0, U):
        """x0: (..., n), U: (..., T, m) -> X: (..., T+1, n)."""
        U = np.asarray(U)
        T = U.shape[-2]
        batch = np.broadcast_shapes(np.shape(x0)[:-1], U.shape[:-2])
        X = np.empty(batch + (T + 1, self.n), dtype=np.result_type(x0, U))
        X[..., 0, :] = x0
        for t in range(T):
            X[..., t + 1, :] = self.step(X[..., t, :], U[..., t, :])
        return X

    def energy(self, x):
        raise NotImplementedError


class DoubleIntegrator(System):
    """p'' = u. Linear, so RK4 is exact and iLQR reduces to LQR."""

    name = "double_integrator"
    n, m = 2, 1

    def __init__(self, dt: float = 0.1):
        super().__init__(dt)

    def f(self, x, u):
        return np.stack([x[..., 1], u[..., 0] + 0.0 * x[..., 0]], axis=-1)

    def fjac(self, x, u):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        return A, B


class Pendulum(System):
    """Torque-actuated pendulum: ml^2 th'' = -mgl sin th - b th' + u."""

    name = "pendulum"
    n, m = 2, 1

    def __init__(self, mass=1.0, length=1.0, damping=0.1, g=9.81, dt=0.05):
        super().__init__(dt)
        self.mass, self.length, self.damping, self.g = mass, length, damping, g

    def f(self, x, u):
        th, w = x[..., 0], x[..., 1]
        ml2 = self.mass * self.length**2
        wd = -(self.g / self.length) * np.sin(th) + (u[..., 0] - self.damping * w) / ml2
        return np.stack([w, wd], axis=-1)

    def fjac(self, x, u):
        ml2 = self.mass * self.length**2
        A = np.array([[0.0, 1.0], [-(self.g / self.length) * np.cos(x[0]), -self.damping / ml2]])
        B = np.array([[0.0], [1.0 / ml2]])
        return A, B

    def energy(self, x):
        th, w = x[..., 0], x[..., 1]
        ml = self.mass * self.length
        return 0.5 * ml * self.length * w**2 - ml * self.g * np.cos(th)


class CartPole(System):
    """Force-actuated cart with a point-mass pole. State [p, th, p', th']."""

    name = "cartpole"
    n, m = 4, 1

    def __init__(self, cart_mass=1.0, pole_mass=0.5, length=0.6, g=9.81, dt=0.05):
        super().__init__(dt)
        self.mc, self.mp, self.l, self.g = cart_mass, pole_mass, length, g

    def f(self, x, u):
        th, pd, thd = x[..., 1], x[..., 2], x[..., 3]
        F = u[..., 0]
        s, c = np.sin(th), np.cos(th)
        mc, mp, l, g = self.mc, self.mp, self.l, self.g
        den = mc + mp * s * s
        pdd = (F + mp * s * (l * thd * thd + g * c)) / den
        thdd = (-F * c - mp * l * thd * thd * c * s - (mc + mp) * g * s) / (l * den)
        return np.stack([pd, thd, pdd, thdd], axis=-1)

    def energy(self, x):
        th, pd, thd = x[..., 1], x[..., 2], x[..., 3]
        mc, mp, l = self.mc, self.mp, self.l
        ke = 0.5 * (mc + mp) * pd**2 + mp * l * np.cos(th) * pd * thd + 0.5 * mp * l**2 * thd**2
        return ke - mp * self.g * l * np.cos(th)


class Acrobot(System):
    """Two-link arm actuated only at the elbow. State [q1, q2, q1', q2'].

    q1 is measured from hanging down and q2 is relative to link 1. The links
    are uniform rods, and the inertias are taken about the joints (Spong 1995).
    """

    name = "acrobot"
    n, m = 4, 1

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, damping=0.0, g=9.81, dt=0.05):
        super().__init__(dt)
        self.m1, self.m2, self.l1, self.l2 = m1, m2, l1, l2
        self.lc1, self.lc2 = l1 / 2, l2 / 2
        self.I1, self.I2 = m1 * l1**2 / 3, m2 * l2**2 / 3
        self.damping, self.g = damping, g

    def mass_matrix(self, q2):
        c2 = np.cos(q2)
        a = self.m2 * self.l1 * self.lc2
        M11 = self.I1 + self.I2 + self.m2 * self.l1**2 + 2 * a * c2
        M12 = self.I2 + a * c2
        return M11, M12, self.I2 + 0.0 * c2

    def f(self, x, u):
        q1, q2, q1d, q2d = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        M11, M12, M22 = self.mass_matrix(q2)
        h = self.m2 * self.l1 * self.lc2 * np.sin(q2)
        s1, s12 = np.sin(q1), np.sin(q1 + q2)
        g = self.g
        # r = tau_gravity + B u - C(q, qd) qd - damping * qd
        r1 = (-self.m1 * g * self.lc1 * s1 - self.m2 * g * (self.l1 * s1 + self.lc2 * s12)
              + h * (2 * q1d * q2d + q2d * q2d) - self.damping * q1d)
        r2 = -self.m2 * g * self.lc2 * s12 + u[..., 0] - h * q1d * q1d - self.damping * q2d
        det = M11 * M22 - M12 * M12
        q1dd = (M22 * r1 - M12 * r2) / det
        q2dd = (M11 * r2 - M12 * r1) / det
        return np.stack([q1d, q2d, q1dd, q2dd], axis=-1)

    def energy(self, x):
        q1, q2, q1d, q2d = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        M11, M12, M22 = self.mass_matrix(q2)
        ke = 0.5 * (M11 * q1d**2 + 2 * M12 * q1d * q2d + M22 * q2d**2)
        pe = (-self.m1 * self.g * self.lc1 * np.cos(q1)
              - self.m2 * self.g * (self.l1 * np.cos(q1) + self.lc2 * np.cos(q1 + q2)))
        return ke + pe
