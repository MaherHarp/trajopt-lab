"""Benchmark swing-up problems shared by tests and experiments."""

from __future__ import annotations

import numpy as np

from .cost import QuadraticCost
from .systems import Acrobot, CartPole, Pendulum


def pendulum_swingup(T=100):
    sys = Pendulum(dt=0.05)
    cost = QuadraticCost(Q=np.diag([0.1, 0.01]), R=[[0.01]], Qf=np.diag([100.0, 10.0]),
                         goal=[np.pi, 0.0])
    return sys, cost, np.zeros(2), np.zeros((T, 1))


def cartpole_swingup(T=80, **params):
    sys = CartPole(dt=0.05, **params)
    cost = QuadraticCost(Q=np.diag([0.1, 0.1, 0.01, 0.01]), R=[[0.01]],
                         Qf=np.diag([100.0, 100.0, 10.0, 10.0]), goal=[0.0, np.pi, 0.0, 0.0])
    return sys, cost, np.zeros(4), np.zeros((T, 1))


def acrobot_swingup(T=100):
    sys = Acrobot(dt=0.05)
    cost = QuadraticCost(Q=np.diag([0.1, 0.1, 0.01, 0.01]), R=[[0.01]],
                         Qf=np.diag([200.0, 200.0, 20.0, 20.0]), goal=[np.pi, 0.0, 0.0, 0.0])
    return sys, cost, np.zeros(4), np.zeros((T, 1))


PROBLEMS = {"pendulum": pendulum_swingup, "cartpole": cartpole_swingup, "acrobot": acrobot_swingup}
