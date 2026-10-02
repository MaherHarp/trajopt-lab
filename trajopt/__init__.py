"""Shooting-based trajectory optimisation: iLQR, DDP, MPPI, CEM, and MPC."""

from .cost import QuadraticCost
from .systems import Acrobot, CartPole, DoubleIntegrator, Pendulum

__all__ = ["QuadraticCost", "Acrobot", "CartPole", "DoubleIntegrator", "Pendulum"]
