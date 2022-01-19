import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from room import dn, lm_interface, rm_interface, left, right, middle, plot_room
from sdirk import DIRK2


class NoConvergence(Exception):
    pass


def dn_solve(u_gamma_left, u_gamma_right, *, theta, tol, maxiter):

    norm = (
        np.linalg.norm(u_gamma_left, 2) +
        np.linalg.norm(u_gamma_right, 2)
    )

    for i in range(maxiter):

        ugl, ugr = dn(u_gamma_left, u_gamma_right, theta=theta)

        diff = (
            np.linalg.norm(ugl - u_gamma_left, 2) +
            np.linalg.norm(ugr - u_gamma_right, 2)
        )

        if  diff < tol * norm:
            print("iterations", i)
            return

        u_gamma_left, u_gamma_right = ugl, ugr

    print("fail", diff)

    raise NoConvergence(f"Iteration ought to have converged by {maxiter} iterations")


class RoomHelmholtz:
    def __init__(self, *, theta, tol, maxiter=100):
        self.theta = theta
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, ubar, tbar, alpha):

        l, m, r = len(left.sol), len(middle.sol), len(right.sol)

        left.dt = middle.dt = right.dt = alpha
        left.u_old = ubar[:l]
        middle.u_old = ubar[l:l+m]
        right.u_old = ubar[l+m:l+m+r]

        print("theta", self.theta(alpha), "alpha", alpha)
        dn_solve(
            middle.u_old[lm_interface],
            middle.u_old[rm_interface],
            theta=self.theta(alpha) if hasattr(self.theta, '__call__') else self.theta,
            tol=alpha * self.tol,
            maxiter=self.maxiter
        )

        u = np.concatenate((
            left.sol,
            middle.sol,
            right.sol,
        ))

        return (u - ubar) / alpha


left.sol = left.u_old.copy()
middle.sol = middle.u_old.copy()
right.sol = right.u_old.copy()

tsmall = 0.12
tlarge = 0.92

theta = lambda dt: (
    1 if dt < tsmall else
    1 - (1 - 0.0243/1.0243/2)*(dt - tsmall)/(tlarge - tsmall) if tsmall <= dt < tlarge else
    0.0243/1.0243/2)
solver = RoomHelmholtz(theta=theta, tol=1e-3, maxiter=50)
stepper = DIRK2(solver, rtol=1e0, atol=0)

u0 = np.concatenate((
    left.u_old,
    middle.u_old,
    right.u_old,
))
t = 0

tend = 100.0

def step_to(stepper, u, t, tend):
    while t + stepper.dt < tend:
        u, t = stepper(u, t)
        print(t)
    return stepper(u, t, tend - t)

u_true, t = step_to(stepper, u0, t, tend)
print(t)
assert t == tend
plot_room()
