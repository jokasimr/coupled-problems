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
            return i + 1

        u_gamma_left, u_gamma_right = ugl, ugr

    print(diff)
    raise NoConvergence(f"Iteration ought to have converged by {maxiter} iterations")


def bisection_optimize(f, a, b, tol, maxiter):
    fa, fb = f(a), f(b)
    if abs(fb - fa) < tol:
        return (a + b) / 2
    return bisection 


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


solver = RoomHelmholtz(theta=1.0, tol=1e-5, maxiter=50)
stepper = DIRK2(solver, rtol=1e-3, atol=0)

u0 = np.concatenate((
    left.u_old,
    middle.u_old,
    right.u_old,
))
t = 0

tend = .05

def step_to(stepper, u, t, tend):
    while t + stepper.dt < tend:
        u, t = stepper(u, t)
    return stepper(u, t, tend - t)

u_true, t = step_to(stepper, u0, t, tend)
assert t == tend
plot_room()

left.update_u_old()
right.update_u_old()
middle.update_u_old()
u = middle.sol.copy()

dts = 10**np.linspace(-2, 1, 50)
thetas = 1.0 / (1.0 + np.linspace(0, 5, 400))**4

iterations = np.zeros((len(dts), len(thetas))) + np.inf

for i, dt in enumerate(dts):
    print("dt", dt)
    for j, theta in enumerate(thetas):
        print("    theta", theta)

        if i > 0 and np.all(np.isinf(iterations[i-1, :j+1])):
            continue
        if i > 0 and j < len(thetas) - 1:
            if iterations[i-1, j+1] < iterations[i-1, j]:
                continue

        left.dt = middle.dt = right.dt = dt
        try:
            its = dn_solve(
                u[lm_interface],
                u[rm_interface],
                theta=theta,
                tol=1e-14,
                maxiter=100
            )

        except NoConvergence:
            iterations[i, j] = np.inf
            if not np.all(np.isinf(iterations[i, :])):
                print("no more")
                break
            else:
                continue

        iterations[i, j] = its
        print(np.min(iterations[i, :]))
        if its > np.min(iterations[i, :]):
            print("no more 2")
            break

        print("        ", its)



plt.plot(dts, thetas[np.argmin(iterations, axis=1)], '-x')
plt.ylabel('optimal theta')
plt.xlabel('dt')
plt.yscale('log')
plt.xscale('log')
plt.show()


