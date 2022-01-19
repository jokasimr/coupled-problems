import sys
sys.path.append('..')

import numpy as np

from solver.laplace import LaplaceOnRectangle, HeatEqOnRectangle
from sdirk import DIRK2


class Helmholtz:
    def __init__(self):
        self.heat = HeatEqOnRectangle(1.0, 1/200, width=1, height=1, f=lambda x, y: x*0, initial_condition=lambda x, y: np.sin(np.pi*x))

    def reset(self):
        self.heat.reset()
        self.heat.set_dirchlet(3, lambda x, y: 0)
        self.heat.set_dirchlet(1, lambda x, y: 0)

    def solve(self, ubar, tbar, alpha):

        self.heat.dt = alpha
        self.reset()

        self.heat.u_old = ubar.copy()
        self.heat.do_euler_step()

        return (self.heat.sol - ubar) / alpha


def step_to(stepper, u, t, tend):
    while t + stepper.dt < tend:
        u, t = stepper(u, t)
    return stepper(u, t, tend - t)


solver = Helmholtz()
stepper = DIRK2(solver, rtol=1e-4)
u0 = solver.heat.u_old.copy()

tend = 0.1
u_true = u0 * np.exp(-np.pi**2 * tend)
u, t = step_to(stepper, u0, 0, tend)

error = np.linalg.norm(u - u_true)
print(error)

solver = Helmholtz()
stepper = DIRK2(solver, rtol=1e-3)
u0 = solver.heat.u_old.copy()

tend = 0.1
u_true = u0 * np.exp(-np.pi**2 * tend)
u, t = step_to(stepper, u0, 0, tend)

error = np.linalg.norm(u - u_true)
print(error)
