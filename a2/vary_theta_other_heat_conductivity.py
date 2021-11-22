import os
import sys
sys.path.append('..')

from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
import numpy as np

from solver.laplace import LaplaceOnRectangle, coords


h = 0.02

left = LaplaceOnRectangle(h, 1, 1, lambda x, y: x*0)
middle = LaplaceOnRectangle(h, 1, 2, lambda x, y: x*0, heat_conductivity=0.0243)
right = LaplaceOnRectangle(h, 1, 1, lambda x, y: x*0)


def reset():
    left.reset()
    middle.reset()
    right.reset()

    left_heater = left.boundary(3)[1:-1]

    left.set_dirchlet(0, lambda x, y: x*0 + 15)
    left.set_dirchlet(left_heater, lambda x, y: x*0 + 40)
    left.set_dirchlet(2, lambda x, y: x*0 + 15)

    middle_window = middle.boundary(0)[1:-1]
    middle_right_wall = middle.boundary(1)
    middle_left_wall = middle.boundary(3)
    middle_wall = np.concatenate([
        np.array([middle.boundary(0)[0]]),
        middle.boundary(2),
        middle_right_wall[:len(middle_right_wall) // 2 + 1],
        middle_left_wall[:len(middle_left_wall) // 2 + 1],
    ])

    middle.set_dirchlet(middle_window, lambda x, y: x*0 + 5)
    middle.set_dirchlet(middle_wall, lambda x, y: x*0 + 15)

    right_heater = right.boundary(1)[1:-1]
    right.set_dirchlet(0, lambda x, y: x*0 + 15)
    right.set_dirchlet(right_heater, lambda x, y: x*0 + 40)
    right.set_dirchlet(2, lambda x, y: x*0 + 15)


# left left
ll_interface = left.boundary(1)[1:-1][::-1]
# left middle
lm_interface = middle.boundary(3)[len(middle.boundary(3)) // 2 + 1:-1]
# right middle
rm_interface = middle.boundary(1)[len(middle.boundary(1)) // 2 + 1:-1]
# right rigth
rr_interface = right.boundary(3)[1:-1][::-1]


def measure_convergence_rate(theta, k=20):
    u_gamma_left = 15 * np.ones(len(ll_interface))
    u_gamma_right = 15 * np.ones(len(rr_interface))

    gammas = [np.concatenate((u_gamma_left, u_gamma_right))]

    for i in range(k):

        reset()
        left.set_dirchlet(ll_interface, u_gamma_left, raw=True)
        right.set_dirchlet(rr_interface, u_gamma_right, raw=True)

        left.solve()
        right.solve()
        flux_left = left.heat_flux_at_nodes(ll_interface)
        flux_right = right.heat_flux_at_nodes(rr_interface)

        middle.set_neumann(lm_interface, -flux_left, raw=True)
        middle.set_neumann(rm_interface, -flux_right, raw=True)
        middle.solve()

        u_gamma_left = ((1 - theta) * u_gamma_left + theta * middle.sol[lm_interface].copy())
        u_gamma_right = ((1 - theta) * u_gamma_right + theta * middle.sol[rm_interface].copy())

        gammas.append(np.concatenate((u_gamma_left, u_gamma_right)))

    errors = np.array([np.linalg.norm(g - gammas[-1]) for g in gammas])
    p, _ = np.polyfit(np.arange(1, 6), np.log(errors[:5]), 1)

    return np.exp(p)


thetas = np.linspace(0.01, 0.025, 30)
ps = [measure_convergence_rate(th, k=20) for th in thetas]
plt.plot(thetas, ps, 'o')
plt.title('Convergence rate for various $\\theta$, where $\\lambda_{\\Omega_2} = 0.0243$')
plt.xlabel('$\\theta$')
plt.ylabel('approx. convergence rate')
clean_figure()
os.makedirs("theta_plot", exist_ok=True)
save('theta_plot/theta_other_heat_cond.tex')
