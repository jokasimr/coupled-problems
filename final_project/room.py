import os
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq

from solver.laplace import LaplaceOnRectangle, HeatEqOnRectangle


dx = 1/50
WALL = 210
WINDOW = 200
HEATER = 230

u0_left = np.vectorize(lambda x, y, q=1e-10: (
    WALL if y < q or y > 1 - q else
    HEATER if x < q else 
    220 if x > 1 - q else 
    WALL
))
u0_middle = np.vectorize(lambda x, y, q=1e-10: (
    WALL if y > 2 - q else
    WALL if x < q and y >= 1 or x > 1 - q and y <= 1 else
    WINDOW if y < q else
    220 
))
u0_right = np.vectorize(lambda x, y, q=1e-10: (
    WALL if y < q or y > 1 - q else
    HEATER if x > 1 - q else 
    220 if x < q else 
    WALL
))

left, middle, right = (
    HeatEqOnRectangle(
        1.0, dx, 
        f=lambda x, y: x*0,
        **kwargs)
    for kwargs in (
        dict(initial_condition=u0_left,
             width=1, height=1),
        dict(initial_condition=u0_middle,
             width=1, height=2,
             heat_capacitance=1300,
             #heat_capacitance=1,
             heat_conductivity=0.0243,
             #heat_conductivity=1
        ),
        dict(initial_condition=u0_right,
             width=1, height=1)
    )
)

def reset():

    left.reset()
    middle.reset()
    right.reset()

    left_heater = left.geometry.boundary_dofs(3)[1:-1]
    left.set_dirchlet(0, lambda x, y: WALL)
    left.set_dirchlet(left_heater, lambda x, y: HEATER)
    left.set_dirchlet(2, lambda x, y: WALL)

    middle_window = middle.geometry.boundary_dofs(0)[1:-1]
    middle_right_wall = middle.geometry.boundary_dofs(1)
    middle_left_wall = middle.geometry.boundary_dofs(3)
    middle_wall = np.concatenate([
        middle.geometry.boundary_dofs(0)[:1],
        middle_right_wall[:len(middle_right_wall) // 2 + 1],
        middle.geometry.boundary_dofs(2)[:-1],
        middle_left_wall[:len(middle_left_wall) // 2 + 1],
    ])
    middle.set_dirchlet(middle_window, lambda x, y: WINDOW)
    middle.set_dirchlet(middle_wall, lambda x, y: WALL)

    right_heater = right.geometry.boundary_dofs(1)[1:-1]
    right.set_dirchlet(0, lambda x, y: WALL)
    right.set_dirchlet(right_heater, lambda x, y: HEATER)
    right.set_dirchlet(2, lambda x, y: WALL)


# left left
ll_interface = left.geometry.boundary_dofs(1)[1:-1][::-1]
# left middle
lm_interface = middle.geometry.boundary_dofs(3)[len(middle.geometry.boundary_dofs(3)) // 2 + 1:-1]
# right middle
rm_interface = middle.geometry.boundary_dofs(1)[len(middle.geometry.boundary_dofs(1)) // 2 + 1:-1]
# right rigth
rr_interface = right.geometry.boundary_dofs(3)[1:-1][::-1]


def dn(u_gamma_left, u_gamma_right, *, theta):

    reset()

    left.set_dirchlet(ll_interface, u_gamma_left, raw=True)
    right.set_dirchlet(rr_interface, u_gamma_right, raw=True)

    left.do_euler_step()
    right.do_euler_step()

    flux_left = left.heat_flux_at_nodes(ll_interface)
    flux_right = right.heat_flux_at_nodes(rr_interface)

    middle.set_neumann(lm_interface, -flux_left, raw=True)
    middle.set_neumann(rm_interface, -flux_right, raw=True)

    middle.do_euler_step()

    u_gamma_left =  (1 - theta) * u_gamma_left  + theta * middle.sol[lm_interface]
    u_gamma_right = (1 - theta) * u_gamma_right + theta * middle.sol[rm_interface]

    return u_gamma_left, u_gamma_right


def plot_room(save_name=None):
    k = 150

    x, y = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
    le = left.evaluate(x, y)
    re = right.evaluate(x, y)

    x, y = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 2, 2*k))
    me = middle.evaluate(x, y)

    image = np.zeros((2*k, 3*k)) + np.nan

    image[:le.shape[0], :le.shape[1]] = le
    image[:, le.shape[1]:-re.shape[1]] = me
    image[-re.shape[0]:, -re.shape[1]:] = re

    plt.imshow(
        image,
        interpolation=None,
        origin="lower",
        extent=[0, 3, 0, 2],
    )
    plt.colorbar(orientation="horizontal", pad=0.2)
    plt.title('Temperature distribution in the room')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
