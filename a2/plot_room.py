import os
import sys
sys.path.append('..')

from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
import numpy as np

from solver.laplace import LaplaceOnRectangle


h = 0.02

left = LaplaceOnRectangle(h, 1, 1, lambda x, y: x*0)
middle = LaplaceOnRectangle(h, 1, 2, lambda x, y: x*0)
right = LaplaceOnRectangle(h, 1, 1, lambda x, y: x*0)


def reset():
    left.reset()
    middle.reset()
    right.reset()

    left_heater = left.geometry.boundary_dofs(3)[1:-1]

    left.set_dirchlet(0, lambda x, y: x*0 + 15)
    left.set_dirchlet(left_heater, lambda x, y: x*0 + 40)
    left.set_dirchlet(2, lambda x, y: x*0 + 15)

    middle_window = middle.geometry.boundary_dofs(0)[1:-1]
    middle_right_wall = middle.geometry.boundary_dofs(1)
    middle_left_wall = middle.geometry.boundary_dofs(3)
    middle_wall = np.concatenate([
        np.array([middle.geometry.boundary_dofs(0)[0]]),
        middle.geometry.boundary_dofs(2),
        middle_right_wall[:len(middle_right_wall) // 2 + 1],
        middle_left_wall[:len(middle_left_wall) // 2 + 1],
    ])

    middle.set_dirchlet(middle_window, lambda x, y: x*0 + 5)
    middle.set_dirchlet(middle_wall, lambda x, y: x*0 + 15)

    right_heater = right.geometry.boundary_dofs(1)[1:-1]
    right.set_dirchlet(0, lambda x, y: x*0 + 15)
    right.set_dirchlet(right_heater, lambda x, y: x*0 + 40)
    right.set_dirchlet(2, lambda x, y: x*0 + 15)


# left left
ll_interface = left.geometry.boundary_dofs(1)[1:-1][::-1]
# left middle
lm_interface = middle.geometry.boundary_dofs(3)[len(middle.geometry.boundary_dofs(3)) // 2 + 1:-1]
# right middle
rm_interface = middle.geometry.boundary_dofs(1)[len(middle.geometry.boundary_dofs(1)) // 2 + 1:-1]
# right rigth
rr_interface = right.geometry.boundary_dofs(3)[1:-1][::-1]


u_gamma_left = 15 * np.ones(len(ll_interface))
u_gamma_right = 15 * np.ones(len(rr_interface))

theta = 0.5


for i in range(10):

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
#plt.savefig('room/room.png')
plt.title('Temperature distribution in the room')
clean_figure()
os.makedirs("room", exist_ok=True)
save('room/room.tex')
plt.show()


k = 75

x, y = np.meshgrid(np.linspace(1-2*h, 1, k), np.linspace(0, 2*h, k))
le = left.evaluate(x, y)

x, y = np.meshgrid(np.linspace(0, 2*h, k), np.linspace(0, 2*h, k))
me = middle.evaluate(x, y)

image = np.zeros((k, 2*k)) + np.nan

image[:, :le.shape[1]] = le
image[:, le.shape[1]:] = me

b, a = image.shape
plt.imshow(
    image,
    interpolation=None,
    origin="lower",
    extent=[1-2*h, 1+2*h, 0, 2*h],
)
plt.colorbar(orientation="horizontal", pad=0.2)
plt.title('A closer look near the window')
clean_figure()
save('room/closer.tex')
plt.show()
