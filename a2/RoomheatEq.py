import os
import sys
sys.path.append('..')

#from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
import numpy as np

from solver.laplace import LaplaceOnRectangle, HeatEqOnRectangle


h = 1/20

dn_iter = 300
tol_wr = 1e-10
T_start = 0
T_end = 10000
dt = 1000

WALL = 210
WINDOW = 200
HEATER = 230

u0_l = np.vectorize(lambda x, y, q=1e-5: (
    WALL if y < q or y > 1 - q else
    HEATER if x < q else 
    WALL
))
u0_m = np.vectorize(lambda x, y, q=1e-5: (
    WALL if y > 2 - q else
    WALL if x < q and y >= 1/2 or x > 1 - q and y <= 1/2 else
    WINDOW if y < q else
    220 
))
u0_r = np.vectorize(lambda x, y, q=1e-5: (
    WALL if y < q or y > 1 - q else
    HEATER if x > 1 - q else 
    WALL
))

left = HeatEqOnRectangle(dt, h, 1, 1, lambda x, y: x*0, initial_condition=u0_l)
middle = HeatEqOnRectangle(dt, h, 1, 2, lambda x, y: x*0, initial_condition=u0_m, heat_capacitance=1300, heat_conductivity=0.0243)
#middle = HeatEqOnRectangle(dt, h, 1, 2, lambda x, y: x*0, initial_condition=u0_m, heat_capacitance=1, heat_conductivity=1)
right = HeatEqOnRectangle(dt, h, 1, 1, lambda x, y: x*0, initial_condition=u0_r)

def reset():
    
    left.reset()
    middle.reset()
    right.reset()

    left_heater = left.geometry.boundary_dofs(3)[1:-1]

    left.set_dirchlet(0, lambda x, y: x*0 + WALL)
    left.set_dirchlet(left_heater, lambda x, y: x*0 + HEATER)
    left.set_dirchlet(2, lambda x, y: x*0 + WALL)

    middle_window = middle.geometry.boundary_dofs(0)[1:-1]
    middle_right_wall = middle.geometry.boundary_dofs(1)
    middle_left_wall = middle.geometry.boundary_dofs(3)
    middle_wall = np.concatenate([
        np.array([middle.geometry.boundary_dofs(0)[0]]),
        middle.geometry.boundary_dofs(2),
        middle_right_wall[:len(middle_right_wall) // 2 + 1],
        middle_left_wall[:len(middle_left_wall) // 2 + 1],
    ])

    middle.set_dirchlet(middle_window, lambda x, y: x*0 + WINDOW)
    middle.set_dirchlet(middle_wall, lambda x, y: x*0 + WALL)

    right_heater = right.geometry.boundary_dofs(1)[1:-1]
    right.set_dirchlet(0, lambda x, y: x*0 + WALL)
    right.set_dirchlet(right_heater, lambda x, y: x*0 + HEATER)
    right.set_dirchlet(2, lambda x, y: x*0 + WALL)


# left left
ll_interface = left.geometry.boundary_dofs(1)[1:-1][::-1]
# left middle
lm_interface = middle.geometry.boundary_dofs(3)[len(middle.geometry.boundary_dofs(3)) // 2 + 1:-1]
# right middle
rm_interface = middle.geometry.boundary_dofs(1)[len(middle.geometry.boundary_dofs(1)) // 2 + 1:-1]
# right rigth
rr_interface = right.geometry.boundary_dofs(3)[1:-1][::-1]


u_gamma_left = WALL * np.ones(len(ll_interface))
u_gamma_right = WALL * np.ones(len(rr_interface))

#theta = 0.0415
#theta = 0.0234/(1+0.0234)
theta = 0.0234
time_steps = int((T_end - T_start) // dt)

for j in range(time_steps):
    for i in range(dn_iter):

        reset()

        '''
        middle.set_dirchlet(lm_interface, u_gamma_left, raw=True)
        middle.set_dirchlet(rm_interface, u_gamma_right, raw=True)
        middle.do_euler_step()

        flux_left = middle.heat_flux_at_nodes(lm_interface)
        flux_right = middle.heat_flux_at_nodes(rm_interface)

        left.set_neumann(ll_interface, -flux_left, raw=True)
        right.set_neumann(rr_interface, -flux_right, raw=True)

        left.do_euler_step()
        right.do_euler_step()

        ug = np.concatenate((u_gamma_left, u_gamma_right))

        u_gamma_left = ((1 - theta) * u_gamma_left + theta * left.sol[ll_interface].copy())
        u_gamma_right = ((1 - theta) * u_gamma_right + theta * right.sol[rr_interface].copy())

        ug_new = np.concatenate((u_gamma_left, u_gamma_right))

        if np.linalg.norm(ug - ug_new, 2) < tol_wr or i + 1 == dn_iter:
            middle.update_u_old()
            left.update_u_old()
            right.update_u_old()
            print(i)
            break

        '''

        left.set_dirchlet(ll_interface, u_gamma_left, raw=True)
        right.set_dirchlet(rr_interface, u_gamma_right, raw=True)

        left.do_euler_step()
        right.do_euler_step()

        flux_left = left.heat_flux_at_nodes(ll_interface)
        flux_right = right.heat_flux_at_nodes(rr_interface)

        middle.set_neumann(lm_interface, -flux_left, raw=True)
        middle.set_neumann(rm_interface, -flux_right, raw=True)

        middle.do_euler_step()

        ug = np.concatenate((u_gamma_left,u_gamma_right))

        u_gamma_left = ((1 - theta) * u_gamma_left + theta * middle.sol[lm_interface].copy())
        u_gamma_right = ((1 - theta) * u_gamma_right + theta * middle.sol[rm_interface].copy())

        ug_new = np.concatenate((u_gamma_left, u_gamma_right))

        if np.linalg.norm(ug - ug_new, 2) < tol_wr or i + 1 == dn_iter:
            middle.update_u_old()
            left.update_u_old()
            right.update_u_old()
            print(i)
            break

        

print(middle.sol)
print(right.sol)
print(left.sol)

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
#clean_figure()
#os.makedirs("room", exist_ok=True)
#save('room/room.tex')
#plt.savefig('room')
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
#clean_figure()
#save('room/closer.tex')
plt.show()
#plt.savefig('A closer look near the window')
