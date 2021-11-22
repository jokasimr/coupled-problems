import os
import sys
sys.path.append('..')

#from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
import numpy as np

from solver.laplace import LaplaceOnRectangle, coords, HeatEqOnRectangle


h = 1/20

dn_iter = 50
tol_wr = 1e-10
T_start =0
T_end = 100
dt = 0.1

u0_l = np.ones([(int(1/h)+1)**2])*210
u0_m = np.ones([2*(int(1/h)+1)**2])*220
u0_r = np.ones([(int(1/h)+1)**2])*210

left = HeatEqOnRectangle(u0_l,dt,h, 1, 1, lambda x, y: x*0)
#middle = HeatEqOnRectangle(u0_m,dt,h, 1, 2, lambda x, y: x*0,alpha = 1300,lamda = 0.0243)
middle = HeatEqOnRectangle(u0_m,dt,h, 1, 2, lambda x, y: x*0,alpha = 1,lamda = 1)

right = HeatEqOnRectangle(u0_r,dt,h, 1, 1, lambda x, y: x*0)

def reset():
    
    left.reset()
    middle.reset()
    right.reset()

    left_heater = left.boundary(3)[1:-1]

    left.set_dirchlet(0, lambda x, y: x*0 + 210)
    left.set_dirchlet(left_heater, lambda x, y: x*0 + 230)
    left.set_dirchlet(2, lambda x, y: x*0 + 210)

    middle_window = middle.boundary(0)[1:-1]
    middle_right_wall = middle.boundary(1)
    middle_left_wall = middle.boundary(3)
    middle_wall = np.concatenate([
        np.array([middle.boundary(0)[0]]),
        middle.boundary(2),
        middle_right_wall[:len(middle_right_wall) // 2 + 1],
        middle_left_wall[:len(middle_left_wall) // 2 + 1],
    ])

    middle.set_dirchlet(middle_window, lambda x, y: x*0 + 200)
    middle.set_dirchlet(middle_wall, lambda x, y: x*0 + 210)

    right_heater = right.boundary(1)[1:-1]
    right.set_dirchlet(0, lambda x, y: x*0 + 210)
    right.set_dirchlet(right_heater, lambda x, y: x*0 + 230)
    right.set_dirchlet(2, lambda x, y: x*0 + 210)


# left left
ll_interface = left.boundary(1)[1:-1][::-1]
# left middle
lm_interface = middle.boundary(3)[len(middle.boundary(3)) // 2 + 1:-1]
# right middle
rm_interface = middle.boundary(1)[len(middle.boundary(1)) // 2 + 1:-1]
# right rigth
rr_interface = right.boundary(3)[1:-1][::-1]


u_gamma_left = 15 * np.ones(len(ll_interface))
u_gamma_right = 15 * np.ones(len(rr_interface))

theta = 0.5
time_steps = int((T_end-T_start)/dt)+1
time_steps = 100
for j in range(time_steps):
    for i in range(dn_iter):

        reset()
        left.set_dirchlet(ll_interface, u_gamma_left, raw=True)
        right.set_dirchlet(rr_interface, u_gamma_right, raw=True)

        left.do_euler_step()
        right.do_euler_step()
        flux_left = left.heat_flux_at_nodes(ll_interface)
        flux_right = right.heat_flux_at_nodes(rr_interface)

        middle.set_neumann(lm_interface, flux_left, raw=True)
        middle.set_neumann(rm_interface, flux_right, raw=True)
        middle.do_euler_step()

        ug = np.concatenate((u_gamma_left,u_gamma_right))
        ug_new = np.concatenate((middle.sol[lm_interface].copy(),middle.sol[rm_interface].copy()))

        if np.linalg.norm(ug-ug_new,2)<tol_wr or i+1 == dn_iter:
            middle.update_u_old()
            left.update_u_old()
            right.update_u_old()
            print(i)
            break


        u_gamma_left = ((1 - theta) * u_gamma_left + theta * middle.sol[lm_interface].copy())
        u_gamma_right = ((1 - theta) * u_gamma_right + theta * middle.sol[rm_interface].copy())

        
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
plt.savefig('room')


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
#plt.show()
plt.savefig('A closer look near the window')
