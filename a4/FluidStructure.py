from tikzplotlib import save, clean_figure
from numba import njit, guvectorize, float64
import numpy as np
import matplotlib.pyplot as plt


@njit
def speed_of_sound(w):
    gamma = 7 / 5
    rho = w[0]
    u = w[1] / rho
    E = w[2] / rho
    e = E - u * u / 2
    p = (gamma - 1) * rho * e
    c = (gamma * p / rho)**0.5
    return c


def lax_friedrichs(f):

    @guvectorize(['void(float64[:], float64[:], float64, float64, float64[:])'],
                 '(n), (n), (), () -> (n)',
                 nopython=True, target="cpu")
    def _(wi, wj, v, n, out):

        # TODO: is this correct?
        lmax = max(
            abs(n * wi[1] / wi[0]) + speed_of_sound(wi),
            abs(n * wj[1] / wj[0]) + speed_of_sound(wi),
        )

        out[:] = 1/2 * (f(wi, v, n) + f(wj, v, n)) - 1/2 * lmax * (wj - wi)

    return _


@njit
def euler_flux(w, v, n):
    gamma = 7 / 5
    rho = w[0]
    u = w[1] / rho
    E = w[2] / rho
    e = E - u * u / 2
    p = (gamma - 1) * rho * e

    return w * (u - v) * n + np.array([0, p * n, p * u * n])


flux = lax_friedrichs(euler_flux) 


def fvm_update(w, v, omega, new_omega, e, dt):

    f = np.zeros((len(w) + 1, 3))

    f[1:-1, :] = flux(w[:-1], w[1:], v[1:-1], 1)

    # boundary fluxes
    f[0] = euler_flux(np.array([w[0, 0], w[0, 0] * v[0], w[0, 2]]), v[0], 1)
    f[-1] = euler_flux(np.array([w[-1, 0], w[-1, 0] * v[-1], w[-1, 2]]), v[-1], 1)

    w = w * omega 
    w -= dt * e * f[1:]
    w += dt * e * f[:-1]

    return w / new_omega


def plot_solution(ts, ws, post=''):

    k, N, _ = ws.shape

    gamma = 7 / 5
    rho = ws[:, :, 0]
    u = ws[:, :, 1] / rho
    E = ws[:, :, 2] / rho
    e = E - u * u / 2
    p = (gamma - 1) * rho * e

    fig, ax = plt.subplots(1, 3, figsize=(6, 8))

    ax[0].set_title("Density")
    ax[0].imshow(rho, extent=[0, 1, ts[-1], ts[0]])
    ax[0].set_ylabel("Time [s]")

    ax[1].set_title("Velocity")
    ax[1].imshow(u, extent=[0, 1, ts[-1], ts[0]])
    ax[1].set_xlabel("$x / x_{max}(t)$")

    ax[2].set_title("Pressure")
    ax[2].imshow(p, extent=[0, 1, ts[-1], ts[0]])
    plt.tight_layout()
    #clean_figure()
    #save(f'report/state_{post}.tex')
    #plt.savefig(f'report/state_{post}.png')
    plt.show()


def position(t):
    return 1 + 0.1 * np.sin(10 * np.pi * t)

def velocity(t):
    return np.pi * np.cos(10 * np.pi * t)

def node_velocities(t, N):
    return np.arange(0, N + 1) * velocity(t) / N

def cell_volume(t, N):
    return position(t) / N


def time_iteration(dt, t0, t1, w0):

    N = len(w0)

    t = 0
    ts = [t]
    ws = [w0]

    for i in range(int((t1 - t0) / dt)):

        v = node_velocities(t, N)
        omega = cell_volume(t, N)
        omega_next = cell_volume(t + dt, N)

        t += dt
        ts.append(t)

        ws.append(
            fvm_update(ws[-1], v, omega, omega_next, 1, dt)
        )

    return ts, np.array(ws)
