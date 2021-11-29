from numba import njit
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
    return (gamma * p / rho)**0.5


def lax_friedrichs(f):

    @njit
    def _(wi, wj, v, n):

        # TODO: is this correct?
        lmax = max(
            abs(n * wi[1] / wi[0]) + speed_of_sound(wi),
            abs(n * wj[1] / wj[0]) + speed_of_sound(wj),
        )

        return 1/2 * (f(wi, v, n) + f(wj, v, n)) + 1/2 * lmax * (wj - wi)

    return _


@njit
def euler_flux(w, v, n):
    gamma = 7 / 5
    rho = w[0]
    u = w[1] / rho
    E = w[2] / rho
    e = E - u * u / 2
    p = (gamma - 1) * rho * e

    return w * (u - v) + np.array([0, p * n, p * u * n])


flux = lax_friedrichs(euler_flux) 
