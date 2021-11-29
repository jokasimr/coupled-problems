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

class FVM():

    def __init__(self,x,nbr_cells,xbmove):
        self.nbr_cells = nbr_cells
        self.vol_i = (x[0]-x[1])/nbr_cells
        self.v_edge = np.zeros([nbr_cells+1])
        self.xbmove = xbmove
        self.sol = []


        return
    
    def solve(self,t_start,t_end,dt,w_0):
        
        
        self.sol.append(w_0)
        self.w = self.vol_i*w_0

        while t_start <= t_end:

            dxb = self.xbmove(t_start) - self.xbmove(t_start+dt)
            self.vol_i += dxb
            self.v_edge = dxb/dt/(self.nbr_cells+1)*self.linspace(0,self.nbr_cells+1,self.nbr_cells+1)

            for i in range(self.nbr_cells+1):
                
                if i == self.nbr_cells:
                    self.w[i-1,:] -= dt*flux(self.w[i-1],self.v_edge[i])
                elif i == 0:
                    self.w[i,:] += dt*flux(self.w[i],self.v_edge[i])
                else:
                    self.w[i,:] += dt*flux(self.w[i],self.v_edge[i])
                    self.w[i-1,:] -= dt*flux(self.w[i-1],self.v_edge[i])
                

            t_start += dt
            self.sol.append(self.w/self.vol_i)

        return self.sol

if __name__ == 'main':
    nbr_cells = 10
    dt = 0.1
    t_start = 0
    t_end = 5
    w_0 = np.ones([nbr_cells,3])*np.array([1,0,2.5])

    xbmove = lambda t: 1+0.1*np.sin(10*np.pi*t)
    x = [0,xbmove(0)]
    
    solver = FVM(x,nbr_cells,xbmove)
    solution = solver.solve(t_start,t_end,dt,w_0)
