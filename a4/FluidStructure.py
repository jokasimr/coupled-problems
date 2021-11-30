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
            abs(n * wj[1] / wj[0]) + speed_of_sound(wi),
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
        self.vol_i = (x[1]-x[0])/nbr_cells
        self.v_edge = np.zeros([nbr_cells+1])
        self.xbmove = xbmove
        self.sol = []


        return
    
    def solve(self,t_start,t_end,dt,w_0):
        
        self.sol.append(w_0)
        self.w = w_0

        while t_start <= t_end:
            self.w_next = self.w*self.vol_i

            #dxb = self.xbmove(t_start) - self.xbmove(t_start+dt)
            dxb = self.xbmove(t_start)*dt
            self.vol_i += dxb/nbr_cells
            self.v_edge = self.xbmove(t_start)/(self.nbr_cells+1)*np.linspace(0,self.nbr_cells+1,self.nbr_cells+1)

            for i in range(self.nbr_cells+1):
                
                if i == self.nbr_cells:
                    w_ghost = self.w[i-1,:]
                    w_ghost[1] = -w_ghost[1] - w_ghost[0]*self.v_edge[i]
                    self.w_next[i-1,:] -= dt*flux(self.w[i-1],w_ghost,self.v_edge[i],1)
                    print(self.v_edge[i])

                elif i == 0:
                    w_ghost = self.w[i,:]
                    w_ghost[1] = -w_ghost[1]
                    self.w_next[i,:] += dt*flux(w_ghost,self.w[i],self.v_edge[i],1)

                else:
                    self.w_next[i,:] += dt*flux(self.w[i-1],self.w[i],self.v_edge[i],1)
                    self.w_next[i-1,:] -= dt*flux(self.w[i-1],self.w[i],self.v_edge[i],1)
            
            self.w = self.w_next/self.vol_i
            t_start += dt
            self.sol.append(self.w)

        return self.sol

if __name__ == '__main__':
    nbr_cells = 5
    dt = 0.01
    t_start = 0
    t_end = 0.1
    w_0 = np.ones([nbr_cells,3])*np.array([1,0,2.5])
    

    

    #xbmove = lambda t: 1+0.1*np.sin(10*np.pi*t)
    xbmove = lambda t: np.cos(10*np.pi*t)
    #xbmove = lambda t: 0
    x = [0,1]
    
    solver = FVM(x,nbr_cells,xbmove)
    solution = solver.solve(t_start,t_end,dt,w_0)

    t = np.linspace(t_start,t_end,int(1/dt)+1)
    #plt.figure()
    #plt.plot()
    #plt.savefig("movement of x point")
    solution = np.stack(solution)
    print(solution)
    print(solution.shape)
    print("speed:")
    print(solution[:,4,1]/solution[:,4,0])

    print("speed 0:")
    print(solution[:,4,1]/solution[:,4,0])

    
    print("density:")
    print(solution[:,4,0])

    print("energy:")
    print(solution[:,4,2])




    
