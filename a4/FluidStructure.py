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

        return 1/2 * (f(wi, v, n) + f(wj, v, -n)) - 1/2 * lmax * (wj - wi)

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
        self.vol_list = []
        self.vol_list.append(self.vol_i)

        while t_start <= t_end:
            self.w_next = self.w*self.vol_i

            #dxb = self.xbmove(t_start)*dt
            #dxb = 0.1*(-np.sin(10*np.pi*t_start)+np.sin(10*np.pi*t_start+dt))
            dxb = self.xbmove(t_start)*dt

            self.vol_i -= dxb/nbr_cells
            self.vol_list.append(self.vol_i)

            self.v_edge = self.xbmove(t_start)/(self.nbr_cells+1)*np.linspace(0,self.nbr_cells+1,self.nbr_cells+1)

            for i in range(self.nbr_cells+1):
                
                if i == self.nbr_cells:
                    w_ghost = self.w[i-1,:].copy()
                    w_ghost[1] = -w_ghost[1]
                    #w_ghost = np.array([1,-2,5])*self.vol_i
                    #self.w_next[i-1,:] += dt*flux(self.w[i-1],w_ghost,self.v_edge[i],-1)
                    
                elif i == 0:
                    w_ghost = self.w[i,:].copy()
                    w_ghost[1] = -w_ghost[1]
                    #w_ghost = np.array([1,-2,5])*self.vol_i
                    #self.w_next[i,:] += dt*flux(w_ghost,self.w[i],self.v_edge[i],1)

                else:
                    self.w_next[i,:] -= dt*flux(self.w[i],self.w[i-1],self.v_edge[i],1)
                    self.w_next[i-1,:] += dt*flux(self.w[i],self.w[i-1],self.v_edge[i],1)
                    #self.w_next[i-1,:] -= dt*flux(self.w[i-1],self.w[i],self.v_edge[i],-1)
            
            #
            self.w_next[0,1] = 0
            self.w_next[-1,1] = 0

            self.w = self.w_next/self.vol_i
            t_start += dt
            self.sol.append(self.w)

        return self.sol,np.array(self.vol_list)

if __name__ == '__main__':
    
    nbr_cells = 20
    dt = 0.001
    t_start = 0
    t_end = 5
    w_0 = np.ones([nbr_cells,3])*np.array([1,0,2.5])
    #w_0[2:10,0] = 5
    

    

    #xbmove = lambda t: 1+0.1*np.sin(10*np.pi*t)
    xbmove = lambda t: np.pi*np.cos(10*np.pi*t)
    #xbmove = lambda t: 0
    #xbmove = lambda t: -100
    x = [0,1]
    
    solver = FVM(x,nbr_cells,xbmove)
    solution,volumes = solver.solve(t_start,t_end,dt,w_0)

    t = np.linspace(t_start,t_end,int(1/dt)+1)
    #plt.figure()
    #plt.plot()
    #plt.savefig("movement of x point")

    solution = np.stack(solution)
    print(solution)
    print(solution.shape)
    print("speed:")
    print(solution[:,:,1]/solution[:,:,0])

    plt.figure()
    plt.imshow(solution[:,:,1]/solution[:,:,0],interpolation=None,
    origin="lower",extent=[0, 1, 0, 1],)
    plt.colorbar()
    plt.xlabel("Cells")
    plt.ylabel("time")
    plt.savefig("speed")

    plt.figure()
    plt.imshow(solution[:,:,0],interpolation=None,
    origin="lower",extent=[0, 1, 0, 1],)
    plt.colorbar()
    plt.xlabel("Cells")
    plt.ylabel("time")

    plt.savefig("density")

    vol = []
    for i in range(nbr_cells):
        vol.append(volumes)
    vol = np.stack(vol)
    plt.figure()
    plt.imshow(np.transpose(vol)*solution[:,:,0],interpolation=None,
    origin="lower",extent=[0, 1, 0, 1],)
    plt.colorbar()
    plt.savefig("mass")

    plt.figure()
    plt.plot(volumes)
    plt.savefig("Volume")
    print(1/nbr_cells)

    plt.figure()
    plt.plot(np.sum(np.transpose(vol)*solution[:,:,0],1))
    plt.xlabel("number of time steps")
    plt.ylabel("Total mass")
    plt.savefig("mass_increase")

    plt.figure()
    plt.plot(np.sum(solution[:,:,0],1))
    plt.savefig("density_increase")

    plt.figure()
    plt.imshow(solution[:,:,2],interpolation=None,
    origin="lower",extent=[0, 1, 0, 1],)
    plt.colorbar()
    plt.savefig("energy")
    plt.figure()



    
