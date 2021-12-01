from tikzplotlib import save, clean_figure
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
    c = (gamma * p / rho)**0.5
    return c


def lax_friedrichs(f):

    @njit
    def _(wi, wj, v, n):

        # TODO: is this correct?
        lmax = max(
            abs(n * wi[1] / wi[0]) + speed_of_sound(wi),
            abs(n * wj[1] / wj[0]) + speed_of_sound(wi),
        )

        return 1/2 * (f(wi, v, n) + f(wj, v, n)) - 1/2 * lmax * (wj - wi)

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

    for i in range(len(w) - 1):
        f[i + 1] = flux(w[i], w[i + 1], v[i + 1], 1)

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
    plt.savefig(f'report/state_{post}.png')


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

    return ts, ws




class FVM():

    def __init__(self,x,nbr_cells,xbmove):
        self.nbr_cells = nbr_cells
        self.vol_i = (x[1]-x[0])/nbr_cells
        self.v_edge = np.zeros([nbr_cells+1])
        self.xbmove = xbmove
        self.sol = []

    
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



    
