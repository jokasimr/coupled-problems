import numpy as np
#from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
def flux(w,v):

    return 1
class FVM():

    def __init__(self,x,nbr_cells,xbmove):
        self.nbr_cells = nbr_cells
        self.vol_i = (x[0]-x[1])/nbr_cells
        self.edges = np.linspace(x[0],x[1],nbr_cells+2)
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
                    self.w[i-1] -= dt*flux(self.w[i-1],self.v_edge[i])
                elif i == 0:
                    self.w[i] += dt*flux(self.w[i],self.v_edge[i])
                else:
                    self.w[i] += dt*flux(self.w[i],self.v_edge[i])
                    self.w[i-1] -= dt*flux(self.w[i-1],self.v_edge[i])
                

            t_start += dt
            self.sol.append(self.w/self.vol_i)

        return self.sol



