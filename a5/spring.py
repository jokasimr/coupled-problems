import numpy as np
from matplotlib import pyplot as plt
class spring():

    def __init__(self):
        self.ms = 100
        self.A = 0.1
        self.k = 1e2
        self.p0 = 1e5

        self.mu = 1e-3
        self.tol_new = 1e-8
        self.max_iter_newton = 1e3

        return
    
    def f(self,x,t,p):
        return np.array([x[1],1/self.ms*(self.A*(p-self.p0)-self.k*x[0])])
    
    def calc_jac(self,Yi,dt,t,x,p,alpha):
            
            def f_hat(x):   
                return x-self.f(Yi+alpha*dt*x,t,p)
            
            ans = np.zeros([len(x),len(x)])
            for i in range(0,len(x)):
                temp = x.copy()
                temp[i] = x[i].copy()+self.mu
                for j in range(len(x)):
                    ans[:,i] = (f_hat(temp)-f_hat(x))/self.mu
            return ans

    def Newton(self,Yi,dt,t,x,p,alpha):
        r = self.tol_new+1
        i=0
        while r>self.tol_new:
            i += 1
            x_old = x
            jac = self.calc_jac(Yi,dt,t,x,p,alpha)
            x = x - np.linalg.solve(jac,x-self.f(Yi+dt*alpha*x,t,p))
            r = np.linalg.norm(x-x_old,2)
            if i == self.max_iter_newton:
                return x+float('nan')
        
        return x

    def sdirk_step(self,dt,t,x,p):
            alpha = 1-np.sqrt(2)/2
            alpha_hat = 2-5/4*np.sqrt(2)-alpha
            A = np.array([[alpha,0],[1-alpha,alpha]])
            c = np.array([alpha,1])
            b = np.array([[1-alpha,alpha]])
            b_hat = np.array([[1-alpha_hat,alpha_hat]])

            k = np.zeros([len(b[0]),len(x)])
            
            for i in range(2):
                Ai = np.array([A[i,:]])
                Yi = x + dt*(Ai@k)[0,:]
                k[i,:] = self.Newton(Yi,dt,t,np.zeros(len(x)),p(t+c[i]*dt),alpha)

                if k[i][0] != k[i][0]:
                    return np.zeros(len(x)),float('nan')

            return x+dt*(b@k)[0,:],dt*(b_hat@k)[0,:]
                




    def solve(self,t_0,t_end,x0,p,method = "euler",Tol= 0,dt = 0):
        
        x_sol = [x0]
        x = x0
        t = [t_0]
        if method=="euler":
            while t_0 < t_end:

                x = x + dt*self.f(x,t_0,p(t_0))
                x_sol.append(x)
                t_0 += dt
                t.append(t_0)
        else: 
            
            
            while t_0 < t_end:
                
                x,e_k = self.sdirk_step(dt,t_0,x,p)
                t_0 += dt
                x_sol.append(x)


                


        return x_sol, t


new_spring = spring()
dt = 1/200
dt = 1/1000
p = lambda t: 1e5 + 100*np.sin(10*np.pi*t)
x0 = np.array([1.0,0.0])

x_sols, _ = new_spring.solve(0,10,x0,p,method="sdirk",dt=dt)
x_sole, _ = new_spring.solve(0,10,x0,p,method="euler",dt=dt)

print(np.linalg.norm(np.array(x_sols)-np.array(x_sole),2))

plt.figure()
plt.plot(x_sols)
plt.savefig("sdirk")

plt.figure()
plt.plot(x_sole)
plt.savefig("euler")
