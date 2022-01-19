from numpy import (
    zeros, array, ones,
    diag, tril, concatenate,
    sqrt, mean
)

from numpy.random import random

from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov


class LinearHelmholtz:
    def __init__(self, G):
        self.G = G
    def solve(self, ubar, tbar, alpha):
        G = self.G(tbar, ubar)
        A = speye(len(G)) - alpha * G
        return spsolve(A, G @ ubar)

class Helmholtz:
    def __init__(self, L):
        self.L = L
    def solve(self, ubar, tbar, alpha):
        return newton_krylov(
            lambda k: k - self.L(tbar, ubar + alpha * k),
            zeros(len(ubar)))

class DIRK:
    def __init__(self, helmholtz, rtol=1e-6, atol=1e-10):
        self.helmholtz = helmholtz
        self.rtol = rtol
        self.atol = atol
        self.order = len(self.A)
        self.dt = max(atol, 1e-2 * rtol)

        self.A = list(map(array, self.A))
        self.b = array(self.b)
        self.c = array(self.c)
        if hasattr(self, 'd'):
            self.d = array(self.d)

        self.maxdt = float('inf')


    def __call__(self, u, t, dt=None):

        dt = dt if dt else self.dt
        stages = zeros((len(self.A), len(u)))

        for i, (a, c) in enumerate(zip(self.A, self.c)):
            alpha = a[i] * dt
            tbar = t + c * dt
            ubar = u + dt * a[:i] @ stages[:i]

            try: 
                stages[i, :] = self.helmholtz.solve(ubar, tbar, alpha)
            except KeyboardInterrupt as err:
                raise err
            except:
                self.maxdt = dt / 2
                return self(u, t, dt/4) 

        self.maxdt = max(self.maxdt, dt)

        t += dt
        sol = u + dt * self.b @ stages

        # Adaptivity
        if hasattr(self, 'd'):
            def rms_norm(y):
                return sqrt(mean(y**2))

            tol = self.rtol * rms_norm(u) + self.atol
            err = dt * self.d @ stages
            self.dt = .9 * dt * (tol / rms_norm(err))**(1 / self.order)
            self.dt = min(self.dt, self.maxdt)

        return sol, t


class DIRK2(DIRK):
    alpha = 1 - 2**0.5 / 2
    alpha_hat = 2 - 5/4 * 2**0.5
    A = [[alpha],
         [1 - alpha, alpha]]
    b = [1 - alpha, alpha]
    c = [alpha, 1]
    d = [alpha_hat - alpha, alpha - alpha_hat]
