import numpy as np

from numpy import zeros, array, diag, tril, concatenate, ones
from numpy.linalg import inv


class SpringDiscretization:
    def __init__(self, k, m, A, p0, p):
        self.m = m
        self.k = k
        self.A = A
        self.p0 = p0
        self.p = p

        self.K = np.array([[0, 1], [-k/m, 0]])
        
    def rhs(self, ubar, t):
        return self.K @ ubar + np.array([0, self.A * (self.p(t) - self.p0)]) / self.m

    def solve(self, ubar, t, alpha):
        S = np.eye(2) - alpha * self.K
        return np.linalg.solve(S, self.rhs(np.array(ubar), t))


class DIRK:
    def __init__(self, helmholtz, rtol=1e-3):
        self.helmholtz = helmholtz
        self.rtol = rtol
        self.atol = 1e-5

        # method coefficients
        self.A = [array(a) for a in self.A]
        self.b = array(self.b)
        self.c = array(self.c)

        if hasattr(self, 'd'):
            self.d = array(self.d)

        self.order = len(self.A)

        self.dt = 1e-3 * rtol


    def __call__(self, u, t, dt=None):

        if dt:
            self.dt = dt

        stages = zeros((len(self.A), len(u)))

        for i, (a, c) in enumerate(zip(self.A, self.c)):

            alpha = a[-1] * self.dt
            tbar = t + c * self.dt
            ubar = u + self.dt * a[:-1] @ stages[:i]

            stages[i, :] = self.helmholtz.solve(ubar, tbar, alpha)

        t += self.dt
        sol = u + self.dt * self.b @ stages

        if hasattr(self, 'd'):

            def rms_norm(y, tol):
                return np.sqrt(np.mean((y / tol)**2))

            tol = self.rtol * np.linalg.norm(u) + self.atol
            err = self.dt * self.d @ stages
            self.dt = self.dt * 0.9 * (tol / rms_norm(err, tol))**(1 / self.order)

        return sol, t


class DIRK3(DIRK):
    A = [[1/2],
         [1/6, 1/2],
         [-1/2, 1/2, 1/2],
         [3/2, -3/2, 1/2, 1/2]]
    b = [3/2, -3/2, 1/2, 1/2]
    c = [1/2, 2/3, 1/2, 1]
    d = [1]

class ImplEuler(DIRK):
    A = [[1]]
    b = [1]
    c = [1]

class DIRK2(DIRK):
    alpha = 1 - 2**0.5 / 2
    alpha_hat = 2 - 5/4 * 2**0.5
    A = [[alpha],
         [1 - alpha, alpha]]
    b = [1 - alpha, alpha]
    c = [alpha, 1]
    d = [alpha_hat - alpha, alpha - alpha_hat]
