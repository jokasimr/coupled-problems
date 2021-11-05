import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numba import jit, vectorize


@vectorize(["float64(float64, float64, float64, float64)"], nopython=True, target='parallel')
def _hatv(_x, _y, dx, dy):
    ' Computes value of canonical hat function at point x, y'
    x, y = _x / dx, _y / dy

    if not -1.0 <= (x + y) <= 1.0: return 0.0
    if not -1.0 <= x <= 1.0: return 0.0
    if not -1.0 <= y <= 1.0: return 0.0

    if x >= 0.0:
        if y >= 0.0: return 1.0 - y - x
        if x >= -y: return 1.0 - x
        else: return 1.0 + y
    else:
        if y <= 0.0: return 1.0 + y + x
        if x <= -y: return 1.0 + x
        else: return 1.0 - y


@vectorize(["float64(float64, float64, float64, float64, float64, float64)"], nopython=True, target='parallel')
def _ghatv(_x, _y, nx, ny, dx, dy):
    ' Computes gradient \dot (nx, ny) of a canonical hat function at point x, y'
    x, y = _x / dx, _y / dy

    if not -1.0 <= (x + y) <= 1.0: return 0.0
    if not -1.0 <= x <= 1.0: return 0.0
    if not -1.0 <= y <= 1.0: return 0.0

    if x >= 0.0:
        if y >= 0.0: return - ny / dy - nx / dx
        if x >= -y: return - nx / dx
        else: return ny / dy
    else:
        if y <= 0.0: return ny / dy + nx / dx
        if x <= -y: return nx / dx
        else: return - ny / dy


def coords(indexes, dx, dy):
    Nx = round(1 / dx) + 1
    Ny = round(1 / dy) + 1
    x = (indexes % Nx) / (Nx - 1)
    y = (indexes // Ny) / (Ny - 1)
    return x, y


@jit(nopython=True, fastmath=True)
def neighboors(i, dofs_per_row):
    N = dofs_per_row
    elems_per_row = (N - 1) * 2
    p = i // elems_per_row
    k = i // 2
    if i % 2 == 0:
        return (k + p, k + 1 + p, k + N + p)
    return (k + N + p + 1, k + N + p, k + p + 1)


def assemble_operator(dx, dy):

    Nx = 2 * round(1 / dx)
    Ny = round(1 / dy)
    dofs_x = round(1 / dx) + 1
    dofs_y = round(1 / dy) + 1

    M = dofs_x * dofs_y
    A = sparse.csr_matrix((M, M))

    V = (dx * dy) / 2

    xd = 1 / dx**2 * V
    yd = 1 / dy**2 * V

    As = np.array([
            [xd + yd, -xd, -yd],
            [    -xd,  xd, 0.0],
            [    -yd, 0.0,  yd]
    ])

    for i in range(Ny * Nx):

        n = neighboors(i, dofs_x)
        A[np.ix_(n, n)] += As

    return A


def assemble_mass(dx, dy):

    Nx = 2 * round(1 / dx)
    Ny = round(1 / dy)
    dofs_x = round(1 / dx) + 1
    dofs_y = round(1 / dy) + 1

    M = dofs_x * dofs_y
    Ma = sparse.csr_matrix((M, M))

    V = (dx * dy) / 2

    xd = 1 / dx**2 * V
    yd = 1 / dy**2 * V

    Ms = np.array([
            [xd + yd, -xd, -yd],
            [    -xd,  xd, 0.0],
            [    -yd, 0.0,  yd]
    ])

    for i in range(Ny * Nx):

        n = neighboors(i, dofs_x)
        A[np.ix_(n, n)] += As

    return A



class LaplaceOnUnitSquare:
    def __init__(self, dx, rhs):
        self.dx = dx
        self.N = round(1 / dx) + 1

        self.A = assemble_operator(dx, dx)
        self.M = self.A.shape[0]
        self.sol = np.zeros(self.M)
        self.boundary_set = set()

        self.dofs = np.arange(self.M, dtype=np.int64)
        x, y = coords(self.dofs, dx, dx)
        self.rhs = rhs(x, y)

    def boundary(self, e):
        N = round(1 / self.dx) + 1
        if e == 0:
            return np.arange(1, N - 1)
        if e == 1:
            return np.arange(0, N) * N + (N - 1)
        if e == 2:
            return np.arange(N - 2, 0, -1) + N * (N - 1)
        if e == 3:
            return np.arange(N - 1, -1, -1) * N
        raise ValueError(f'boundary index must be in (0, 1, 2, 3)')

    def set_dirchlet(self, e, fd):
        boundary = self.boundary(e)
        self.boundary_set = self.boundary_set.union(boundary)
        x, y = coords(boundary, self.dx, self.dx)
        self.sol[boundary] = fd(x, y)
        self.rhs -= self.A[:, boundary] @ self.sol[boundary]

    def set_neumann(self, e, fn):
        boundary = self.boundary(e)
        x, y = coords(boundary, self.dx, self.dx)
        self.rhs[boundary] += fn(x, y) * self.dx
        
    def solve(self):
        active_dofs = [i for i in self.dofs if i not in self.boundary_set]
        self.sol[active_dofs] = spsolve(
                self.A[active_dofs, :][:, active_dofs],
                self.rhs[active_dofs]
        )
        return self.sol

    def evaluate(self, x, y):
        xc, yc = coords(self.dofs, self.dx, self.dx)
        xx = x[..., np.newaxis] - xc.reshape(*(1 for _ in x.shape), -1)
        yy = y[..., np.newaxis] - yc.reshape(*(1 for _ in y.shape), -1)

        # somewhat inefficient since we evaluate all basis functions in every point
        return np.sum(self.sol * _hatv(xx, yy, self.dx, self.dx), axis=-1)

    def plot(self, k=50):
        x, y = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
        z = self.evaluate(x, y)
        plt.imshow(z, interpolation=None, origin='lower')
        plt.colorbar()
        plt.show()

    def heat_flux(self, x, y, n):
        ' Warning, heat flux undefined on grid skeleton '
        xc, yc = coords(self.dofs, self.dx, self.dx)
        xx = x[..., np.newaxis] - xc.reshape(*(1 for _ in x.shape), -1)
        yy = y[..., np.newaxis] - yc.reshape(*(1 for _ in y.shape), -1)

        # somewhat inefficient since we evaluate all basis functions in every point
        return np.sum(self.sol * _ghatv(xx, yy, *n, self.dx, self.dx), axis=-1)


def f(x, y):
    pi = np.pi
    return (
        np.sin(pi * y**2) * (    pi * np.cos(pi * x**2) -     pi**2 * x**2 * np.sin(pi * x**2)) +
        np.sin(pi * x**2) * (2 * pi * np.cos(pi * y**2) - 4 * pi**2 * y**2 * np.sin(pi * y**2))
    )

