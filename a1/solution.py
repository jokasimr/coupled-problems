import warnings
import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve
from numba import jit, vectorize, guvectorize


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


@guvectorize('void(int64[:], int64, int64[:], int64[:])',
             '(),(),(n)->(n)',
             target='parallel', nopython=True)
def _neighboors_interior(i, N, _, n):
    # N = dofs per row
    elems_per_row = (N - 1) * 2
    p = i[0] // elems_per_row
    k = i[0] // 2

    if i[0] % 2 == 0:
        n[0] = k + p
        n[1] = k + 1 + p
        n[2] = k + N + p

    else:
        n[0] = k + N + p + 1
        n[1] = k + N + p
        n[2] = k + p + 1


def neighboors_interior(i, N):
    return _neighboors_interior(i, N, np.ones((1, 3), np.int64))


@guvectorize('void(int64[:], int64, int64[:], int64[:])',
             '(),(),(n)->(n)',
             target='parallel', nopython=True)
def _neighboors_exterior(i, N, _, n):
    # N = dofs per row
    i = i[0]
    elems_per_row = (N - 1)
    border = i // elems_per_row

    if border == 0:
        n[0] = i
        n[1] = i + 1
    elif border == 1:
        n[0] = N * (1 + i - elems_per_row) - 1
        n[1] = n[0] + N
    elif border == 2:
        n[0] = (N**2 - 1) - (i - 2 * elems_per_row)
        n[1] = n[0] - 1
    else:
        n[0] = N * (N - (i - 3 * elems_per_row) - 1)
        n[1] = n[0] - N


def neighboors_exterior(i, N):
    return _neighboors_exterior(i, N, np.ones((1, 2), np.int64))


def coords(indexes, dx, dy):
    Nx = round(1 / dx) + 1
    Ny = round(1 / dy) + 1
    x = (indexes % Nx) / (Nx - 1)
    y = (indexes // Ny) / (Ny - 1)
    return x, y


def assemble(m, neighboors):
    _, k = neighboors.shape
    i = np.stack([
        np.repeat(neighboors, k, axis=1).flatten(),
        np.tile(neighboors, k).flatten()
    ])
    m = np.tile(m.reshape(-1), len(neighboors))
    return sparse.coo_matrix((m, i)).tocsr()


def assemble_mass(dx, dy):
    m = dx * dy * np.array([
        [1/12, 1/24, 1/24],
        [1/24, 1/12, 1/24],
        [1/24, 1/24, 1/12],
    ])

    dofs_x = round(1 / dx) + 1
    dofs_y = round(1 / dy) + 1
    ndofs = dofs_x * dofs_y

    nelems = 2 * (dofs_x - 1) * (dofs_y - 1)
    elems = np.arange(nelems)

    return assemble(m, neighboors_interior(elems, dofs_x))


def assemble_boundary_mass(elems, d):
    # TODO: must depend on dy to handle nonunuiform grid
    m = d * np.array([
        [1/3, 1/6],
        [1/6, 1/3],
    ])

    dofs = round(1 / d) + 1

    return assemble(m, neighboors_exterior(elems, dofs))



def assemble_stiffness(dx, dy):
    a = - np.array([
        [(dx**2 + dy**2) / (2 * dx * dy), - dy / (2 * dx), - dx / (2 * dy)],
        [                - dy / (2 * dx),   dy / (2 * dx),             0.0],
        [                - dx / (2 * dy),             0.0,   dx / (2 * dy)],
    ])

    dofs_x = round(1 / dx) + 1
    dofs_y = round(1 / dy) + 1
    ndofs = dofs_x * dofs_y

    nelems = 2 * (dofs_x - 1) * (dofs_y - 1)
    elems = np.arange(nelems)

    return assemble(a, neighboors_interior(elems, dofs_x))


class LaplaceOnUnitSquare:
    def __init__(self, dx, f):
        self.dx = dx
        self.f = f
        self.N = round(1 / dx) + 1

        self.A = assemble_stiffness(dx, dx)
        self.M = assemble_mass(dx, dx)
        self.Mbx = assemble_boundary_mass(
            np.concatenate([
                np.arange(self.N - 1),
                np.arange(2 * (self.N - 1), 3 * (self.N - 1))]),
            dx)
        self.Mby = assemble_boundary_mass(
            np.concatenate([
                np.arange(self.N - 1, 2 * (self.N - 1)),
                np.arange(3 * (self.N - 1), 4 * (self.N - 1))]),
            dx)

        self.ndofs = self.A.shape[0]
        self.dofs = np.arange(self.ndofs, dtype=np.int64)
        self.reset()

    def reset(self, f=None):
        if f is None:
            f = self.f
        self.sol = np.zeros(self.ndofs)
        self.boundary_set = set()

        x, y = coords(self.dofs, self.dx, self.dx)
        self.rhs = self.M @ f(x, y)

    def boundary(self, e):
        N = round(1 / self.dx) + 1
        if e == 0:
            return np.arange(0, N)
        if e == 1:
            return np.arange(0, N) * N + (N - 1)
        if e == 2:
            return np.arange(N - 1, -1, -1) + N * (N - 1)
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
        M = self.Mbx if e in (0, 2) else self.Mby
        self.rhs += M[:, boundary] @ fn(x, y)

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

    def plot(self, k=100, show=True, **kwargs):
        x, y = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
        z = self.evaluate(x, y)
        plt.imshow(z, interpolation=None, origin='lower', **kwargs)
        plt.colorbar()
        if show:
            plt.show()

    def heat_flux(self, x, y, n):
        xc, yc = coords(self.dofs, self.dx, self.dx)
        xx = x[..., np.newaxis] - xc.reshape(*(1 for _ in x.shape), -1)
        yy = y[..., np.newaxis] - yc.reshape(*(1 for _ in y.shape), -1)

        if np.any(xx == 0) or np.any(yy == 0):
            warnings.warn('Heat flux is undefined on grid skeleton')

        # somewhat inefficient since we evaluate all basis functions in every point
        return - np.sum(self.sol * _ghatv(xx, yy, *n, self.dx, self.dx), axis=-1)


def f(x, y):
    pi = np.pi
    return (
        np.sin(pi * y**2) * (    pi * np.cos(pi * x**2) -     pi**2 * x**2 * np.sin(pi * x**2)) +
        np.sin(pi * x**2) * (2 * pi * np.cos(pi * y**2) - 4 * pi**2 * y**2 * np.sin(pi * y**2))
    )
