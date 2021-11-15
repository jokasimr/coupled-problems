import warnings
import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve
from numba import jit, vectorize, guvectorize


@vectorize(
    ["float64(float64, float64, float64, float64)"], nopython=True, target="parallel"
)
def _hatv(_x, _y, dx, dy):
    " Computes value of canonical hat function at point x, y"
    x, y = _x / dx, _y / dy

    if not -1.0 <= (x + y) <= 1.0:
        return 0.0
    if not -1.0 <= x <= 1.0:
        return 0.0
    if not -1.0 <= y <= 1.0:
        return 0.0

    if x >= 0.0:
        if y >= 0.0:
            return 1.0 - y - x
        if x >= -y:
            return 1.0 - x
        else:
            return 1.0 + y
    else:
        if y <= 0.0:
            return 1.0 + y + x
        if x <= -y:
            return 1.0 + x
        else:
            return 1.0 - y


@vectorize(
    ["float64(float64, float64, float64, float64, float64, float64)"],
    nopython=True,
    target="parallel",
)
def _ghatv(_x, _y, nx, ny, dx, dy):
    " Computes gradient \dot (nx, ny) of a canonical hat function at point x, y"
    x, y = _x / dx, _y / dy

    if not -1.0 <= (x + y) <= 1.0:
        return 0.0
    if not -1.0 <= x <= 1.0:
        return 0.0
    if not -1.0 <= y <= 1.0:
        return 0.0

    if x >= 0.0:
        if y >= 0.0:
            return -ny / dy - nx / dx
        if x >= -y:
            return -nx / dx
        else:
            return ny / dy
    else:
        if y <= 0.0:
            return ny / dy + nx / dx
        if x <= -y:
            return nx / dx
        else:
            return -ny / dy


@guvectorize(
    "void(int64[:], int64, int64[:], int64[:])",
    "(),(),(n)->(n)",
    target="parallel",
    nopython=True,
)
def _neighboors_interior(_element_index, dofs_per_row, _, n):
    element_index = _element_index[0]
    elems_per_row = 2 * (dofs_per_row - 1)
    row = element_index // elems_per_row
    dof_index_if_all_elements_on_one_row = element_index // 2

    if element_index % 2 == 0:
        n[0] = dof_index_if_all_elements_on_one_row + row
        n[1] = dof_index_if_all_elements_on_one_row + row + 1
        n[2] = dof_index_if_all_elements_on_one_row + row + dofs_per_row

    else:
        n[0] = dof_index_if_all_elements_on_one_row + row + dofs_per_row + 1
        n[1] = dof_index_if_all_elements_on_one_row + row + dofs_per_row
        n[2] = dof_index_if_all_elements_on_one_row + row + 1


def neighboors_interior(*args):
    return _neighboors_interior(*args, np.ones((1, 3), np.int64))


@guvectorize(
    "void(int64[:], int64, int64, int64[:], int64[:])",
    "(),(),(),(n)->(n)",
    target="parallel",
    nopython=True,
)
def _neighboors_exterior(_element_index, dofs_per_row, dofs_per_col, _, n):
    # N = dofs per row
    element_index = _element_index[0]
    elems_per_row = dofs_per_row - 1
    elems_per_col = dofs_per_col - 1

    if 0 <= element_index < elems_per_row:
        # bottom
        n[0] = element_index
        n[1] = element_index + 1

    elif elems_per_row <= element_index < elems_per_row + elems_per_col:
        # right
        row = element_index - elems_per_row
        n[0] = (row + 1) * dofs_per_row - 1
        n[1] = (row + 2) * dofs_per_row - 1

    elif (
        elems_per_row + elems_per_col
        <= element_index
        < elems_per_row * 2 + elems_per_col
    ):
        # top
        dof_max_index = dofs_per_row * dofs_per_col - 1
        column = element_index - elems_per_row - elems_per_col
        n[0] = dof_max_index - column
        n[1] = dof_max_index - column - 1

    else:
        # left
        row = elems_per_col - (element_index - 2 * elems_per_row - elems_per_col) - 1
        n[0] = (row + 1) * dofs_per_row
        n[1] = row * dofs_per_row


def neighboors_exterior(*args):
    return _neighboors_exterior(*args, np.ones((1, 2), np.int64))


def coords(indexes, width, height, dx, dy):
    dofs_per_row = round(width / dx) + 1
    x = dx * (indexes % dofs_per_row)
    y = dy * (indexes // dofs_per_row)
    return x, y


def assemble(m, neighboors):
    _, k = neighboors.shape
    i = np.stack(
        [np.repeat(neighboors, k, axis=1).flatten(), np.tile(neighboors, k).flatten()]
    )
    m = np.tile(m.reshape(-1), len(neighboors))
    return sparse.coo_matrix((m, i)).tocsr()


def assemble_mass(dofs_per_row, dofs_per_col, dx, dy):
    m = dx * dy * np.array([
        [1 / 12, 1 / 24, 1 / 24],
        [1 / 24, 1 / 12, 1 / 24],
        [1 / 24, 1 / 24, 1 / 12]]) 

    nelems = 2 * (dofs_per_row - 1) * (dofs_per_col - 1)
    elems = np.arange(nelems)

    return assemble(m, neighboors_interior(elems, dofs_per_row))


def assemble_boundary_mass(elems, dofs_per_row, dofs_per_col, dx):
    # TODO: must depend on dy to handle nonunuiform grid
    m = dx * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])

    return assemble(m, neighboors_exterior(elems, dofs_per_row, dofs_per_col))


def assemble_stiffness(dofs_per_row, dofs_per_col, dx, dy):
    a = np.array([
        [(dx ** 2 + dy ** 2) / (2 * dx * dy), -dy / (2 * dx), -dx / (2 * dy)],
        [-dy / (2 * dx), dy / (2 * dx), 0.0],
        [-dx / (2 * dy), 0.0, dx / (2 * dy)]])

    nelems = 2 * (dofs_per_row - 1) * (dofs_per_col - 1)
    elems = np.arange(nelems)

    return assemble(a, neighboors_interior(elems, dofs_per_row))


class LaplaceOnUnitSquare:
    def __init__(self, dx, width, height, f):
        self.dx = dx
        self.f = f
        assert width / dx == round(width / dx), "`width` must be evenly divisible by `dx`"
        assert height / dx == round(height / dx), "`height` must be evenly divisible by `dx`"
        self.dofs_per_row = round(width / dx) + 1
        self.dofs_per_col = round(height / dx) + 1
        self.width = width
        self.height = height

        self.A = assemble_stiffness(self.dofs_per_row, self.dofs_per_col, dx, dx)
        self.M = assemble_mass(self.dofs_per_row, self.dofs_per_col, dx, dx)

        elems_per_row = self.dofs_per_row - 1
        elems_per_col = self.dofs_per_col - 1

        self.Mbx = assemble_boundary_mass(
            np.concatenate([
                np.arange(elems_per_row),
                np.arange(elems_per_row + elems_per_col, 2 * elems_per_row + elems_per_col)]),
            self.dofs_per_row,
            self.dofs_per_col,
            dx,
        )

        self.Mby = assemble_boundary_mass(
            np.concatenate([
                np.arange(elems_per_row, elems_per_row + elems_per_col),
                np.arange(
                    2 * elems_per_row + elems_per_col,
                    2 * (elems_per_row + elems_per_col))]),
            self.dofs_per_row,
            self.dofs_per_col,
            dx,
        )

        self.ndofs = self.A.shape[0]
        self.dofs = np.arange(self.ndofs, dtype=np.int64)
        self.reset()

    def reset(self, f=None):
        if f is None:
            f = self.f
        self.sol = np.zeros(self.ndofs)
        self.boundary_set = set()

        x, y = coords(self.dofs, self.width, self.height, self.dx, self.dx)
        self.rhs = self.M @ f(x, y)

    def boundary(self, e):
        if e == 0:
            return np.arange(self.dofs_per_row)
        if e == 1:
            return np.arange(self.dofs_per_col) * self.dofs_per_row + (self.dofs_per_row - 1)
        if e == 2:
            total_dofs = self.dofs_per_col * self.dofs_per_row - 1
            return total_dofs - np.arange(self.dofs_per_row)
        if e == 3:
            return np.arange(self.dofs_per_col - 1, -1, -1) * self.dofs_per_row
        raise ValueError(f"boundary index must be in (0, 1, 2, 3)")

    def set_dirchlet(self, e, fd):
        boundary = self.boundary(e)
        self.boundary_set = self.boundary_set.union(boundary)
        x, y = coords(boundary, self.width, self.height, self.dx, self.dx)
        self.sol[boundary] = fd(x, y)
        self.rhs -= self.A[:, boundary] @ self.sol[boundary]

    def set_neumann(self, e, fn, raw=False):
        boundary = self.boundary(e)
        if raw:
            self.rhs[boundary] += fn
        else:
            x, y = coords(boundary, self.width, self.height, self.dx, self.dx)
            M = self.Mbx if e in (0, 2) else self.Mby
            self.rhs += M[:, boundary] @ fn(x, y)

    def solve(self):
        active_dofs = [i for i in self.dofs if i not in self.boundary_set]
        self.sol[active_dofs] = spsolve(
            self.A[active_dofs, :][:, active_dofs], self.rhs[active_dofs]
        )
        return self.sol

    def evaluate(self, x, y):
        xc, yc = coords(self.dofs, self.width, self.height, self.dx, self.dx)
        xx = x[..., np.newaxis] - xc.reshape(*(1 for _ in x.shape), -1)
        yy = y[..., np.newaxis] - yc.reshape(*(1 for _ in y.shape), -1)

        # somewhat inefficient since we evaluate all basis functions in every point
        return np.sum(self.sol * _hatv(xx, yy, self.dx, self.dx), axis=-1)

    def plot(self, k=100, show=True, **kwargs):
        x, y = np.meshgrid(
            np.linspace(0, self.width, k), np.linspace(0, self.height, k)
        )
        z = self.evaluate(x, y)
        plt.imshow(
            z,
            interpolation=None,
            origin="lower",
            extent=[0, self.width, 0, self.height],
            **kwargs,
        )
        plt.colorbar()
        if show:
            plt.show()

    def heat_flux(self, x, y, n):
        xc, yc = coords(self.dofs, self.width, self.height, self.dx, self.dx)
        xx = x[..., np.newaxis] - xc.reshape(*(1 for _ in x.shape), -1)
        yy = y[..., np.newaxis] - yc.reshape(*(1 for _ in y.shape), -1)

        if np.any(xx == 0) or np.any(yy == 0):
            warnings.warn("Heat flux is undefined on grid skeleton")

        # somewhat inefficient since we evaluate all basis functions in every point
        return -np.sum(self.sol * _ghatv(xx, yy, *n, self.dx, self.dx), axis=-1)


def f(x, y):
    pi = np.pi
    return (
        np.sin(pi * y ** 2) * (pi * np.cos(pi * x ** 2) - pi ** 2 * x ** 2 * np.sin(pi * x ** 2))
      + np.sin(pi * x ** 2) * (2 * pi * np.cos(pi * y ** 2) - 4 * pi ** 2 * y ** 2 * np.sin(pi * y ** 2))
    )
