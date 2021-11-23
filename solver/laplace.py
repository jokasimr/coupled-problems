import warnings
import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve

from geometry import Rectangle
from basis_functions import _hatv, _ghatv


def assemble(m, neighboors):
    _, k = neighboors.shape
    i = np.stack(
        [np.repeat(neighboors, k, axis=1).flatten(), np.tile(neighboors, k).flatten()]
    )
    m = np.tile(m.reshape(-1), len(neighboors))
    return sparse.coo_matrix((m, i)).tocsr()


def assemble_mass(geometry):
    dx, dy = geometry.dx, geometry.dy
    m = dx * dy * np.array([
        [1 / 12, 1 / 24, 1 / 24],
        [1 / 24, 1 / 12, 1 / 24],
        [1 / 24, 1 / 24, 1 / 12]])

    return assemble(m, geometry.neighboors_interior(geometry.elems))


def assemble_boundary_mass_y(geometry):
    m = geometry.dy * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])

    return assemble(m,
            geometry.neighboors_exterior(
                np.concatenate((
                    geometry.boundary_elems(1),
                    geometry.boundary_elems(3)
                ))
            )
    )


def assemble_boundary_mass_x(geometry):
    m = geometry.dx * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])

    return assemble(m,
            geometry.neighboors_exterior(
                np.concatenate((
                    geometry.boundary_elems(0),
                    geometry.boundary_elems(2)
                ))
            )
    )


def assemble_stiffness(geometry):
    dx, dy = geometry.dx, geometry.dy
    a = np.array([
        [(dx ** 2 + dy ** 2) / (2 * dx * dy), -dy / (2 * dx), -dx / (2 * dy)],
        [-dy / (2 * dx), dy / (2 * dx), 0.0],
        [-dx / (2 * dy), 0.0, dx / (2 * dy)]])

    return assemble(a, geometry.neighboors_interior(geometry.elems))


class LaplaceOnRectangle:
    def __init__(self, dx, width, height, f, heat_conductivity=1.0):

        self.lamda = heat_conductivity
        self.f = f

        self.geometry = Rectangle(width, height, dx, dx)

        self.A = self.lamda * assemble_stiffness(self.geometry)
        self.M = assemble_mass(self.geometry)

        self.Mbx = assemble_boundary_mass_x(self.geometry)
        self.Mby = assemble_boundary_mass_y(self.geometry)

        self.reset()

    def reset(self, f=None):
        if f is None:
            f = self.f
        self.sol = np.zeros(self.geometry.ndofs)
        self.boundary_set = set()

        x, y = self.geometry.coords(self.geometry.dofs)
        self.rhs = self.M @ f(x, y)

    def set_dirchlet(self, e, fd, raw=False):
        if hasattr(e, '__iter__'):
            boundary = e
        else:
            boundary = self.geometry.boundary_dofs(e)

        self.boundary_set = self.boundary_set.union(boundary)

        if raw:
            self.sol[boundary] = fd
            self.rhs -= self.A[:, boundary] @ self.sol[boundary]

        else:
            x, y = self.geometry.coords(boundary)
            self.sol[boundary] = fd(x, y)
            self.rhs -= self.A[:, boundary] @ self.sol[boundary]

    def set_neumann(self, e, fn, raw=False):
        if hasattr(e, '__iter__'):
            boundary = e
        else:
            boundary = self.geometry.boundary_dofs(e)

        if raw:
            self.rhs[boundary] += fn

        else:
            x, y = self.geometry.coords(boundary)
            M = self.Mbx if e in (0, 2) else self.Mby
            self.rhs += self.lamda * M[:, boundary] @ fn(x, y)

    def solve(self):
        active_dofs = list(set(self.geometry.dofs) - self.boundary_set)
        self.sol[active_dofs] = spsolve(
            self.A[active_dofs, :][:, active_dofs], self.rhs[active_dofs]
        )
        return self.sol

    def evaluate(self, x, y):
        xc, yc = self.geometry.coords(self.geometry.dofs)
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

    def heat_flux_at_nodes(self, i):
        return self.A[i] @ self.sol

    def heat_flux(self, x, y, n):
        xc, yc = self.geometry.coords(self.geometry.dofs)
        xx = x[..., np.newaxis] - xc.reshape(*(1 for _ in x.shape), -1)
        yy = y[..., np.newaxis] - yc.reshape(*(1 for _ in y.shape), -1)

        if np.any(xx == 0) or np.any(yy == 0):
            warnings.warn("Heat flux is undefined on grid skeleton")

        # somewhat inefficient since we evaluate all basis functions in every point
        return -np.sum(self.sol * _ghatv(xx, yy, *n, self.dx, self.dx), axis=-1)


class HeatEqOnRectangle(LaplaceOnRectangle):

    def __init__(self, u_0, dt, dx, width, height, f, heat_conductivity=1.0, heat_capacitance=1.0):
        super().__init__(dx, width, height, f, heat_conductivity=heat_conductivity)
        self.dt = dt
        self.lamda = heat_conductivity
        self.u_old = u_0
        self.sol = u_0
        self.alpha = heat_capacitance
        self.A_hat = self.alpha * self.M + dt * self.A

    def set_dirchlet(self, e, fd, raw=False):
        if hasattr(e, '__iter__'):
            boundary = e
        else:
            boundary = self.geometry.boundary_dofs(e)

        self.boundary_set = self.boundary_set.union(boundary)

        if raw:
            self.sol[boundary] = fd
            self.rhs += self.alpha * self.M[:, boundary] @ self.u_old[boundary] - self.A_hat[:, boundary] @ self.sol[boundary]

        else:
            x, y = self.geometry.coords(boundary)
            self.sol[boundary] = fd(x, y)
            self.rhs += self.alpha * self.M[:, boundary] @ self.u_old[boundary] - self.A_hat[:, boundary] @ self.sol[boundary]

    def set_neumann(self, e, fn, raw=False):
        if hasattr(e, '__iter__'):
            boundary = e
        else:
            boundary = self.geometry.boundary_dofs(e)

        if raw:
            self.rhs[boundary] += fn

        else:
            x, y = self.geometry.coords(boundary)
            M = self.Mbx if e in (0, 2) else self.Mby
            self.rhs += self.dt * self.lamda * M[:, boundary] @ fn(x, y)

    def do_euler_step(self):
        active_dofs = list(set(self.geometry.dofs) - self.boundary_set)
        rhs = self.rhs[active_dofs] + self.M[active_dofs, :][:, active_dofs] @ self.u_old[active_dofs]
        
        self.sol[active_dofs] = spsolve(self.A_hat[active_dofs, :][:, active_dofs], rhs)

        return self.sol
    
    def heat_flux_at_nodes(self, i):
        return self.geometry.dx * (-self.A_hat[i] @ self.sol + self.alpha * self.M[i] @ self.u_old)

    def update_u_old(self):
        self.u_old = self.sol
