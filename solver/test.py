import numpy as np
from .laplace import LaplaceOnRectangle


def test_convergence():
    def f(x, y):
        pi = np.pi
        return np.sin(pi * y ** 2) * (
            2 * pi * np.cos(pi * x ** 2) - 4 * pi ** 2 * x ** 2 * np.sin(pi * x ** 2)
        ) + np.sin(pi * x ** 2) * (
            2 * pi * np.cos(pi * y ** 2) - 4 * pi ** 2 * y ** 2 * np.sin(pi * y ** 2)
        )

    def u(x, y):
        pi = np.pi
        return -np.sin(pi * x ** 2) * np.sin(pi * y ** 2)

    hs = [1 / 5, 1 / 10, 1 / 20, 1 / 50]
    errors = []

    for h in hs:
        L = LaplaceOnRectangle(h, 1, 1, f)

        for e in range(4):
            L.set_dirchlet(e, lambda x, y: x * 0)

        L.solve()

        x, y = L.geometry.coords(L.geometry.dofs)

        errors.append(np.sqrt(h ** 2 * np.sum((L.sol - u(x, y)) ** 2)))

    p = np.polyfit(np.log(hs), np.log(errors), 1)
    assert 1.9 < p[0] < 2.0, "Convergence rate is not 2"


def test_zero_boundary_unit_square_no_rhs():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x)

    for i in range(4):
        L.set_dirchlet(i, lambda x, y: 0 * x)

    L.solve()
    assert np.all(L.sol == 0), "Solution should be all 0"


def test_nonzero_boundary_unit_square_no_rhs_x():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x)

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_nonzero_boundary_unit_square_no_rhs_y():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x)

    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_dirchlet(2, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, y), "Solution should be `y`"


def test_neumann_boundary_x():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x)

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_neumann(1, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_neumann_boundary_y():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x)

    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_neumann(2, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, y), "Solution should be `y`"


def test_nonzero_boundary_rectangle_no_rhs_x():
    L = LaplaceOnRectangle(0.1, 2, 1, lambda x, y: 0 * x)

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x + 2)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_nonzero_boundary_rectangle_no_rhs_y():
    L = LaplaceOnRectangle(0.1, 1, 2, lambda x, y: 0 * x)

    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_dirchlet(2, lambda x, y: 0 * x + 2)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, y), "Solution should be `y`"


def test_constant_boundary_rectangle_no_rhs():
    L = LaplaceOnRectangle(0.1, 1, 2, lambda x, y: 0 * x)

    L.set_dirchlet(0, lambda x, y: 0 * x + 15)
    L.set_dirchlet(1, lambda x, y: 0 * x + 15)
    L.set_dirchlet(2, lambda x, y: 0 * x + 15)
    L.set_dirchlet(3, lambda x, y: 0 * x + 15)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, 15), "Solution should be `15`"


def test_rectangular_neumann_boundary_x():
    L = LaplaceOnRectangle(0.1, 2, 1, lambda x, y: 0 * x)

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_neumann(1, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_rectangular_neumann_boundary_y():
    L = LaplaceOnRectangle(0.1, 2, 1, lambda x, y: 0 * x)

    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_neumann(2, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, y), "Solution should be `y`"


def test_diagonal_neumann():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x)

    L.set_dirchlet(0, lambda x, y: x)
    L.set_dirchlet(3, lambda x, y: y)
    L.set_neumann(1, lambda x, y: 0 * x + 1)
    L.set_neumann(2, lambda x, y: 0 * x + 1)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x + y), "Solution should be `x + y`"


def test_constant_rhs_x():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x + 1)

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(
        L.sol, x * (1 - x) / 2
    ), "Solution should be upside down parabola"


def test_adjustable_heat_conductivity():
    L = LaplaceOnRectangle(0.1, 1, 1, lambda x, y: 0 * x + 1, heat_conductivity=0.1)

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x)

    L.solve()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(
        L.sol, 1 / 0.1 * x * (1 - x) / 2
    ), "Solution should be upside down parabola, scaled by 1/heat_conductivity"
