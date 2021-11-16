import numpy as np
from laplace import LaplaceOnUnitSquare, coords


def convergence_test():

    def f(x, y):
        pi = np.pi
        return (
            np.sin(pi * y**2) * (2 * pi * np.cos(pi * x**2) - 4 * pi**2 * x**2 * np.sin(pi * x**2)) +
            np.sin(pi * x**2) * (2 * pi * np.cos(pi * y**2) - 4 * pi**2 * y**2 * np.sin(pi * y**2))
        )


    def u(x, y):
        pi = np.pi
        return np.sin(pi * x**2) * np.sin(pi * y**2)


    hs = [1/5, 1/10, 1/20, 1/50]
    errors = []

    for h in hs:
        L = LaplaceOnUnitSquare(h, [0, 1], [0, 1], f)

        for e in range(4):
            L.set_dirchlet(e, lambda x, y: x*0)

        L.solve()

        x, y = coords(L.dofs, L.dx, L.dx)

        errors.append(np.sqrt(h**2 * np.sum((L.sol - u(x, y))**2)))


    p = np.polyfit(np.log(hs), np.log(errors), 1)
    assert 1.9 < p[0] < 2.0, 'Convergence rate is not 2'


def test_zero_boundary_unit_square_no_rhs():
    L = LaplaceOnUnitSquare(0.1, [0, 1], [0, 1], lambda x, y: 0*x)

    for i in range(4):
        L.set_dirchlet(i, lambda x, y: 0*x)

    L.solve()
    assert np.all(L.sol == 0), 'Solution should be all 0'


def test_nonzero_boundary_unit_square_no_rhs_x():
    L = LaplaceOnUnitSquare(0.5, [0, 1], [0, 1], lambda x, y: 0*x)

    L.set_dirchlet(3, lambda x, y: 0*x)
    L.set_dirchlet(1, lambda x, y: 0*x + 1)

    L.solve()

    x, y = coords(L.dofs, L.dx, L.dx)

    assert np.allclose(L.sol, x), 'Solution should be `x`'


def test_nonzero_boundary_unit_square_no_rhs_y():
    L = LaplaceOnUnitSquare(0.1, [0, 1], [0, 1], lambda x, y: 0*x)

    L.set_dirchlet(0, lambda x, y: 0*x)
    L.set_dirchlet(2, lambda x, y: 0*x + 1)

    L.solve()

    x, y = coords(L.dofs, L.dx, L.dx)

    assert np.allclose(L.sol, y), 'Solution should be `y`'

def test_nonzero_boundary_unit_Rectangle_no_rhs_y():
    L = LaplaceOnUnitSquare(0.01, [0, 2], [0, 1], lambda x, y: 0*x)

    L.set_dirchlet(0, lambda x, y: 0*x)
    L.set_dirchlet(2, lambda x, y: 0*x + 1)

    L.solve()

    x, y = coords(L.dofs, L.dx, L.dx/2)
    y=y*2

    assert np.allclose(L.sol, y), 'Solution should be `y`'

def test_nonzero_boundary_unit_Rectangle_no_rhs_y():
    L = LaplaceOnUnitSquare(0.01, [0, 1], [0, 2], lambda x, y: 0*x)

    L.set_dirchlet(0, lambda x, y: 0*x)
    L.set_dirchlet(2, lambda x, y: 0*x + 1)

    L.solve()

    x, y = coords(L.dofs, L.dx, L.dx/2)
    y = y

    assert np.allclose(L.sol, y), 'Solution should be `y`'
