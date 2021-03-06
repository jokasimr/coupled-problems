import numpy as np
from .laplace import LaplaceOnRectangle, HeatEqOnRectangle


def convergence():
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
        dt = 0.1
        u_0 = np.zeros([(int(1/h)+1)*(int(1/h)+1)])
        L = HeatEqOnRectangle(u_0,dt,h, 1, 1, f)

        for e in range(4):
            L.set_dirchlet(e, lambda x, y: x * 0)

        for i in range(500):
            L.do_euler_step()
            L.update_u_old()

        x, y = L.geometry.coords(L.geometry.dofs)

        errors.append(np.sqrt(h ** 2 * np.sum((L.sol - u(x, y)) ** 2)))

    p = np.polyfit(np.log(hs), np.log(errors), 1)
    assert 1.9 < p[0] < 2.0, "Convergence rate is not 2"


def test_zero_boundary_unit_square_no_rhs():
    dx = 0.2
    dt = 0.1
    L = HeatEqOnRectangle(
        dt, dx, 1, 1, lambda x, y: 0 * x,
        lambda x, y: 1.0 * ((x * (1 - x) > 0.01) & (y * (1 - y) > 0.01)))

    for i in range(4):
        L.set_dirchlet(i, lambda x, y: 0 * x)

    for i in range(500):
        L.do_euler_step()
        L.update_u_old()

    assert np.allclose(L.sol, 0), "Solution should be all 0"


def test_nonzero_boundary_unit_square_no_rhs_x():
    dx = 0.1
    dt = 0.2
    L = HeatEqOnRectangle(
        dt, dx, 1, 1, lambda x, y: 0 * x,
        lambda x, y: 1.0 * (x > 0.01))

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x + 1)

    for i in range(500):
        L.do_euler_step()
        L.update_u_old()

    x, y = L.geometry.coords(L.geometry.dofs)
    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_nonzero_boundary_unit_square_no_rhs_y():
    dx = 0.1
    dt = 100
    L = HeatEqOnRectangle(
        dt,dx, 1, 1, lambda x, y: 0 * x,
        lambda x, y: 1.0 * (y > 0.01),
        heat_capacitance = 10, heat_conductivity = 10)

    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_dirchlet(2, lambda x, y: 0 * x + 1)

    for i in range(500):
        L.do_euler_step()
        L.update_u_old()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, y, rtol=1e-2), "Solution should be `y`"


def test_neumann_boundary_x():
    dx = 0.1
    dt = 0.1
    L = HeatEqOnRectangle(
        dt, dx, 1, 1, lambda x, y: 0 * x,
        lambda x, y: 1.0 * (x > 0.01))

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_neumann(1, lambda x, y: 0 * x + 1)

    for i in range(500):
        L.do_euler_step()
        L.update_u_old()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_nonzero_boundary_rectangle_no_rhs_x():
    dx = 0.2
    dt = 0.1
    L = HeatEqOnRectangle(
        dt, dx, 2, 1, lambda x, y: 0 * x,
        lambda x, y: 2.0 * (x  > 0.01))

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x + 2)

    for i in range(500):
        L.do_euler_step()
        L.update_u_old()

    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, x), "Solution should be `x`"


def test_nonzero_boundary_rectangle_no_rhs_y():
    dx = 0.2
    dt = 0.1
    L = HeatEqOnRectangle(
        dt, dx, 1, 2, lambda x, y: 0 * x,
        lambda x, y: 2.0 * (y > 0.01))

    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_dirchlet(2, lambda x, y: 0 * x + 2)

    for i in range(500):
        L.do_euler_step()
        L.update_u_old()


    x, y = L.geometry.coords(L.geometry.dofs)

    assert np.allclose(L.sol, y), "Solution should be `y`"


def nonzero_boundary_rectangle_no_rhs_x():
    dx = 0.05
    dt = 0.01

    L = HeatEqOnRectangle(
         dt, dx, 1, 1, lambda x, y: 0 * x,
         lambda x, y: np.sin(x * np.pi) * np.sin(y * np.pi)
    )

    L.set_dirchlet(3, lambda x, y: 0 * x)
    L.set_dirchlet(2, lambda x, y: 0 * x)
    L.set_dirchlet(0, lambda x, y: 0 * x)
    L.set_dirchlet(1, lambda x, y: 0 * x)

    for i in range(1000):
        L.do_euler_step()
        L.update_u_old()
    
    def u(x, y):
        pi = np.pi
        return 100*np.exp(-(np.pi)**2 - (np.pi)**2)*np.sin(pi * x) * np.sin(pi * y)
    
    x, y = L.geometry.coords(L.geometry.dofs)
    
    assert np.allclose(L.sol,u(x,y)), "Solution should be `decay with e^(-kt)`"
