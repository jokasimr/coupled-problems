from pprint import pprint
from sympy import symbols, integrate, Matrix, init_printing, diff


init_printing()

x, y, dx, dy = symbols(('x', 'y', 'dx', 'dy'))

phis = [1 - x / dx - y / dy, x / dx, y / dy]

M = Matrix([[
    integrate(
        integrate(
            p1 * p2,
            (y, 0, dy * (1 - x / dx))
        ),
        (x, 0, dx)
    )
    for p2 in phis] for p1 in phis])

print('M =')
pprint(M)


def grad_dot(a, b):
    return diff(a, x) * diff(b, x) + diff(a, y) * diff(b, y)


A = Matrix([[
    integrate(
        integrate(
            grad_dot(p1, p2),
            (y, 0, dy * (1 - x / dx))
        ),
        (x, 0, dx)
    )
    for p2 in phis] for p1 in phis])

print('A =')
pprint(A)


Mb = Matrix([[
    integrate(
        p1 * p2,
        (x, 0, dx)
    )
    for p2 in (x / dx, 1 - x / dx)] for p1 in (x / dx, 1 - x / dx)])

print('Mb =')
pprint(Mb)


