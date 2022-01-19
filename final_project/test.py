import pytest
import numpy as np
from sdirk import DIRK2, Helmholtz, LinearHelmholtz


def convergence_coefficient(stepper_cls, problem, solution):
    stepper = stepper_cls(problem)
    end_values = []
    ks = np.array([50, 75, 100, 150, 200])
    for k in ks:
        u = problem.u0
        t = 0
        dt = 1 / k
        for i in range(k):
            u, t = stepper(u, t, dt)
        end_values.append(u)

    errors = np.linalg.norm(np.array(end_values) - np.array(solution), axis=-1)
    p = np.polyfit(np.log(1/ks), np.log(errors), 1)[0]

    return p


def linear_test_equation(A, u0):
    h = LinearHelmholtz(lambda t, u: np.array(A))
    h.u0 = np.array(u0)
    return h


def test_identity_DIRK2():
    ''' Convergence test for linear test equation du/dt = Iu '''

    p = convergence_coefficient(DIRK2,
        linear_test_equation(
            [[1, 0], [0, 1]],
            [0, 1],
        ),
        [0, np.e]
    )
    assert 1.9 < p < 2, 'Convergence rate is not approximately 2'


def test_stiff_DIRK2():
    p = convergence_coefficient(DIRK2,
        linear_test_equation(
            [[10, 0], [0, 1]],
            [1, 1],
        ),
        [np.e**10, np.e]
    )
    assert 1.9 < p < 2, 'Convergence rate is not approximately 2'


def test_nonlin_DIRK2():
    h = Helmholtz(lambda t, u: u * (1 - u))
    u0 = 0.1
    h.u0 = np.array([u0])
    p = convergence_coefficient(
        DIRK2,
        h, [u0 * np.e/(1 - u0 + u0 * np.e)]
    )
    assert 1.9 < p < 2, 'Convergence rate is not approximately 2'



def solve_to_time(problem, rtol, tend=1.0):

    stepper = DIRK2(problem, rtol)
    u = problem.u0
    t = 0

    while t + stepper.dt < tend:
        u, t = stepper(u, t)

    # last step
    u, t = stepper(u, t, tend - t)

    return u


@pytest.mark.parametrize("rtol", [1e-2, 1e-5, 1e-8])
@pytest.mark.parametrize(
    "A,u0,sol",
    [
        ([[1, 0], [0, 1]],
         [0, 1],
         [0, np.e]),
    ]
)
def test_adaptivity_DIRK2(A, u0, sol, rtol):
    u = solve_to_time(
        linear_test_equation(A, u0),
        rtol
    )
    rel_err = np.linalg.norm(u - sol) / np.linalg.norm(u)
    assert rtol / 10 <= rel_err < rtol
