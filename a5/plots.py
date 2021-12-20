import matplotlib.pyplot as plt
import numpy as np
from a5_2 import SpringDiscretization, DIRK2


def timestep_loop(u0, t0, t, stepper, dt=None):
    t = t0; u = u0
    while t < 1 - (dt if dt else 0):
        u, t = stepper(u, t, dt=dt)
    u, t = stepper(u, t, 1 - t)
    return u, t

def convergence_plot():
    s = SpringDiscretization(
        m=100, A=0.1, p0=1e5, k=1000,
        p=lambda t: 1e5 + 100*np.sin(10*np.pi*t))

    stepper = DIRK2(s, rtol=1e-7)

    solutions = []
    timesteps = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4)

    for dt in timesteps:
        t = 0; u = [0.0, 0.0]
        u, t = timestep_loop(u, t, 1, stepper, dt)
        solutions.append(u)

    errors = [np.linalg.norm(sol - solutions[-1])
              for sol in solutions[:-1]]

    plt.plot(timesteps[:-1], errors, '--o', label='DIRK2')
    plt.plot(timesteps[:-1], np.array(timesteps[:-1])**2, label='Reference line slope 2')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Convergence plot')
    plt.xlabel('$\Delta t$')
    plt.ylabel('$||e_n||_2$')
    plt.legend()
    plt.savefig('error.png')
    plt.show()


def solution_plot():
    p = lambda t: 1e5 + 100 * np.sin(10 * np.pi * t)

    s = SpringDiscretization(
        m=100, A=0.1, p0=1e5, k=100,
        p=lambda t: 1e5 + 100*np.sin(10*np.pi*t))

    stepper = DIRK2(s, rtol=1e-1)

    ts = [0]; us=[[0.0, 0.0]]
    while ts[-1] < 10:
        u, t = stepper(us[-1], ts[-1])
        us.append(u); ts.append(t)

    plt.plot(ts, np.stack(us)[:, 0], '-.', label='Displacement')
    plt.plot(ts, np.stack(us)[:, 1], label='Velocity')
    plt.title('Solution, $k=100$')
    plt.xlabel('time')
    plt.legend()
    plt.savefig('solutionk100.png')
    plt.show()
