from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
import numpy as np
from solution import LaplaceOnUnitSquare, coords


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
    L = LaplaceOnUnitSquare(h, f)

    for e in range(4):
        L.set_dirchlet(e, lambda x, y: x*0)

    L.solve()

    x, y = coords(L.dofs, L.dx, L.dx)

    errors.append(np.sqrt(h**2 * np.sum((L.sol - u(x, y))**2)))


p = np.polyfit(np.log(hs), np.log(errors), 1)
x = np.linspace(hs[0], hs[-1])

plt.plot(hs, errors, 'o', label='measurements')
plt.plot(x, np.exp(np.polyval(p, np.log(x))), '--', label=f'slope: {p[0]:.2f}')
plt.title('Convergence')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\Delta x$')
plt.ylabel('$l_2$ error')
plt.legend()
clean_figure()
save('convergence_plot.tex')
#plt.savefig('convergence_plot.png')
