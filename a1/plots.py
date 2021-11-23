from tikzplotlib import clean_figure, save
import matplotlib.pyplot as plt
import numpy as np
from solution import LaplaceOnRectangle


def f(x, y):
    pi = np.pi
    return (
        np.sin(pi * y ** 2) * (pi * np.cos(pi * x ** 2) - pi ** 2 * x ** 2 * np.sin(pi * x ** 2))
      + np.sin(pi * x ** 2) * (2 * pi * np.cos(pi * y ** 2) - 4 * pi ** 2 * y ** 2 * np.sin(pi * y ** 2))
    )

L = LaplaceOnRectangle(0.02, 1, 1, f)

for i in range(4):
    L.set_dirchlet(i, lambda x, y: x*0)

L.solve()

L.plot(300, show=False, cmap='jet', extent=[0,1,0,1])
plt.title('rhs $=f(x, y)$, $u_\mathcal{D} = 0$, $\partial\Omega_\mathcal{D}= \partial \Omega$')
clean_figure()
save('problem.tex')

plt.show()


L = LaplaceOnRectangle(0.02, 1, 1, lambda x, y: x*0)

for i in (3,):
    L.set_dirchlet(i, lambda x, y: x*0)

L.set_dirchlet(1, lambda x, y: x*0 + 1)

L.solve()

L.plot(300, show=False, cmap='jet', extent=[0,1,0,1])
plt.title('rhs $=0$, $u_\mathcal{D} = x$, $\partial\Omega_\mathcal{D}= \Gamma_{left} \cup \Gamma_{right} $')
clean_figure()
save('linear.tex')

plt.show()


L = LaplaceOnRectangle(0.02, 1, 1, lambda x, y: x*0)

for i in (0,):
    L.set_neumann(i, lambda x, y: x*0 + 1)

for i in (1, 2, 3):
    L.set_dirchlet(i, lambda x, y: x*0)

L.solve()

L.plot(300, show=False, cmap='jet', extent=[0,1,0,1])
plt.title('rhs $=0$, $u_\mathcal{D} = 0$, $\partial\Omega_\mathcal{D}= \partial \Omega / \Gamma_{bottom} $, $g_\mathcal{N} = 1(y=0)$')
clean_figure()
save('neumann.tex')

plt.show()
