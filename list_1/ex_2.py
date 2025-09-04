import numpy as np
import matplotlib.pyplot as plt
from utils import euler, eq_membrana

# Configurações
t0 = 0
tf = 80
h  = 1
y0 = 0

# Função wrapper
def f(t, y):
    g_m = 100
    C_m = 1
    E = 0
    J_inj = 2 if t >= 5 and t < 50 else 0
    return eq_membrana(t, y, g_m, C_m, E, J_inj, SI=False)

# Resolver usando o método de Euler
sol = euler(f, t0, tf, y0, h)

# Plotar
plt.plot(sol[0], sol[1])
plt.xlabel("t [ms]")
plt.ylabel("V(t) [mV]")
plt.grid(True, alpha=.3)
plt.show()
