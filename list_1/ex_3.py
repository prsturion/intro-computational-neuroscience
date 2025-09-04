import numpy as np
import matplotlib.pyplot as plt
from utils import euler, eq_membrana

# Configurações
t0 = 0
tf = 80
h  = 1
y0 = 0

# Loop para g_m
g_m = [10, 20, 50, 100, 200]
for g in g_m:

   # Função wrapper
   def f(t, y):
      g_m = g
      C_m = 1
      E = 0
      J_inj = 2 if t >= 5 and t < 50 else 0
      return eq_membrana(t, y, g_m, C_m, E, J_inj, SI=False)

   # Resolver usando o método de Euler
   sol = euler(f, t0, tf, y0, h)

   # Plotar
   plt.plot(sol[0], sol[1], label='g_m = ' + str(g))

plt.xlabel("t [ms]")
plt.ylabel("V(t) [mV]")
plt.grid(True, alpha=.3)
plt.legend()
plt.show()
