import numpy as np
import matplotlib.pyplot as plt
from utils import euler, eq_membrana, analytic_solution
import time

# Configurações
t0 = 0
tf = 20
h  = [i * 1e3 for i in [5e-4, 1e-3, 2e-3, 1e-2, 1e-1]]
y0 = 0

plt.figure(figsize=(15, 6))


# Loop para g_m
for delta_t in h:

   # Função wrapper
   def f(t, y):
      g_m = 100
      C_m = 1
      E = 0
      J_inj = 2
      return eq_membrana(t, y, g_m, C_m, E, J_inj, SI=False)

   # Resolver usando o método de Euler
   start = time.time()
   sol_num = euler(f, t0, tf, y0, delta_t)
   print('Δt = ' + str(delta_t) + 'ms',  str(time.time() - start))

   # Calcular erro relativo 
   t_list = sol_num[0]
   V_num_list = sol_num[1]
   V_ana_list = analytic_solution(t_list, y0, g_m=100, C_m=1, E=0, J_inj=2, SI=False)
   err = np.abs((V_num_list - V_ana_list) / (V_ana_list + 1e-10))

   # Plotar
   plt.subplot(1, 2, 1)
   plt.plot(sol_num[0], sol_num[1], label='Δt = ' + str(delta_t) + 'ms')
   plt.subplot(1, 2, 2)
   plt.plot(t_list, err, label='Δt = ' + str(delta_t) + 'ms')

plt.subplot(1, 2, 1)
ts = np.array([i for i in np.arange(t0, tf, 1e-3)])
sol_ana = analytic_solution(ts, y0, g_m=100, C_m=1, E=0, J_inj=2, SI=False)
plt.plot(ts, sol_ana, label='analítico')

plt.xlabel("t [ms]")
plt.ylabel("V(t) [mV]")
plt.grid(True, alpha=.3)
plt.legend()
plt.title("Soluções")

plt.subplot(1, 2, 2)
plt.xlabel("t [ms]")
plt.ylabel("Erro relativo")
plt.grid(True, alpha=.3)
plt.legend()
plt.title("Erro relativo")

plt.savefig("Lista 1/ex_5.png")
plt.show()





