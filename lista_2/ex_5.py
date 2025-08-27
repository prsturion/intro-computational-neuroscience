from utils import *
import matplotlib.pyplot as plt
import numpy as np

t0 = 0
tf = 500
delta_t = .025

V0 = -65
n0 = 0.32
m0 = 0.05
h0 = 0.6

J_min = 6.183
J_max = 100
J_values = np.linspace(J_min, J_max, 100)

def HH_eq_wrapper(t, V, n, m, h):

   HH_params = {
      'C_m': 1,
      'g_Na': 120,
      'g_K': 36,
      'g_V': 0.3,
      'E_Na': 50,
      'E_K': -77,
      'E_V': -54.4,
      'J_inj': J_inj(t)
   }

   return HH_equation(t, V, n, m, h, **HH_params)

HH_n_gating = lambda t, V, n, m, h: HH_gating_eq(t, V, n, m, h, gating='n')
HH_m_gating = lambda t, V, n, m, h: HH_gating_eq(t, V, n, m, h, gating='m')
HH_h_gating = lambda t, V, n, m, h: HH_gating_eq(t, V, n, m, h, gating='h')

freqs = list()
for J in J_values:
   def J_inj(t):
      if t >= 0 and t < 0:
         return 0
      elif t >= 50 and t < 450:
         return J
      else:
         return 0

   sol = euler([HH_eq_wrapper, HH_n_gating, HH_m_gating, HH_h_gating], [V0, n0, m0, h0], t0, tf, delta_t)

   t_axis = np.array(sol[0])
   V_axis = sol[1][0]

   cross_idxes = find_crossings(V_axis, -30, direction='up')

   crossing_ts = t_axis[cross_idxes]

   freq = ((len(crossing_ts) - 1) / (crossing_ts[-1] - crossing_ts[0])) * 1000 if len(crossing_ts) > 1 else 0 # convert to Hz

   freqs.append(freq)




# Plotting
plt.plot(J_values, freqs, color='olivedrab')

plt.ylabel("f [Hz]")
plt.xlabel("J [μA/cm²]")
plt.grid(True, alpha=.3)
plt.gca().ticklabel_format(useOffset=False)

plt.savefig("./intro-computational-neuroscience/lista_2/figures/ex_5.png")
# plt.show()

