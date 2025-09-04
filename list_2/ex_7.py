from utils import *
import matplotlib.pyplot as plt

t0 = 0
tf = 30
delta_t = 1e-3

T = 1.5
t1 = 5
t2 = t1 + T

def J_inj(t):
   if t >= t1 and t < t2:
      return -15
   else:
      return 0

V0 = -65
n0 = 0.32
m0 = 0.05
h0 = 0.6

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


sol = euler([HH_eq_wrapper, HH_n_gating, HH_m_gating, HH_h_gating], [V0, n0, m0, h0], t0, tf, delta_t)

t_axis = sol[0]
V_axis = sol[1][0]
n_axis = sol[1][1]
m_axis = sol[1][2]
h_axis = sol[1][3]
J_axis = [J_inj(t) for t in t_axis]

t_fire = 0
fire_latency: None | float = None
for i in range(len(V_axis)):
   if V_axis[i] > 20:
      t_fire = t_axis[i]
      fire_latency = t_fire - t2

if fire_latency is not None:
   print(f"Fire latency: {fire_latency:.2f} ms")

# Plotting
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(t_axis, V_axis, color='indigo')
# plt.xlabel("t [ms]")
plt.ylabel("V(t) [mV]")
plt.grid(True, alpha=.3)
plt.text(20, 1, f'Latência = {fire_latency:.2f} ms', fontsize=10)
plt.gca().ticklabel_format(useOffset=False)

plt.subplot(3, 1, 2)
plt.plot(t_axis, n_axis, label='n(t)', color='blue')
plt.plot(t_axis, m_axis, label='m(t)', color='black')
plt.plot(t_axis, h_axis, label='h(t)', color='red')
# plt.xlabel("t [ms]")
plt.ylabel("Variáveis de gating")
plt.grid(True, alpha=.3)
plt.gca().ticklabel_format(useOffset=False)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_axis, J_axis)
plt.xlabel("t [ms]")
plt.ylabel("J(t) [μA/cm²]")
plt.grid(True, alpha=.3)
plt.gca().ticklabel_format(useOffset=False)

plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_2/figures/ex_7_2.png")
# plt.show()

