import numpy as np
import matplotlib.pyplot as plt
import utils_ex_6 as ut

# Definindo o intervalo de Vd
Vd_values = np.linspace(-100, 0, 300) * 1e-3  # Valores de Vd no intervalo de -100 a 0

# Calculando m_h_infinito e tau_m_h
m_h_inf_values = ut.mh_inf(Vd_values)
tau_m_h_values = ut.tau_mh(Vd_values)

# Gerando os gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Gráfico de m_h_infinito
ax1.plot(Vd_values * 1e3, m_h_inf_values, lw=1.5, color='b')
ax1.set_ylabel(r"$m_{h,\infty}$")
ax1.grid(True)

# Gráfico de tau_m_h
ax2.plot(Vd_values * 1e3, tau_m_h_values, lw=1.5, color='r')
ax2.set_xlabel(r"$V_D$ (mV)")
ax2.set_ylabel(r"$\tau_{m,h}$")
ax2.grid(True)

# Exibindo a figura
plt.tight_layout()
# plt.show()
plt.savefig('intro-computational-neuroscience/list_4/figures/ex_6a.png')