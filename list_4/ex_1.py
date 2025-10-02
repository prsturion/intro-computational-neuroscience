
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n,
    alpha_mca, beta_mca, alpha_mkca, beta_mkca,
    alpha_mkahp, beta_mkahp,
)

# ----------------------------
# Malhas de varredura
# ----------------------------
V_mV = np.linspace(-85.0, 50.0, 1000)   # mV
V = V_mV / 1e3                          # V
Ca = np.linspace(0.0, 2e-3, 800)        # M

# ----------------------------
# Funções a plotar
# ----------------------------
functions = [
    (alpha_m,    V,   V_mV, "α_m(V)",    "V [mV]", "s⁻¹"),
    (beta_m,     V,   V_mV, "β_m(V)",    "V [mV]", "s⁻¹"),
    (alpha_h,    V,   V_mV, "α_h(V)",    "V [mV]", "s⁻¹"),
    (beta_h,     V,   V_mV, "β_h(V)",    "V [mV]", "s⁻¹"),
    (alpha_n,    V,   V_mV, "α_n(V)",    "V [mV]", "s⁻¹"),
    (beta_n,     V,   V_mV, "β_n(V)",    "V [mV]", "s⁻¹"),
    (alpha_mca,  V,   V_mV, "α_mCa(V)",  "V [mV]", "s⁻¹"),
    (beta_mca,   V,   V_mV, "β_mCa(V)",  "V [mV]", "s⁻¹"),
    (alpha_mkca, V,   V_mV, "α_mKCa(V)", "V [mV]", "s⁻¹"),
    (beta_mkca,  V,   V_mV, "β_mKCa(V)", "V [mV]", "s⁻¹"),
    (alpha_mkahp,Ca,  Ca,   "α_mKAHp([Ca])", "[Ca] [M]", "s⁻¹"),
    (beta_mkahp, Ca,  Ca,   "β_mKAHp([Ca])", "[Ca] [M]", "s⁻¹"),
]

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
axes = axes.flatten()

for ax, (fun, x_eval, x_plot, title, xlabel, ylabel) in zip(axes, functions):
    y = fun(x_eval)
    ax.plot(x_plot, y, lw=1.5, color='maroon')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_4/figures/ex_1.png", dpi=150)
# plt.show()
