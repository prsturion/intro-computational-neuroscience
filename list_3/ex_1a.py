# plot_gating.py
import numpy as np
import matplotlib.pyplot as plt
import utils_A as ut   # import all gating functions from utils_A.py

# Voltage range (mV)
V = np.linspace(-100, 50, 500)

# Dictionary with gating variables and their functions
gates = {
    "n": (ut.n_inf, ut.tau_n),
    "m": (ut.m_inf, ut.tau_m),
    "h": (ut.h_inf, ut.tau_h),
    "a": (ut.a_inf, ut.tau_a),
    "b": (ut.b_inf, ut.tau_b),
}

# Plot X_inf(V) and tau_X(V)
fig, axes = plt.subplots(len(gates), 2, figsize=(10, 10))

for i, (gate, (f_inf, f_tau)) in enumerate(gates.items()):
    # Left column: steady-state activation/inactivation
    axes[i, 0].plot(V, f_inf(V), label=f"${gate}_\infty(V)$") # type: ignore
    axes[i, 0].set_ylabel(f"{gate}∞")
    axes[i, 0].set_xlim(V[0], V[-1])
    axes[i, 0].legend()
    axes[i, 0].grid(True)

    # Right column: time constants
    axes[i, 1].plot(V, f_tau(V), label=f"$\\tau_{gate}(V)$", color="tab:red")
    axes[i, 1].set_ylabel(f"τ_{gate} (ms)")
    axes[i, 1].set_xlim(V[0], V[-1])
    axes[i, 1].legend()
    axes[i, 1].grid(True)

# Common labels
axes[-1, 0].set_xlabel("Membrane potential V (mV)")
axes[-1, 1].set_xlabel("Membrane potential V (mV)")

plt.tight_layout()
# plt.show()
plt.savefig('./intro-computational-neuroscience/list_3/figures/ex_1a.png')
