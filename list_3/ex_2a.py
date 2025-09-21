# q2a_simulate_T_with_current.py
# Thalamic relay neuron (T-type Ca current) — item (a)
# Plots J(t) on top, then V, n, h, h_T versus t, and saves the figure.

import numpy as np
import matplotlib.pyplot as plt
import utils_T as ut  # utilities for Question 2 (SI units)

# -----------------------------
# Inputs (pA) — change here
# -----------------------------
BASE_pA = 0.0     # base current (pA)
STEP_pA = 40.0    # step amplitude (pA)

# -----------------------------
# Simulation setup (ms → s)
# -----------------------------
T_MS  = 750.0      # total duration (ms)
DT_MS = 0.01       # time step (ms)

# Current protocol: base (0–250 ms), base+step (250–500 ms), base (500–750 ms)
J_of_t = ut.base_plus_step_protocol(BASE_pA, STEP_pA,
                                    t_on_ms=250.0, t_off_ms=500.0, T_ms=T_MS)

# Initial state at rest (V = E_L, gates at steady state for E_L)
y0 = ut.initial_state_at_rest()

# Integrate with RK4 (SI units inside)
t, Y = ut.rk4_solve(ut.thalamic_rhs, # type: ignore
                    t0=0.0, t1=T_MS*ut.ms, dt=DT_MS*ut.ms,
                    y0=y0, Iinj_of_t=J_of_t)

# -----------------------------
# Prepare series for plotting
# -----------------------------
t_ms = t / ut.ms
V_mV = Y[:, 0] / ut.mV
h    = Y[:, 1]
n    = Y[:, 2]
hT   = Y[:, 3]
J_pA = np.array([J_of_t(tt) for tt in t]) / ut.pA  # current in pA for the plot

# -----------------------------
# Plots: J(t) on top, then V, n, h, h_T
# -----------------------------
fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1, 2, 1, 1, 1]})

# Current protocol
axes[0].plot(t_ms, J_pA, color="tab:red", lw=1.8)
axes[0].set_ylabel("J (pA)")
axes[0].set_title(f"base={BASE_pA} pA, step={STEP_pA} pA")
axes[0].grid(True, alpha=0.3)

# Membrane potential
axes[1].plot(t_ms, V_mV, color="tab:blue", lw=1.6)
axes[1].set_ylabel("V (mV)")
axes[1].grid(True, alpha=0.3)

# Gating variables
axes[2].plot(t_ms, n, lw=1.2)
axes[2].set_ylabel("n"); axes[2].grid(True, alpha=0.3)

axes[3].plot(t_ms, h, lw=1.2)
axes[3].set_ylabel("h"); axes[3].grid(True, alpha=0.3)

axes[4].plot(t_ms, hT, lw=1.2)
axes[4].set_ylabel("h_T"); axes[4].set_xlabel("t (ms)")
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_2a.png", dpi=300)
# plt.show()
