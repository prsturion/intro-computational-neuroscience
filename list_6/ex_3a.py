# lif_adaptive_q3a.py
# LIF with spike-triggered adaptation current:
#   C dV/dt = (V_rest - V)/R + G_a*(V_reset - V) + I(t)
#   dG_a/dt = -G_a / tau_a
# Spike rule: if V >= V_th (from below), then V <- V_reset and G_a <- G_a + b.
# No explicit absolute refractory here (as per the statement).

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Fixed neuron parameters (SI units)
# ----------------------------
V_rest  = -70e-3   # V (V_rep)
R       = 40e6     # Ohm
C       = 250e-12  # F
V_th    = -50e-3   # V (threshold V_L)
V_reset = -80e-3   # V (V_redef, deeper reset for Q3)
# Adaptation parameters
b       = 1e-9     # S  (1 nS added to G_a at each spike)
tau_a   = 200e-3   # s  (200 ms)

# Initial conditions
V0      = V_rest   # V(0) = V_rest
Ga0     = 0.0      # G_a(0) = 0

# Simulation settings
dt      = 0.01e-3  # s (0.01 ms)
t_max   = 1.5      # s
n_steps = int(np.round(t_max / dt))
t       = np.linspace(0.0, n_steps*dt, n_steps+1)

# ----------------------------
# Input current: rectangular pulse
#   I = 501 pA between 0.5 s and 1.0 s; zero otherwise.
# ----------------------------
I_amp   = 501e-12  # A
t_on    = 0.5      # s
t_off   = 1.0      # s

def I_of_t(t):
    """Piecewise-constant current pulse."""
    return np.where((t >= t_on) & (t <= t_off), I_amp, 0.0)

# Precompute I(t) for plotting and for use inside the loop
I_t = I_of_t(t)

# ----------------------------
# State arrays
# ----------------------------
V  = np.empty_like(t);   V[0]  = V0
Ga = np.empty_like(t);   Ga[0] = Ga0
spike_times = []

# ----------------------------
# Euler integration loop
# ----------------------------
for k in range(n_steps):
    # Drift terms (currents / C)
    dVdt  = ((V_rest - V[k]) / R + Ga[k] * (V_reset - V[k]) + I_t[k]) / C
    dGadt = -Ga[k] / tau_a

    # Euler proposals
    V_next  = V[k]  + dt * dVdt
    Ga_next = Ga[k] + dt * dGadt

    # Threshold crossing from below: V_k < V_th <= V_{k+1}
    if (V[k] < V_th) and (V_next >= V_th):
        # Optional linear interpolation for spike time (not strictly needed for plots)
        denom = V_next - V[k]
        frac  = (V_th - V[k]) / denom if denom != 0.0 else 1.0
        frac  = float(np.clip(frac, 0.0, 1.0))
        spike_times.append(t[k] + dt * frac)

        # Apply reset and adaptation jump
        V_next  = V_reset
        Ga_next = Ga_next + b

    # Commit step
    V[k+1]  = V_next
    Ga[k+1] = Ga_next

print(f"Number of spikes: {len(spike_times)}")

# ----------------------------
# Plots (Portuguese labels)
# ----------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# 1) Current
axs[0].plot(t*1e3, I_t*1e12, color="tab:blue")
axs[0].set_ylabel("Corrente I (pA)")
axs[0].set_title("Pulso de corrente, potencial de membrana e condutância adaptativa")
axs[0].grid(True)

# 2) Membrane potential
axs[1].plot(t*1e3, V*1e3, color="tab:orange")
axs[1].axhline(V_th*1e3, ls="--", color="k", lw=1.0, label="Limiar")
axs[1].set_ylabel("Voltagem V (mV)")
axs[1].legend(loc="upper right")
axs[1].grid(True)

# 3) Adaptive conductance
axs[2].plot(t*1e3, Ga*1e9, color="tab:green")
axs[2].set_xlabel("Tempo (ms)")
axs[2].set_ylabel("Condutância G_a (nS)")
axs[2].grid(True)

plt.tight_layout()
# plt.show()
plt.savefig("intro-computational-neuroscience/list_6/figures/ex_3a.png", dpi=300, bbox_inches="tight")
