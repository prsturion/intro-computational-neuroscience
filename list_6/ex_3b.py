# lif_adaptive_q3b_zero_segment.py
# LIF with spike-triggered adaptation: f–I curves for f1 and f∞.
# This version forces a visible zero-frequency segment: if <2 spikes, both f1 and f∞ are set to 0.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Model parameters (SI units)
# ----------------------------
V_rest  = -70e-3   # V
R       = 40e6     # Ohm
C       = 250e-12  # F
V_th    = -50e-3   # V
V_reset = -80e-3   # V  (deeper reset for Q3)

# Adaptation
b       = 1e-9     # S (ΔG_a at each spike)
tau_a   = 200e-3   # s

# Initial conditions
V0      = V_rest
Ga0     = 0.0

# Simulation config
dt      = 0.01e-3  # s
T_max   = 5.0      # s
N_steps = int(np.round(T_max / dt))

# Current sweep: 400 pA → 800 pA (step 20 pA)
I_vals  = (np.arange(400, 801, 10) * 1e-12)

# ----------------------------
# Simulator
# ----------------------------
def simulate_adaptive(I):
    """Euler explicit integration with threshold/reset and adaptation jump."""
    V = V0
    Ga = Ga0
    spikes = []

    for k in range(N_steps):
        dVdt  = ((V_rest - V) / R + Ga * (V_reset - V) + I) / C
        dGadt = -Ga / tau_a

        V_next  = V + dt * dVdt
        Ga_next = Ga + dt * dGadt

        if (V < V_th) and (V_next >= V_th):
            denom = V_next - V
            frac  = (V_th - V) / denom if denom != 0.0 else 1.0
            frac  = float(np.clip(frac, 0.0, 1.0))
            t_star = (k * dt) + dt * frac
            spikes.append(t_star)

            V_next  = V_reset
            Ga_next = Ga_next + b

        V, Ga = V_next, Ga_next

    return np.asarray(spikes)

def f1_and_finf_from_spikes(spike_times):
    """
    Returns (f1, finf) in Hz.
    If fewer than 2 spikes: both are defined as 0 to show a zero-frequency segment.
    Otherwise:
      f1  = 1 / (t2 - t1)
      finf = 1 / mean(last 5 ISIs)  (or last ISI if fewer than 5)
    """
    if spike_times.size < 2:
        return 0.0, 0.0
    isis = np.diff(spike_times)
    T1 = float(isis[0])
    Tinf = float(np.mean(isis[-5:])) if isis.size >= 5 else float(isis[-1])
    return 1.0 / T1, 1.0 / Tinf

# ----------------------------
# Sweep
# ----------------------------
f1_list, finf_list, spike_counts = [], [], []

for I in I_vals:
    spk = simulate_adaptive(I)
    spike_counts.append(spk.size)
    f1, finf = f1_and_finf_from_spikes(spk)
    f1_list.append(f1)
    finf_list.append(finf)

f1_arr   = np.array(f1_list)
finf_arr = np.array(finf_list)

# Onset current (first I with >= 2 spikes)
onset_idx = next((i for i, c in enumerate(spike_counts) if c >= 2), None)
if onset_idx is not None:
    print(f"I_onset ≈ {I_vals[onset_idx]*1e12:.0f} pA  |  f1_onset ≈ {f1_arr[onset_idx]:.2f} Hz")

# Type-I/II heuristic (based on onset frequency)
if onset_idx is not None:
    neuron_type = "Tipo I" if f1_arr[onset_idx] < 5.0 else "Tipo II"
else:
    neuron_type = "Indeterminado (sem onset na faixa)"

print(f"Classificação heurística: {neuron_type}")

# ----------------------------
# Plot (Portuguese labels)
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(I_vals*1e12, f1_arr,  "o-", label="f₁ = 1/T₁")
plt.plot(I_vals*1e12, finf_arr, "s-", label="f∞ = 1/T∞")
plt.xlabel("Corrente I (pA)")
plt.ylabel("Frequência (Hz)")
plt.title("Curvas f–I com adaptação")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("intro-computational-neuroscience/list_6/figures/ex_3b.png", dpi=300, bbox_inches="tight")
