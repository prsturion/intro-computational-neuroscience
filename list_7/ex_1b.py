#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdEx f–I curves for T1^{-1} and T_inf^{-1}.

Simulates 5 s for 20 constant-input currents between 200 and 500 pA.
For each current:
- Detect spike times (when V crosses V_peak -> reset).
- T1  = t_spikes[1] - t_spikes[0]       (if >= 2 spikes)
- Tinf = mean ISI over the last 2 s     (if >= 2 spikes in that window)
        otherwise mean of the last up to 5 ISIs, if available.

Numerics: forward Euler, safe exponential clamp, pre/post spike reset, ΔV limiter.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# AdEx single-run simulator
# ------------------------
def simulate_adex_constant_I(
    I_pA,
    T=5.0,                 # total sim time [s]
    dt=1e-5,               # 0.01 ms (use 1e-6 if strictly required after validating)
    # Parameters (Paul Miller, 2018)
    C_pF=100.0, G_nS=10.0, Vr_mV=-70.0, VL_mV=-50.0, Vreset_mV=-80.0,
    dL_mV=2.0, a_nS=2.0, b_pA=20.0, tau_u_ms=200.0, Vpeak_mV=50.0,
    exp_clip=40.0, max_dV_per_step_mV=5.0
):
    # ---- SI units ----
    C      = C_pF   * 1e-12
    G      = G_nS   * 1e-9
    Vr     = Vr_mV  * 1e-3
    VL     = VL_mV  * 1e-3
    Vreset = Vreset_mV * 1e-3
    dL     = dL_mV  * 1e-3
    a      = a_nS   * 1e-9
    b      = b_pA   * 1e-12
    tau_u  = tau_u_ms * 1e-3
    Vpeak  = Vpeak_mV * 1e-3
    max_dV = max_dV_per_step_mV * 1e-3
    I_const = I_pA * 1e-12

    # ---- time ----
    t = np.arange(0.0, T + dt, dt, dtype=np.float64)
    n = t.size

    # ---- states ----
    V = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)
    V[0] = Vr
    u[0] = 0.0

    spike_times = []

    def safe_exp_arg(x):
        return np.exp(np.minimum(x, exp_clip))

    for k in range(n - 1):
        v = V[k]
        w = u[k]

        # Pre-check reset
        if v >= Vpeak:
            v = Vreset
            w = w + b

        # Derivatives (forward Euler)
        exp_term = safe_exp_arg((v - VL) / dL)
        dVdt = (-G * (v - Vr) + G * dL * exp_term - w + I_const) / C
        dudt = (a * (v - Vr) - w) / tau_u

        dv = dVdt * dt
        du = dudt * dt

        # ΔV limiter
        if dv >  max_dV: dv =  max_dV
        if dv < -max_dV: dv = -max_dV

        v_next = v + dv
        w_next = w + du

        # Post-update spike/reset and record spike time
        if v_next >= Vpeak:
            # Spike occurs within [t[k], t[k+1]]. Use t[k+1] as spike time (OK for small dt).
            spike_times.append(t[k+1])
            v_next = Vreset
            w_next = w_next + b

        V[k+1] = v_next
        u[k+1] = w_next

    return t, V, u, np.array(spike_times, dtype=np.float64)

# ------------------------
# Extract T1 and T_inf
# ------------------------
def compute_T1_and_Tinf(spike_times, T_total, tail_window=2.0, max_last_intervals=5):
    """
    spike_times: array of spike times [s]
    returns (T1, Tinf) with np.nan if not defined.
    """
    if spike_times.size < 2:
        return np.nan, np.nan

    # T1
    T1 = spike_times[1] - spike_times[0]

    # T_inf: ISIs in the last 'tail_window' seconds
    isis = np.diff(spike_times)
    tail_mask = spike_times[1:] >= (T_total - tail_window)
    isis_tail = isis[tail_mask]

    if isis_tail.size >= 1:
        Tinf = np.mean(isis_tail)
    else:
        # fallback: mean of the last up to K intervals
        if isis.size >= 1:
            K = min(max_last_intervals, isis.size)
            Tinf = np.mean(isis[-K:])
        else:
            Tinf = np.nan

    return T1, Tinf

# ------------------------
# Sweep currents and plot f–I
# ------------------------
if __name__ == "__main__":
    T = 5.0
    dt = 1e-5  # use 1e-6 for the official run if required

    currents_pA = np.linspace(200.0, 500.0, 20)
    f1 = np.zeros_like(currents_pA)
    finf = np.zeros_like(currents_pA)

    for i, I_pA in enumerate(currents_pA):
        _, _, _, spikes = simulate_adex_constant_I(I_pA, T=T, dt=dt)
        T1, Tinf = compute_T1_and_Tinf(spikes, T_total=T)
        f1[i]   = 0.0 if np.isnan(T1)   else 1.0 / T1
        finf[i] = 0.0 if np.isnan(Tinf) else 1.0 / Tinf

    # Plot f–I curves
    plt.figure(figsize=(7,5))
    plt.plot(currents_pA, f1,  marker="o", linestyle="", label="f1 = 1/T1")
    plt.plot(currents_pA, finf, marker="s", linestyle="", label="f∞ = 1/T∞")
    plt.xlabel("Input current I (pA)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("AdEx f–I curves: initial vs steady-state")
    plt.legend()
    plt.tight_layout()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_7/figures/ex_1b.png", dpi=300, bbox_inches="tight")
