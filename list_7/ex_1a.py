#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdEx neuron (stable Euler solver) with spike pre-check and safe exponential.

Model:
C dV/dt = -G (V - V_r) + G*ΔL*exp((V - V_L)/ΔL) - u + I(t)
τ_u du/dt = a (V - V_r) - u
If V >= V_peak: V -> V_reset; u -> u + b

Key numerics:
- Forward Euler (more stable than RK4 here due to very steep exponential).
- Pre-check for spike (apply reset before derivative when V is already high).
- Clamp exp argument to avoid overflow.
- Optional limiter on ΔV per step to avoid runaway if something goes wrong.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_adex(
    T=3.0,                 # total time [s]
    dt=1e-5,               # time step [s] -> 0.01 ms (use 1e-6 after validating)
    I_step_pA=221.0,       # step amplitude [pA]
    t_on=0.5,              # step ON [s]
    t_off=2.5,             # step OFF [s]
    # Parameters (Paul Miller, 2018) in human units -> converted to SI below
    C_pF=100.0,
    G_nS=10.0,
    Vr_mV=-70.0,
    VL_mV=-50.0,
    Vreset_mV=-80.0,
    dL_mV=2.0,
    a_nS=2.0,
    b_pA=20.0,
    tau_u_ms=200.0,
    Vpeak_mV=50.0,
    # Numerics
    exp_clip=40.0,         # clamp for exponential argument
    max_dV_per_step_mV=5.0 # optional limiter for ΔV per step (safety net)
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

    # ---- time ----
    t = np.arange(0.0, T + dt, dt, dtype=np.float64)
    n = t.size

    # ---- input current ----
    I = np.zeros(n, dtype=np.float64)
    on_idx  = int(round(t_on  / dt))
    off_idx = int(round(t_off / dt))
    I[on_idx:off_idx] = I_step_pA * 1e-12  # Amperes

    # ---- states ----
    V = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)
    V[0] = Vr
    u[0] = 0.0

    # ---- helpers ----
    def safe_exp_arg(x):
        # clamp only the upper side; lower side can be very negative safely
        return np.exp(np.minimum(x, exp_clip))

    for k in range(n - 1):
        v = V[k]
        w = u[k]

        # Pre-check: if already above peak (can happen numerically), reset first
        if v >= Vpeak:
            v = Vreset
            w = w + b

        # Derivatives (forward Euler)
        exp_term = safe_exp_arg((v - VL) / dL)
        dVdt = (-G * (v - Vr) + G * dL * exp_term - w + I[k]) / C
        dudt = (a * (v - Vr) - w) / tau_u

        dv = dVdt * dt
        du = dudt * dt

        # Safety limiter on ΔV per step (prevents rare runaways)
        if dv >  max_dV: dv =  max_dV
        if dv < -max_dV: dv = -max_dV

        v_next = v + dv
        w_next = w + du

        # Spike/reset after update
        if v_next >= Vpeak:
            v_next = Vreset
            w_next = w_next + b

        V[k+1] = v_next
        u[k+1] = w_next

    return t, I, V, u

if __name__ == "__main__":
    # Start with dt=1e-5 (0.01 ms) for stability/speed; switch to 1e-6 after validating.
    t, I, V, U = simulate_adex(dt=1e-5)

    # Plot in convenient units
    t_ms = t * 1e3
    I_pA = I * 1e12
    V_mV = V * 1e3
    U_pA = U * 1e12

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t_ms, I_pA)
    axs[0].set_ylabel("I (pA)")
    axs[0].set_title("Step current input")

    axs[1].plot(t_ms, V_mV)
    axs[1].set_ylabel("V (mV)")
    axs[1].set_title("Membrane potential")

    axs[2].plot(t_ms, U_pA)
    axs[2].set_ylabel("u (pA)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_title("Adaptation current")

    plt.tight_layout()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_7/figures/ex_1a.png", dpi=300, bbox_inches="tight")
