#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdEx patterns (Gerstner et al., Ch.6) — reproduce multiple firing patterns.

We use the R, tau_m form:
  tau_m * dV/dt = (E_L - V) + Delta_T * exp((V - V_T)/Delta_T) + R * (I - w)
  tau_w * dw/dt = a * (V - E_L) - w
Reset when V >= V_peak: V -> V_reset, w -> w + b

Constants (from Table 6.1 caption):
  E_L = -70 mV, R = 500 MΩ, V_T = -50 mV, Delta_T = 2 mV
  Step current: 65 pA (except 'delayed': 25 pA)
  V_peak (not given by Gerstner): set to 20 mV as instructed.

Implements safe exponential clamp and a small ΔV limiter for stability.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Base constants (SI units) ----------
E_L     = -70e-3       # V
R       = 500e6        # Ohm
V_T     = -50e-3       # V
Delta_T = 2e-3         # V
V_peak  = 20e-3        # V  (as requested)

# Simulation controls
DT      = 1e-5         # s (0.01 ms)
T_TOTAL = 1.5          # s total per pattern
EXP_CLIP = 40.0        # clamp argument of exp
DV_LIMIT = 5e-3        # V per step (5 mV max change each step)

# ---------- Table 6.1 parameter sets ----------
# Columns: tau_m [ms], a [nS], tau_w [ms], b [pA], V_reset [mV], I_step [pA]
PATTERNS = {
    "tonic":       dict(tau_m_ms=20.0, a_nS=0.0,  tau_w_ms=30.0,  b_pA=60.0, Vreset_mV=-55.0, I_pA=65.0),
    "adapting":    dict(tau_m_ms=20.0, a_nS=0.0,  tau_w_ms=100.0, b_pA=5.0,  Vreset_mV=-55.0, I_pA=65.0),
    "initial_burst":dict(tau_m_ms=5.0, a_nS=0.5,  tau_w_ms=100.0, b_pA=7.0,  Vreset_mV=-51.0, I_pA=65.0),
    "bursting":    dict(tau_m_ms=5.0,  a_nS=-0.5, tau_w_ms=100.0, b_pA=7.0,  Vreset_mV=-46.0, I_pA=65.0),
    "irregular":   dict(tau_m_ms=9.9, a_nS=-0.5,  tau_w_ms=100.0, b_pA=7.0,  Vreset_mV=-46.0, I_pA=65.0),
    "transient":   dict(tau_m_ms=10.0,a_nS=1.0,   tau_w_ms=100.0, b_pA=10.0, Vreset_mV=-60.0, I_pA=65.0),
    "delayed":     dict(tau_m_ms=5.0, a_nS=-1.0,  tau_w_ms=100.0, b_pA=10.0, Vreset_mV=-60.0, I_pA=25.0),
}

def simulate_adex_R_taum(pattern_name, step_on_s=0.0, step_off_s=None):
    """Simulate one AdEx run for a given Table 6.1 pattern."""
    p = PATTERNS[pattern_name]
    tau_m = p["tau_m_ms"] * 1e-3      # s
    a     = p["a_nS"]      * 1e-9     # S
    tau_w = p["tau_w_ms"]  * 1e-3     # s
    b     = p["b_pA"]      * 1e-12    # A
    Vreset= p["Vreset_mV"] * 1e-3     # V
    Istep = p["I_pA"]      * 1e-12    # A

    if step_off_s is None:
        step_off_s = T_TOTAL  # constant current for whole window

    t = np.arange(0.0, T_TOTAL + DT, DT, dtype=np.float64)
    n = t.size

    V = np.empty(n); w = np.empty(n)
    V[0] = E_L
    w[0] = 0.0

    spikes = []

    def I_of_t(tt):
        return Istep if (step_on_s <= tt < step_off_s) else 0.0

    def safe_exp(arg):
        return np.exp(np.minimum(arg, EXP_CLIP))

    for k in range(n - 1):
        v = V[k]; u = w[k]

        # Pre-reset check
        if v >= V_peak:
            v = Vreset
            u = u + b

        # Input
        I = I_of_t(t[k])

        # AdEx dynamics in (R, tau_m) form
        dVdt = ((E_L - v) + Delta_T * safe_exp((v - V_T)/Delta_T) + R * (I - u)) / tau_m
        dudt = (a * (v - E_L) - u) / tau_w

        dv = dVdt * DT
        du = dudt * DT

        # ΔV limiter
        if dv >  DV_LIMIT: dv =  DV_LIMIT
        if dv < -DV_LIMIT: dv = -DV_LIMIT

        v_next = v + dv
        u_next = u + du

        # Post-update spike
        if v_next >= V_peak:
            spikes.append(t[k+1])
            v_next = Vreset
            u_next = u_next + b

        V[k+1] = v_next
        w[k+1] = u_next

    return t, V, w, np.array(spikes)

# ---------- Choose at least three patterns to reproduce ----------
to_show = ["tonic", "adapting", "bursting"]  # you can add more: "initial_burst", "irregular", "transient", "delayed"

fig, axes = plt.subplots(len(to_show), 2, figsize=(10, 3.2*len(to_show)), sharex=True)
if len(to_show) == 1:
    axes = np.array([axes])  # keep 2D

for i, name in enumerate(to_show):
    t, V, w, spikes = simulate_adex_R_taum(name)

    axV = axes[i, 0]
    axw = axes[i, 1]
    axV.plot(t*1e3, V*1e3)
    axw.plot(t*1e3, w*1e12)

    axV.set_ylabel("V (mV)")
    axw.set_ylabel("w (pA)")
    axV.set_title(f"{name.replace('_',' ').title()} — membrane potential")
    axw.set_title(f"{name.replace('_',' ').title()} — adaptation current")

axes[-1,0].set_xlabel("Time (ms)")
axes[-1,1].set_xlabel("Time (ms)")
plt.tight_layout()
# plt.show()
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_2a.png", dpi=300, bbox_inches="tight")
