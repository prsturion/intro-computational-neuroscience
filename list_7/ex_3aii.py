#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 (a)(ii) — Izhikevich model (quadratic form)
Simulate for 500 ms with a square current pulse from 20 ms to 450 ms.
Currents: 300 pA, 370 pA, 500 pA, 550 pA.
Produce one figure with four subplots (voltage vs time), one for each I.

Units used throughout:
  v in mV, u in pA, t in ms
  C in pF, k in pA/mV^2, b in nS (= pA/mV), a in 1/ms
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Model parameters (given by the problem) ----------------
C      = 150.0      # pF
vr     = -75.0      # mV
vL     = -45.0      # mV
kpar   = 1.2        # pA/mV^2
a      = 0.01       # 1/ms
b      = 5.0        # nS = pA/mV
c      = -56.0      # mV
d      = 130.0      # pA
v_peak = 50.0       # mV

# ---------------- Simulation setup ----------------
dt   = 0.001        # ms (Euler or RK4 step)
T    = 500.0        # ms total duration
t    = np.arange(0.0, T + dt, dt)

# Square pulse I(t): on in [t_on, t_off]
def I_square(t_ms: float, I_amp_pA: float, t_on_ms: float = 20.0, t_off_ms: float = 450.0) -> float:
    return I_amp_pA if (t_ms >= t_on_ms and t_ms <= t_off_ms) else 0.0

# Right-hand sides (units preserved)
def dvdt(v, u, I):
    # C dv/dt = k (v-vr)(v-vL) - u + I  -> dv/dt in mV/ms
    return (kpar * (v - vr) * (v - vL) - u + I) / C

def dudt(v, u):
    # du/dt = a ( b (v-vr) - u ) -> pA/ms
    return a * (b * (v - vr) - u)

# One simulation for a given current amplitude (pA)
def simulate_izh(I_amp_pA: float, integrator: str = "rk4"):
    v = np.empty_like(t)
    u = np.empty_like(t)
    v[0] = vr
    u[0] = 0.0

    if integrator.lower() == "euler":
        for k in range(len(t) - 1):
            Ik = I_square(t[k], I_amp_pA)
            v[k+1] = v[k] + dvdt(v[k], u[k], Ik) * dt
            u[k+1] = u[k] + dudt(v[k], u[k]) * dt
            if v[k+1] >= v_peak:
                v[k+1] = c
                u[k+1] += d
    else:  # RK4 (recommended)
        for k in range(len(t) - 1):
            Ik  = I_square(t[k], I_amp_pA)
            Ikh = I_square(t[k] + 0.5*dt, I_amp_pA)
            Ik1 = I_square(t[k] + dt, I_amp_pA)

            # k1
            k1v = dvdt(v[k],             u[k],             Ik)
            k1u = dudt(v[k],             u[k])
            # k2
            vk2 = v[k] + 0.5*dt*k1v
            uk2 = u[k] + 0.5*dt*k1u
            k2v = dvdt(vk2,              uk2,              Ikh)
            k2u = dudt(vk2,              uk2)
            # k3
            vk3 = v[k] + 0.5*dt*k2v
            uk3 = u[k] + 0.5*dt*k2u
            k3v = dvdt(vk3,              uk3,              Ikh)
            k3u = dudt(vk3,              uk3)
            # k4
            vk4 = v[k] + dt*k3v
            uk4 = u[k] + dt*k3u
            k4v = dvdt(vk4,              uk4,              Ik1)
            k4u = dudt(vk4,              uk4)

            v[k+1] = v[k] + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
            u[k+1] = u[k] + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)

            if v[k+1] >= v_peak:
                v[k+1] = c
                u[k+1] += d

    return v, u

# ---------------- Run for the four requested amplitudes ----------------
I_list = [300.0, 370.0, 500.0, 550.0]  # pA

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
axes = axes.ravel()

for ax, Iamp in zip(axes, I_list):
    v, u = simulate_izh(Iamp, integrator="rk4")
    ax.plot(t, v, lw=1.2)
    # visualize the pulse window (optional, light band)
    ax.axvspan(20.0, 450.0, alpha=0.08, hatch=None)
    ax.set_title(f"I = {Iamp:.0f} pA")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("v (mV)")
    ax.set_xlim(0.0, 500.0)
    ax.grid(alpha=0.15, linestyle=":")

fig.suptitle("Izhikevich model — Voltage traces for square pulse (20–450 ms)", y=0.98)
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3aii.png", dpi=300)
# plt.show()
