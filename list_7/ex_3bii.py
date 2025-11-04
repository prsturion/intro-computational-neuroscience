#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 (b)(ii) — Izhikevich (parameter set b) — Voltage traces per pulse
Currents: 200, 300, 400, 500 pA. Pulse: 20–450 ms.
Style matches the earlier (a)(ii): common y-limits, shaded pulse, spike markers, rate on titles.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------- Izhikevich parameters: set (b) --------
C      = 50.0      # pF
vr     = -60.0     # mV
vL     = -40.0     # mV
kpar   = 1.5       # pA/mV^2
a      = 0.03      # 1/ms
b      = 1.0       # nS (pA/mV)
c      = -40.0     # mV  (reset)
d      = 150.0     # pA  (adaptation jump)
v_peak = 25.0      # mV  (cutoff)

# -------- Simulation grid --------
dt   = 0.001                   # ms
T    = 500.0                   # ms
t    = np.arange(0.0, T+dt, dt)
t_on, t_off = 20.0, 450.0      # ms

def I_square(t_ms, I_amp):
    return I_amp if (t_ms >= t_on and t_ms <= t_off) else 0.0

def dvdt(v, u, I):
    # C dv/dt = k (v-vr)(v-vL) - u + I
    return (kpar*(v - vr)*(v - vL) - u + I) / C

def dudt(v, u):
    # du/dt = a ( b(v-vr) - u )
    return a*(b*(v - vr) - u)

def simulate(I_amp):
    """RK4 + reset rule. Returns V(t), U(t), spike_times (ms)."""
    V = np.empty_like(t); U = np.empty_like(t)
    V[0] = vr; U[0] = 0.0
    spike_times = []

    for k in range(len(t)-1):
        Ik  = I_square(t[k], I_amp)
        Ikh = I_square(t[k] + 0.5*dt, I_amp)
        Ik1 = I_square(t[k] + dt, I_amp)

        k1v = dvdt(V[k], U[k], Ik);   k1u = dudt(V[k], U[k])
        vk2 = V[k] + 0.5*dt*k1v;      uk2 = U[k] + 0.5*dt*k1u
        k2v = dvdt(vk2, uk2, Ikh);    k2u = dudt(vk2, uk2)
        vk3 = V[k] + 0.5*dt*k2v;      uk3 = U[k] + 0.5*dt*k2u
        k3v = dvdt(vk3, uk3, Ikh);    k3u = dudt(vk3, uk3)
        vk4 = V[k] + dt*k3v;          uk4 = U[k] + dt*k3u
        k4v = dvdt(vk4, uk4, Ik1);    k4u = dudt(vk4, uk4)

        V[k+1] = V[k] + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        U[k+1] = U[k] + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)

        # spike & reset
        if V[k+1] >= v_peak:
            V[k+1] = c
            U[k+1] += d
            spike_times.append(t[k+1])

    return V, U, np.array(spike_times)

# -------- Run experiments --------
I_list = [200.0, 300.0, 400.0, 500.0]  # pA
sims = [simulate(Iamp) for Iamp in I_list]

# -------- Plot (matching the earlier style) --------
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), sharex=True, sharey=True)
axes = axes.ravel()

# common y-limits to ease comparison
ymin, ymax = -90.0, 60.0

for ax, Iamp, (V, U, spikes) in zip(axes, I_list, sims):
    # shaded pulse region
    ax.axvspan(t_on, t_off, alpha=0.10, ec="none")
    # voltage trace
    ax.plot(t, V, lw=1.6)
    # spike markers (at v_peak line)
    if spikes.size:
        ax.plot(spikes, np.full_like(spikes, v_peak), "x", ms=5.5, mew=1.2, color="#d62728")
        rate = len(spikes) / ((t_off - t_on)/1000.0)  # Hz during pulse
        ax.set_title(f"I = {int(Iamp)} pA   —   {rate:.1f} Hz")
    else:
        ax.set_title(f"I = {int(Iamp)} pA   —   no spikes")

    # v_peak reference (light dashed)
    ax.axhline(v_peak, ls="--", lw=0.7, color="0.6")
    ax.grid(alpha=0.15, linestyle=":")

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, T)
    ax.set_ylabel("v (mV)")

axes[2].set_xlabel("Time (ms)")
axes[3].set_xlabel("Time (ms)")

fig.suptitle("Izhikevich (set b) — Voltage traces for square pulse (20–450 ms)", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore

# Save if needed:
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3bii.png", dpi=300)
# plt.show()
