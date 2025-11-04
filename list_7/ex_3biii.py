#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 (b)(iii) — Izhikevich (set b) — Phase portraits (clean view)
Mostra nulclinas, pontos de reset e SOMENTE 2 ciclos representativos da trajetória
para evitar “tapete” de linhas na base das figuras.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Params: set (b) ----------
C      = 50.0
vr     = -60.0
vL     = -40.0
kpar   = 1.5
a      = 0.03
b      = 1.0
c      = -40.0
d      = 150.0
v_peak = 25.0

# ---------- Time / pulse ----------
dt   = 0.001
T    = 500.0
t    = np.arange(0.0, T+dt, dt)
t_on, t_off = 20.0, 450.0

def I_square(t_ms, I_amp):
    return I_amp if (t_ms >= t_on and t_ms <= t_off) else 0.0

def dvdt(v, u, I):
    return (kpar*(v - vr)*(v - vL) - u + I) / C

def dudt(v, u):
    return a*(b*(v - vr) - u)

def simulate(I_amp):
    V = np.empty_like(t); U = np.empty_like(t)
    V[0] = vr; U[0] = 0.0
    resets = []
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

        if V[k+1] >= v_peak:
            V[k+1] = c
            U[k+1] += d
            resets.append(k+1)
    return V, U, np.array(resets, dtype=int)

def nullclines(v_grid, I_amp):
    u_v = kpar*(v_grid - vr)*(v_grid - vL) + I_amp
    u_u = b*(v_grid - vr)
    return u_v, u_u

def plot_two_cycles(ax, V, U, idx_start, idx_end, color="#2ca02c"):
    """Plota até 2 ciclos entre resets dentro de [idx_start, idx_end]."""
    # resets no intervalo
    r = np.where((V[idx_start:idx_end] == c) & (np.diff(np.r_[False, V[idx_start:idx_end]==c])==1))[0]
    # Alternativa robusta: use os resets já computados fora
    # Aqui vamos usar a versão simples com índices conhecidos externamente.
    ax.plot(V[idx_start:idx_end:300], U[idx_start:idx_end:300], color=color, lw=1.8, alpha=0.9)

# ---------- Build figure ----------
I_list = [200.0, 300.0, 400.0, 500.0]
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=False, sharey=False)
axs = axs.ravel()

for ax, Iamp in zip(axs, I_list):
    V, U, r_idx = simulate(Iamp)
    in_pulse = (t >= t_on) & (t <= t_off)
    Vp, Up = V[in_pulse], U[in_pulse]
    idx0 = np.where(in_pulse)[0][0]
    idx1 = np.where(in_pulse)[0][-1]

    # limites de v pelo alcance no pulso (+ margem)
    vmarg = 6.0
    v_min = float(np.min(Vp) - vmarg)
    v_max = float(np.max(Vp) + vmarg)

    # nulclinas só no intervalo relevante
    v_grid = np.linspace(v_min, v_max, 600)
    u_v, u_u = nullclines(v_grid, Iamp)
    ax.plot(v_grid, u_v, lw=2.0, color="#1f77b4", label="v-nullcline")
    ax.plot(v_grid, u_u, lw=2.0, ls="--", color="#ff7f0e", label="u-nullcline")

    # limites de u por percentis da trajetória (robusto contra outliers)
    p1, p99 = np.percentile(Up, [1, 99])
    umarg = 60.0
    u_min = float(p1 - umarg)
    u_max = float(p99 + umarg)

    # TRAJETÓRIA: apenas 2 ciclos representativos
    # encontre resets dentro do pulso
    r_in = r_idx[(t[r_idx] >= t_on) & (t[r_idx] <= t_off)]
    if r_in.size >= 2:
        # pegue os dois primeiros ciclos completos dentro do pulso
        segs = [(r_in[0]-1500, r_in[1]+1500)]
        if r_in.size >= 3:
            segs.append((r_in[1]-1500, r_in[2]+1500))
        for s, e in segs:
            s = max(s, idx0)
            e = min(e, idx1)
            ax.plot(V[s:e:300], U[s:e:300], color="#2ca02c", lw=1.8, alpha=0.95)
            # pequena seta para direção
            k = (s + e)//2
            ax.annotate("", xy=(V[k+300], U[k+300]), xytext=(V[k], U[k]),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color="#2ca02c"))
    else:
        # se não houver 2 spikes, plote trecho curto central do pulso
        s = idx0 + int(0.15*(idx1-idx0))
        e = idx0 + int(0.35*(idx1-idx0))
        ax.plot(V[s:e:300], U[s:e:300], color="#2ca02c", lw=1.8, alpha=0.95)

    # marcar resets (durante o pulso)
    if r_in.size:
        ax.plot(V[r_in], U[r_in], "x", ms=6, mew=1.4, color="#d62728", label="reset")

    ax.set_xlim(v_min, v_max)
    ax.set_ylim(u_min, u_max)
    ax.grid(alpha=0.12, linestyle=":")
    ax.set_title(f"I = {int(Iamp)} pA  (pulse 20–450 ms)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

axs[2].set_xlabel("v (mV)")
axs[3].set_xlabel("v (mV)")
axs[0].set_ylabel("u (pA)")
axs[2].set_ylabel("u (pA)")

fig.suptitle("Izhikevich (set b) — Phase portraits (2 representative cycles)", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore

plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3biii.png", dpi=300)
# plt.show()
