#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 (a)(iii) — Izhikevich phase portraits (clean layout)
Legenda global no rodapé para evitar sobreposição com o título.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- Model params ----------
C      = 150.0
vr     = -75.0
vL     = -45.0
kpar   = 1.2
a      = 0.01
b      = 5.0
c      = -56.0
d      = 130.0
v_peak = 50.0

# ---------- Simulation ----------
dt = 0.001
T  = 500.0
t  = np.arange(0.0, T + dt, dt)
t_on, t_off = 20.0, 450.0

def I_square(t_ms, I_amp_pA):
    return I_amp_pA if (t_ms >= t_on and t_ms <= t_off) else 0.0

def dvdt(v, u, I):
    return (kpar*(v - vr)*(v - vL) - u + I) / C

def dudt(v, u):
    return a * (b*(v - vr) - u)

def nullclines(v_grid, I_amp):
    u_v = kpar*(v_grid - vr)*(v_grid - vL) + I_amp
    u_u = b*(v_grid - vr)
    return u_v, u_u

def fixed_points_for_I(I_amp):
    A = kpar
    B = -kpar*(vL + vr) - b
    Cq = kpar*vL*vr + b*vr + I_amp
    D  = B*B - 4*A*Cq
    fps = []
    if D >= 0:
        rt = np.sqrt(D)
        for v_star in ((-B + rt)/(2*A), (-B - rt)/(2*A)):
            u_star = b*(v_star - vr)
            fps.append((v_star, u_star))
    return fps

def classify(vs, us):
    dFdV = kpar*(2*vs - vL - vr) / C
    dFdU = -1.0 / C
    dGdV = a*b
    dGdU = -a
    J = np.array([[dFdV, dFdU],[dGdV, dGdU]], float)
    eig = np.linalg.eigvals(J)
    re, im = np.real(eig), np.imag(eig)
    if np.allclose(im, 0.0):
        if np.all(re < 0):             typ = "stable node"
        elif re.min() < 0 < re.max():  typ = "saddle"
        else:                          typ = "unstable node"
    else:
        typ = "stable focus" if np.max(re) < 0 else "unstable focus"
    return typ

def simulate_with_pulse(I_amp):
    v = np.empty_like(t); u = np.empty_like(t)
    v[0] = vr; u[0] = 0.0
    resets = []
    for k in range(len(t)-1):
        Ik  = I_square(t[k], I_amp)
        Ikh = I_square(t[k] + 0.5*dt, I_amp)
        Ik1 = I_square(t[k] + dt, I_amp)

        # RK4
        k1v = dvdt(v[k], u[k], Ik);        k1u = dudt(v[k], u[k])
        vk2 = v[k] + 0.5*dt*k1v;           uk2 = u[k] + 0.5*dt*k1u
        k2v = dvdt(vk2, uk2, Ikh);         k2u = dudt(vk2, uk2)
        vk3 = v[k] + 0.5*dt*k2v;           uk3 = u[k] + 0.5*dt*k2u
        k3v = dvdt(vk3, uk3, Ikh);         k3u = dudt(vk3, uk3)
        vk4 = v[k] + dt*k3v;               uk4 = u[k] + dt*k3u
        k4v = dvdt(vk4, uk4, Ik1);         k4u = dudt(vk4, uk4)

        v[k+1] = v[k] + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
        u[k+1] = u[k] + (dt/6)*(k1u + 2*k2u + 2*k3u + k4u)

        if v[k+1] >= v_peak:
            v[k+1] = c
            u[k+1] += d
            resets.append((v[k+1], u[k+1]))
    return v, u, np.array(resets)

# ---------- Plot (2x2) ----------
I_list = [300.0, 370.0, 500.0, 550.0]
fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
axes = axes.ravel()

COL_V = "#1f77b4"
COL_U = "#ff7f0e"
COL_TRAJ = "#2ca02c"
GRAY = "0.78"

v_grid = np.linspace(-90, -35, 1400)

for ax, Iamp in zip(axes, I_list):
    u_v0, u_u0 = nullclines(v_grid, 0.0)
    u_vI, u_uI = nullclines(v_grid, Iamp)
    v, u, resets = simulate_with_pulse(Iamp)

    v_all = np.concatenate([v_grid, v])
    u_all = np.concatenate([u_v0, u_u0, u_vI, u_uI, u])
    vmin, vmax = np.min(v_all), np.max(v_all)
    umin, umax = np.min(u_all), np.max(u_all)
    vpad = 0.05*(vmax - vmin + 1e-6); upad = 0.08*(umax - umin + 1e-6)
    ax.set_xlim(vmin - vpad, vmax + vpad)
    ax.set_ylim(umin - upad, umax + upad)

    Vq = np.linspace(*ax.get_xlim(), 24)# type:ignore
    Uq = np.linspace(*ax.get_ylim(), 24)# type:ignore
    VV, UU = np.meshgrid(Vq, Uq)
    dV = dvdt(VV, UU, Iamp); dU = dudt(VV, UU)
    mag = np.hypot(dV, dU) + 1e-12
    ax.quiver(VV, UU, dV/mag, dU/mag, angles="xy", scale_units="xy",
              scale=24, alpha=0.18, linewidth=0.5, color="0.25")

    ax.plot(v_grid, u_v0, color=GRAY, lw=1.0)
    ax.plot(v_grid, u_u0, color=GRAY, lw=1.0, ls="--")

    ax.plot(v_grid, u_vI, color=COL_V, lw=2.0, label="v-nullcline (I)")
    ax.plot(v_grid, u_uI, color=COL_U, lw=2.0, ls="--", label="u-nullcline")

    ax.plot(v, u, color=COL_TRAJ, lw=1.8, label="trajectory")
    if resets.size:
        ax.plot(resets[:,0], resets[:,1], "x", color="#d62728", ms=6, mew=1.4,
                label="reset points")

    fps = fixed_points_for_I(Iamp)
    if fps:
        for v_star, u_star in fps:
            ax.plot([v_star], [u_star], marker="o", mfc="white", mec="black", ms=6)
            typ = classify(v_star, u_star)
            ax.annotate(typ, xy=(v_star, u_star),
                        xytext=(v_star+2.0, u_star+0.06*(ax.get_ylim()[1]-ax.get_ylim()[0])),
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    else:
        ax.text(ax.get_xlim()[0]+0.02*(ax.get_xlim()[1]-ax.get_xlim()[0]),
                ax.get_ylim()[1]-0.08*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                "no fixed points (oscillatory)",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9))

    ax.set_title(f"I = {int(Iamp)} pA  (pulse 20–450 ms)")
    ax.set_xlabel("v (mV)")
    ax.set_ylabel("u (pA)")
    ax.grid(alpha=0.12, linestyle=":", linewidth=0.7)

# ---- Legend at bottom, outside axes (no overlap with title) ----
legend_handles = [
    Line2D([0],[0], color=COL_V, lw=2, label="v-nullcline (I)"),
    Line2D([0],[0], color=COL_U, lw=2, ls="--", label="u-nullcline"),
    Line2D([0],[0], color=GRAY, lw=1, label="nullclines (I=0)"),
    Line2D([0],[0], color=COL_TRAJ, lw=2, label="trajectory"),
    Line2D([0],[0], marker="x", color="#d62728", lw=0, ms=7, mew=1.4, label="reset points"),
    Line2D([0],[0], marker="o", color="k", mfc="white", lw=0, ms=6, label="fixed point"),
]

# Reservar espaço inferior para a legenda e superior para o título:
fig.subplots_adjust(top=0.90, bottom=0.14, left=0.06, right=0.98, wspace=0.18, hspace=0.22)

fig.legend(legend_handles, [h.get_label() for h in legend_handles],
           loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=6, frameon=False, fontsize=10)

fig.suptitle("Izhikevich — Phase portraits for each pulse amplitude", y=0.985)
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3aiii.png", dpi=300)
# plt.show()
