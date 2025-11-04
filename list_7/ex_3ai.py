#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Izhikevich model (quadratic form) — Q3 (a)(i)
Units: v [mV], u [pA], t [ms]; C [pF], k [pA/mV^2], b [nS=pA/mV], a [1/ms]
I = 0: phase portrait with auto axis limits, nullclines, fixed points, stability,
vector field and a small subthreshold trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
C      = 150.0      # pF
vr     = -75.0      # mV
vL     = -45.0      # mV
kpar   = 1.2        # pA/mV^2
a      = 0.01       # 1/ms
b      = 5.0        # nS = pA/mV
c      = -56.0      # mV (unused here)
d      = 130.0      # pA (unused here)
v_peak = 50.0       # mV (unused here)
Iext   = 0.0        # pA

# ---------------- Dynamics -------------------
def dvdt(v, u, I=0.0):
    return (kpar*(v - vr)*(v - vL) - u + I) / C  # mV/ms

def dudt(v, u):
    return a * ( b*(v - vr) - u )                # pA/ms

def nullclines_I0(v_grid):
    u_v = kpar*(v_grid - vr)*(v_grid - vL)       # V-nullcline (dv/dt=0)
    u_u = b*(v_grid - vr)                        # U-nullcline (du/dt=0)
    return u_v, u_u

def fixed_points_I0():
    v1 = vr
    v2 = vL + b/kpar
    u1 = b*(v1 - vr)         # 0 pA
    u2 = b*(v2 - vr)
    return [(v1, u1), (v2, u2)]

def classify(vs, us):
    dFdV = kpar*(2*vs - vL - vr)/C
    dFdU = -1.0/C
    dGdV = a*b
    dGdU = -a
    J = np.array([[dFdV, dFdU],
                  [dGdV, dGdU]], float)
    eig = np.linalg.eigvals(J)
    re, im = np.real(eig), np.imag(eig)
    if np.allclose(im, 0.0):
        if np.all(re < 0):             typ = "stable node"
        elif re.min() < 0 < re.max():  typ = "saddle"
        else:                          typ = "unstable node"
    else:
        typ = "stable focus" if np.max(re) < 0 else "unstable focus"
    return typ, eig

def sample_trajectory(I=0.0):
    dt, T = 0.001, 20.0   # ms
    t = np.arange(0.0, T+dt, dt)
    v = np.empty_like(t); u = np.empty_like(t)
    v[0] = vr + 2.0   # pequena perturbação
    u[0] = 0.0
    for k in range(len(t)-1):
        v[k+1] = v[k] + dvdt(v[k], u[k], I)*dt
        u[k+1] = u[k] + dudt(v[k], u[k])*dt
    return t, v, u

# ---------------- Compute --------------------
vmin_raw, vmax_raw = -90.0, -35.0
v_grid = np.linspace(vmin_raw, vmax_raw, 1200)
u_v, u_u = nullclines_I0(v_grid)
fps = fixed_points_I0()
fp_info = [(v, u, *classify(v, u)) for (v, u) in fps]
t, v_tr, u_tr = sample_trajectory(I=0.0)

# Axis limits chosen automatically to include everything with margin
all_u = np.concatenate([u_v, u_u, u_tr, np.array([fp[1] for fp in fps])])
all_v = np.concatenate([v_grid, v_tr, np.array([fp[0] for fp in fps])])
u_min, u_max = float(np.min(all_u)), float(np.max(all_u))
v_min, v_max = float(np.min(all_v)), float(np.max(all_v))
# add margins
u_pad = 0.08*(u_max - u_min + 1e-6)
v_pad = 0.05*(v_max - v_min + 1e-6)
u_min -= u_pad; u_max += u_pad
v_min -= v_pad; v_max += v_pad

# ---------------- Plot -----------------------
plt.figure(figsize=(9.5, 7))
ax = plt.gca()

# Vector field (normalized)
Vq = np.linspace(v_min, v_max, 25)
Uq = np.linspace(u_min, u_max, 25)
VV, UU = np.meshgrid(Vq, Uq)
dV = dvdt(VV, UU, Iext); dU = dudt(VV, UU)
mag = np.hypot(dV, dU) + 1e-20
ax.quiver(VV, UU, dV/mag, dU/mag, angles="xy", scale_units="xy",
          scale=22, alpha=0.25, linewidth=0.5)

# Nullclines and trajectory
ax.plot(v_grid, u_v, label="v-nullcline (I = 0)")
ax.plot(v_grid, u_u, label="u-nullcline")
ax.plot(v_tr, u_tr, lw=1.8, label="sample trajectory")

# Fixed points with on-plot annotations
for v_star, u_star, typ, eig in fp_info:
    ax.plot([v_star], [u_star], marker="o", mec="k", mfc="w", ms=7, zorder=3)
    ax.annotate(typ, xy=(v_star, u_star), xytext=(v_star+2, u_star+0.07*(u_max-u_min)),
                textcoords="data", fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=0.8))

ax.set_xlim(v_min, v_max)
ax.set_ylim(u_min, u_max)
ax.set_xlabel("v (mV)")
ax.set_ylabel("u (pA)")
ax.set_title("Izhikevich model — Phase portrait for I = 0")
ax.legend(loc="best", fontsize=10, frameon=False)
plt.tight_layout()
# plt.show()
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3ai.png", dpi=300, bbox_inches="tight")

# Console summary
print("Fixed points for I = 0:")
for v_star, u_star, typ, eig in fp_info:
    print(f"  v* = {v_star:.2f} mV, u* = {u_star:.2f} pA  -> {typ}; eigenvalues = {eig}")
