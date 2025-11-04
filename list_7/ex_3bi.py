#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 (b)(i) — Izhikevich (conjunto b) — Retrato de fase para I=0
C = 50 pF, v_r = -60 mV, v_L = -40 mV, k = 1.5, a = 0.03, b = 1, c = -40 mV, d = 150 pA, v_peak = 25 mV
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
C      = 50.0
vr     = -60.0
vL     = -40.0
kpar   = 1.5
a      = 0.03
b      = 1.0
c      = -40.0
d      = 150.0
v_peak = 25.0
I_const = 0.0

# ---------------- Dinâmica ----------------
def dvdt(v, u, I=I_const):
    return (kpar*(v - vr)*(v - vL) - u + I) / C

def dudt(v, u):
    return a*(b*(v - vr) - u)

def nullclines(v_grid, I=I_const):
    u_v = kpar*(v_grid - vr)*(v_grid - vL) + I
    u_u = b*(v_grid - vr)
    return u_v, u_u

def fixed_points_for_I(I=I_const):
    # k v^2 + v*(-k(vL+vr) - b) + (k vL vr + b vr + I) = 0
    A = kpar
    B = -kpar*(vL + vr) - b
    Cq = kpar*vL*vr + b*vr + I
    D  = B*B - 4*A*Cq
    fps = []
    if D >= 0:
        rt = np.sqrt(D)
        for v_star in ((-B + rt)/(2*A), (-B - rt)/(2*A)):
            u_star = b*(v_star - vr)
            fps.append((float(v_star), float(u_star)))
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

# ---------------- Trajetória curta (I=0) ----------------
def short_trajectory_from_stable(v_box=(-90, -30), u_pad=60.0):
    # Começa próximo ao nó estável (menor v dos dois pontos fixos).
    fps = fixed_points_for_I(I_const)
    if not fps:
        v0, u0 = vr, 0.0
    else:
        v0, u0 = min(fps, key=lambda p: p[0])  # nó estável fica à esquerda

    dt = 0.02  # ms (curta e estável)
    T  = 250.0
    t  = np.arange(0.0, T+dt, dt)
    v  = np.empty_like(t); u = np.empty_like(t)
    v[0] = v0 + 0.5   # pequena perturbação
    u[0] = u0 - 0.5

    # caixa de segurança para evitar “fugas”
    vmin_box, vmax_box = v_box
    umin_box = min(nullclines(np.array([vmin_box]), 0.0)[0][0],
                   nullclines(np.array([vmin_box]), 0.0)[1][0]) - u_pad
    umax_box = max(nullclines(np.array([vmax_box]), 0.0)[0][0],
                   nullclines(np.array([vmax_box]), 0.0)[1][0]) + u_pad

    for k in range(len(t)-1):
        k1v = dvdt(v[k], u[k]);              k1u = dudt(v[k], u[k])
        k2v = dvdt(v[k] + 0.5*dt*k1v, u[k] + 0.5*dt*k1u)
        k2u = dudt(v[k] + 0.5*dt*k1v, u[k] + 0.5*dt*k1u)
        k3v = dvdt(v[k] + 0.5*dt*k2v, u[k] + 0.5*dt*k2u)
        k3u = dudt(v[k] + 0.5*dt*k2v, u[k] + 0.5*dt*k2u)
        k4v = dvdt(v[k] + dt*k3v,  u[k] + dt*k3u)
        k4u = dudt(v[k] + dt*k3v,  u[k] + dt*k3u)
        v[k+1] = v[k] + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        u[k+1] = u[k] + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)

        # corta se sair da caixa
        if (v[k+1] < vmin_box or v[k+1] > vmax_box or
            u[k+1] < umin_box or u[k+1] > umax_box):
            v = v[:k+2]; u = u[:k+2]
            break
    return v, u

# ---------------- Plot ----------------
v_grid = np.linspace(-90.0, -30.0, 1200)
u_v, u_u = nullclines(v_grid, I_const)

fig, ax = plt.subplots(figsize=(7.6, 6.0))

# Campo vetorial leve
Vq = np.linspace(-90, -30, 24)
Uq = np.linspace(min(u_v.min(), u_u.min()) - 60,
                 max(u_v.max(), u_u.max()) + 60, 24)
VV, UU = np.meshgrid(Vq, Uq)
dV = dvdt(VV, UU, I_const); dU = dudt(VV, UU)
mag = np.hypot(dV, dU) + 1e-12
ax.quiver(VV, UU, dV/mag, dU/mag, angles="xy", scale_units="xy",
          scale=22, alpha=0.18, color="0.25", linewidth=0.5)

# Nulclinas
ax.plot(v_grid, u_v, color="#1f77b4", lw=2.2, label="v-nullcline (I=0)")
ax.plot(v_grid, u_u, color="#ff7f0e", lw=2.2, ls="--", label="u-nullcline")

# Pontos fixos e classificação
fps = fixed_points_for_I(I_const)
for v_star, u_star in fps:
    typ = classify(v_star, u_star)
    ax.plot([v_star], [u_star], marker="o", mfc="white", mec="k", ms=6, zorder=5)
    ax.annotate(typ, xy=(v_star, u_star),
                xytext=(v_star+2.0, u_star+40.0),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.95),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))

# Trajetória curta a partir do nó estável
v_traj, u_traj = short_trajectory_from_stable()
ax.plot(v_traj, u_traj, color="#2ca02c", lw=1.8, label="trajectory (I=0)")

# Eixos e estilo
ax.set_xlim(-90, -30)
ax.set_ylim(min(u_v.min(), u_u.min()) - 60, max(u_v.max(), u_u.max()) + 60)
ax.set_title("Izhikevich — Phase portrait for I = 0")
ax.set_xlabel("v (mV)")
ax.set_ylabel("u (pA)")
ax.grid(alpha=0.12, linestyle=":")
ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3bi.png", dpi=300)
# plt.show()
