# Q3 (c)(iii) — Izhikevich variant with cubic u(v):
# Phase portraits for each pulse amplitude (same cases as in (ii)),
# showing v- and u-nullclines, fixed points with stability, trajectory,
# and reset points. Code in ENGLISH as requested.

import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Parameters (item c)
# --------------------
C      = 25.0     # pF
vr     = -50.0    # mV
vL     = -30.0    # mV
kpar   = 1.0
a      = 0.5
b      = 25.0
p      = 0.009
c      = -40.0    # mV (reset for v only)
v_peak = 10.0     # mV (spike threshold for v reset)

# Initial conditions
v0 = vr
u0 = 0.0

# Time and pulse
dt   = 0.02                  # ms
Tend = 200.0                 # ms
t    = np.arange(0.0, Tend + dt, dt)
t_on, t_off = 20.0, 180.0    # ms (square pulse window)
on_idx  = int(np.floor(t_on/dt))
off_idx = int(np.floor(t_off/dt))

# -----------------
# Model definitions
# -----------------
def dvdt(v, u, I):
    # C dv/dt = k(v - vr)(v - vL) - u + I
    return (kpar * (v - vr) * (v - vL) - u + I) / C

def dudt(v, u):
    # du/dt = a [ b(v - vr) + p (v - vr)^3 - u ]
    x = (v - vr)
    return a * (b * x + p * x**3 - u)

def v_nullcline(v, I=0.0):
    # From dv/dt = 0 ⇒ u = k(v - vr)(v - vL) + I
    return kpar * (v - vr) * (v - vL) + I

def u_nullcline(v):
    # From du/dt = 0 ⇒ u = b(v - vr) + p (v - vr)^3
    x = (v - vr)
    return b * x + p * x**3

def square_pulse(current):
    I = np.zeros_like(t)
    I[on_idx:off_idx] = current
    return I

def simulate(current):
    I_tr = square_pulse(current)
    V = np.empty_like(t); U = np.empty_like(t)
    V[0] = v0; U[0] = u0
    reset_vs, reset_us = [], []

    for k in range(len(t) - 1):
        I_k = I_tr[k]

        # --- RK4 integration of continuous dynamics ---
        k1v = dvdt(V[k],             U[k],             I_k); k1u = dudt(V[k],             U[k])
        k2v = dvdt(V[k]+0.5*dt*k1v,  U[k]+0.5*dt*k1u,  I_k); k2u = dudt(V[k]+0.5*dt*k1v,  U[k]+0.5*dt*k1u)
        k3v = dvdt(V[k]+0.5*dt*k2v,  U[k]+0.5*dt*k2u,  I_k); k3u = dudt(V[k]+0.5*dt*k2v,  U[k]+0.5*dt*k2u)
        k4v = dvdt(V[k]+    dt*k3v,  U[k]+    dt*k3u,  I_k); k4u = dudt(V[k]+    dt*k3v,  U[k]+    dt*k3u)

        v_next = V[k] + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        u_next = U[k] + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)

        # --- spike/reset rule (variant c): ONLY v resets, u is not reset ---
        if v_next >= v_peak:
            # mark the point that hit v_peak
            reset_vs.append(v_peak)
            reset_us.append(U[k])  # u at the spike moment
            # clip visual spike and reset v
            V[k] = v_peak
            v_next = c

        V[k+1] = v_next
        U[k+1] = u_next

    return I_tr, V, U, np.array(reset_vs), np.array(reset_us)

# Jacobian and classification of fixed points
def jacobian(v, u):
    df_dv = (kpar * (2.0 * v - (vr + vL))) / C
    df_du = -1.0 / C
    dg_dv = a * (b + 3.0 * p * (v - vr)**2)
    dg_du = -a
    return df_dv, df_du, dg_dv, dg_du

def classify_fixed_point(v, u):
    df_dv, df_du, dg_dv, dg_du = jacobian(v, u)
    tr  = df_dv + dg_du
    det = df_dv * dg_du - df_du * dg_dv
    disc = tr**2 - 4.0 * det
    if det < 0:
        kind = "saddle"
    else:
        if tr < 0:
            kind = "stable focus" if disc < 0 else "stable node"
        elif tr > 0:
            kind = "unstable focus" if disc < 0 else "unstable node"
        else:
            kind = "center / borderline"
    return kind

def fixed_points_for_I(Iamp):
    # Solve u_v(v, I) = u_u(v)
    # Let x = v - vr:
    # p x^3 - k x^2 + (b - k*(vr - vL)) x + I = 0
    A3 = p
    A2 = -kpar
    A1 = (b - kpar*(vr - vL))
    A0 = Iamp
    roots = np.roots([A3, A2, A1, A0])

    v_list, u_list, kinds = [], [], []
    for r in roots:
        if abs(r.imag) < 1e-6:
            x = r.real
            v = vr + x
            u = u_nullcline(v)  # equals v_nullcline(v, Iamp) by construction
            v_list.append(v); u_list.append(u)
            kinds.append(classify_fixed_point(v, u))
    return np.array(v_list), np.array(u_list), kinds

# -----------------------
# Simulate the 5 currents
# -----------------------
currents = [200.0, 400.0, 470.0, 500.0, 600.0]  # pA
sims = [simulate(I) for I in currents]

# -------------------------
# Plot: phase portraits (5)
# -------------------------
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
axes = axes.flatten()

for i, Iamp in enumerate(currents):
    I_tr, V, U, rV, rU = sims[i]
    ax = axes[i]

    # Choose a focused window around the trajectory to keep the picture clean
    vmin = np.percentile(V, 1) - 10.0
    vmax = np.percentile(V, 99) + 10.0
    v_grid = np.linspace(vmin, vmax, 600)

    # Nullclines for this constant I (during pulse)
    uv = v_nullcline(v_grid, I=Iamp)
    uu = u_nullcline(v_grid)

    # Vector field (light)
    VV, UU = np.meshgrid(np.linspace(vmin, vmax, 28),
                         np.linspace(np.percentile(U, 1)-30, np.percentile(U, 99)+30, 28))
    FF = dvdt(VV, UU, Iamp); GG = dudt(VV, UU)
    N = np.hypot(FF, GG) + 1e-9
    ax.quiver(VV, UU, FF/N, GG/N, pivot='mid', alpha=0.12, linewidth=0.5)

    # Plot nullclines
    ax.plot(v_grid, uv, lw=2.0, label="v-nullcline (I)", color="#1f77b4")
    ax.plot(v_grid, uu, lw=2.0, ls="--", label="u-nullcline", color="#ff7f0e")

    # Plot only the portion of trajectory within the pulse window to avoid clutter
    traj_mask = np.zeros_like(t, dtype=bool)
    # Keep two representative cycles near the middle of the pulse
    mid_a = int((on_idx*2 + off_idx)//3)
    mid_b = int((on_idx + off_idx*2)//3)
    traj_mask[mid_a:mid_b] = True

    ax.plot(V[traj_mask], U[traj_mask], color="#2ca02c", lw=2.0, label="trajectory (pulse)")

    # Reset points that fell inside the window
    if rV.size:
        # Find spike indices by matching v_peak moments near the selected window
        spike_indices = np.where((t >= t[mid_a]) & (t <= t[mid_b]))[0]
        # Plot all recorded resets (they already correspond to spikes)
        ax.scatter(rV, rU, marker='x', s=50, color='crimson', label="reset")

    # Fixed points for constant I = Iamp
    vfp, ufp, kinds = fixed_points_for_I(Iamp)
    if len(vfp) == 0:
        ax.text(0.03, 0.94, "no fixed points (oscillatory)", transform=ax.transAxes,
                fontsize=10)
    else:
        ax.scatter(vfp, ufp, facecolors='none', edgecolors='k', s=60, label="fixed point")
        # annotate first two (if any)
        for j, (vv, uu_, kd) in enumerate(zip(vfp, ufp, kinds)):
            if j < 2:
                ax.annotate(kd, xy=(vv, uu_), xytext=(vv+2.5, uu_+20),
                            arrowprops=dict(arrowstyle='->', lw=0.9), fontsize=9)

    # Axes cosmetics
    ax.set_title(f"I = {int(Iamp)} pA  (pulse {int(t_on)}–{int(t_off)} ms)")
    ax.set_xlabel("v (mV)")
    ax.set_ylabel("u (pA)")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

# Hide the unused 6th axis
axes[-1].axis("off")

plt.suptitle("Izhikevich variant (cubic u) — Phase portraits for each pulse amplitude", y=0.98)
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3ciii.png", dpi=300)
# plt.show()
