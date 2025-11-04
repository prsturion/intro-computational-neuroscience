# Q3 (c)(i) — Izhikevich variant with cubic u(v): phase portrait for I = 0
# Code in ENGLISH as requested.

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

# Initial conditions required by the statement (used for trajectory demo)
v0 = vr
u0 = 0.0

# --------------
# Model dynamics
# --------------
def dvdt(v, u, I=0.0):
    # C dv/dt = k(v - vr)(v - vL) - u + I
    return (kpar * (v - vr) * (v - vL) - u + I) / C

def dudt(v, u):
    # du/dt = a [ b(v - vr) + p (v - vr)^3 - u ]
    x = (v - vr)
    return a * (b * x + p * x**3 - u)

# Nullclines
def v_nullcline(v, I=0.0):
    return kpar * (v - vr) * (v - vL) + I

def u_nullcline(v):
    x = (v - vr)
    return b * x + p * x**3

# Jacobian and fixed-point classification
def jacobian(v, u):
    # ∂f/∂v = (k(2v - (vr + vL)))/C ; ∂f/∂u = -1/C
    df_dv = (kpar * (2.0 * v - (vr + vL))) / C
    df_du = -1.0 / C
    # ∂g/∂v = a [ b + 3p (v - vr)^2 ] ; ∂g/∂u = -a
    dg_dv = a * (b + 3.0 * p * (v - vr)**2)
    dg_du = -a
    return df_dv, df_du, dg_dv, dg_du

def classify_fixed_point(v, u):
    df_dv, df_du, dg_dv, dg_du = jacobian(v, u)
    tr = df_dv + dg_du
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
    return kind, tr, det, disc

# -------------
# Fixed points
# -------------
# Intersection u_v(v, I=0) = u_u(v)
# Analytically, letting x = v - vr:
# x * [-p x**2 + k x + (k*(vr - vL) - b)] = 0
# For given parameters, only x = 0 is real -> v* = vr, u* = 0
v_star = vr
u_star = 0.0
fp_kind, tr, det, disc = classify_fixed_point(v_star, u_star)
print(f"Fixed point at (v*, u*) = ({v_star:.2f} mV, {u_star:.2f} pA) → {fp_kind}")
print(f"trace = {tr:.3f}, det = {det:.3f}, discriminant = {disc:.3f}")

# ---------------------------
# Example trajectory (I = 0)
# ---------------------------
# Small perturbation from (v*, u*) just to show spiraling back to the fixed point.
dt = 0.02     # ms (coarser is enough for a clean curve)
T  = 200.0    # ms
t  = np.arange(0.0, T + dt, dt)

V = np.empty_like(t)
U = np.empty_like(t)
V[0] = v_star + 4.0   # slight voltage offset
U[0] = u_star + 200.0 # push u so we can see the spiral

for k in range(len(t) - 1):
    # RK4, I = 0 (no external current), v reset only if v reaches v_peak (won't happen here)
    k1v = dvdt(V[k], U[k], 0.0);     k1u = dudt(V[k], U[k])
    vk2 = V[k] + 0.5 * dt * k1v;     uk2 = U[k] + 0.5 * dt * k1u
    k2v = dvdt(vk2, uk2, 0.0);       k2u = dudt(vk2, uk2)
    vk3 = V[k] + 0.5 * dt * k2v;     uk3 = U[k] + 0.5 * dt * k2u
    k3v = dvdt(vk3, uk3, 0.0);       k3u = dudt(vk3, uk3)
    vk4 = V[k] + dt * k3v;           uk4 = U[k] + dt * k3u
    k4v = dvdt(vk4, uk4, 0.0);       k4u = dudt(vk4, uk4)

    V[k+1] = V[k] + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    U[k+1] = U[k] + (dt/6.0) * (k1u + 2*k2u + 2*k3u + k4u)

    # (With I=0 and these initials, v never reaches v_peak; no resets expected.)

# ----------------------
# Plot: phase portrait
# ----------------------
# Choose v-range around the interesting region and compute nullclines
vmin, vmax = -70.0, -20.0
v_grid = np.linspace(vmin, vmax, 800)
u_v = v_nullcline(v_grid, I=0.0)
u_u = u_nullcline(v_grid)

# Robust u-limits based on nullclines and trajectory
u_all = np.concatenate([u_v, u_u, U])
u_low = np.percentile(u_all, 1) - 30.0
u_high = np.percentile(u_all, 99) + 30.0

fig, ax = plt.subplots(figsize=(9, 6))

# light vector field (sparse) just for orientation
VV, UU = np.meshgrid(np.linspace(vmin, vmax, 24), np.linspace(u_low, u_high, 24))
DDV = dvdt(VV, UU, 0.0)
DDU = dudt(VV, UU)
norm = np.hypot(DDV, DDU) + 1e-9
ax.quiver(VV, UU, DDV/norm, DDU/norm, pivot='mid', alpha=0.15, linewidth=0.5)

# nullclines
ax.plot(v_grid, u_v, label="v-nullcline (I=0)", lw=2.2, color="#1f77b4")
ax.plot(v_grid, u_u, label="u-nullcline", lw=2.2, ls="--", color="#ff7f0e")

# trajectory
ax.plot(V, U, color="#2ca02c", lw=2.0, label="sample trajectory (I=0)")

# fixed point + label
ax.plot([v_star], [u_star], 'o', ms=7, color='k')
ax.annotate(fp_kind, xy=(v_star, u_star), xytext=(v_star+2.5, u_star+80),
            arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=10)

ax.set_xlim(vmin, vmax)
ax.set_ylim(u_low, u_high) # type: ignore
ax.set_xlabel("v (mV)")
ax.set_ylabel("u (pA)")
ax.set_title("Izhikevich variant (cubic u) — Phase portrait for I = 0")
ax.grid(alpha=0.15, linestyle=":")
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3ci.png", dpi=300)
# plt.show()
