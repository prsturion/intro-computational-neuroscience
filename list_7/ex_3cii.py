# Q3 (c)(ii) — Izhikevich variant with cubic u(v):
# Square-pulse simulations for 5 amplitudes; plot v(t) in 5 panels.
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

# Initial conditions
v0 = vr
u0 = 0.0

# Time and pulse
dt   = 0.02                  # ms
Tend = 200.0                 # ms
t    = np.arange(0.0, Tend + dt, dt)
t_on, t_off = 20.0, 180.0    # ms (square pulse window)

# Model equations
def dvdt(v, u, I):
    # C dv/dt = k(v - vr)(v - vL) - u + I
    return (kpar * (v - vr) * (v - vL) - u + I) / C

def dudt(v, u):
    # du/dt = a [ b(v - vr) + p (v - vr)^3 - u ]
    x = (v - vr)
    return a * (b * x + p * x**3 - u)

def square_pulse(current):
    # piecewise-constant current over time grid t
    I = np.zeros_like(t)
    on  = int(np.floor(t_on/dt))
    off = int(np.floor(t_off/dt))
    I[on:off] = current
    return I

def simulate(current):
    I_tr = square_pulse(current)
    V = np.empty_like(t); U = np.empty_like(t)
    V[0] = v0; U[0] = u0
    for k in range(len(t) - 1):
        I_k = I_tr[k]

        # --- RK4 step (continuous dynamics) ---
        k1v = dvdt(V[k],         U[k],         I_k); k1u = dudt(V[k],         U[k])
        k2v = dvdt(V[k]+0.5*dt*k1v, U[k]+0.5*dt*k1u, I_k); k2u = dudt(V[k]+0.5*dt*k1v, U[k]+0.5*dt*k1u)
        k3v = dvdt(V[k]+0.5*dt*k2v, U[k]+0.5*dt*k2u, I_k); k3u = dudt(V[k]+0.5*dt*k2v, U[k]+0.5*dt*k2u)
        k4v = dvdt(V[k]+    dt*k3v, U[k]+    dt*k3u, I_k); k4u = dudt(V[k]+    dt*k3v, U[k]+    dt*k3u)

        v_next = V[k] + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        u_next = U[k] + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)

        # --- spike/reset rule (variant c): only v resets; u is NOT reset ---
        if v_next >= v_peak:
            # clip the current sample to v_peak so the spike is visible
            V[k] = v_peak
            v_next = c  # reset v
            # u_next unchanged

        V[k+1] = v_next
        U[k+1] = u_next

    return I_tr, V, U

# Currents to test
currents = [200.0, 400.0, 470.0, 500.0, 600.0]  # pA

# Run all sims to set a common y-limit
results = [simulate(Iamp) for Iamp in currents]
v_min = min(V.min() for (_, V, _) in results)
v_max = max(V.max() for (_, V, _) in results)
# Provide comfortable bounds
v_lo  = min(-90.0, v_min - 5.0)
v_hi  = max( 60.0, v_max + 5.0)

# -----------------------
# Plot: 5 voltage panels
# -----------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, Iamp in enumerate(currents):
    I_tr, V, U = results[idx]
    ax = axes[idx]
    ax.plot(t, V, lw=1.8)
    # Shade pulse window
    ax.axvspan(t_on, t_off, alpha=0.10)
    ax.set_ylim([v_lo, v_hi])
    ax.set_xlim([0, Tend])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("v (mV)")
    ax.set_title(f"I = {int(Iamp)} pA  (pulse {int(t_on)}–{int(t_off)} ms)")
    ax.grid(alpha=0.2, linestyle=":")

# Hide the unused 6th axis cleanly
axes[-1].axis("off")

plt.suptitle("Izhikevich variant (cubic u) — Voltage traces for square pulse (20–180 ms)", y=0.97)
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_7/figures/ex_3cii.png", dpi=300)
# plt.show()
