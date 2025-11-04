#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdEx phase-plane (improved):
- Ensures the pulse elicits a spike (so w jumps by b)
- Zoomed phase-plane window to avoid exponential blow-up flattening the view
- Adds vector field (quiver), nullclines, fixed points + stability labels
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Base constants (SI) ----------
E_L     = -70e-3   # V
R       = 500e6    # Ohm
V_T     = -50e-3   # V
Delta_T =  2e-3    # V
V_peak  = 20e-3    # V

DT       = 1e-5    # s
EXP_CLIP = 40.0
DV_LIMIT = 5e-3    # V per step

# ---------- Choose a pattern ----------
PARAMS = dict(
    name="tonic",
    tau_m_ms=20.0,   # ms
    a_nS=0.0,        # nS (tip: set to 1.0 to see w move even subthreshold)
    tau_w_ms=30.0,   # ms
    b_pA=60.0,       # pA
    Vreset_mV=-55.0  # mV
)

# Pulse chosen to actually spike
PULSE = dict(t_on=0.050, t_off=0.055, amp_pA=300.0)
T_TOTAL = 0.20

def safe_exp(x):
    return np.exp(np.minimum(x, EXP_CLIP))

def rhs(v, w, I, tau_m, tau_w, a):
    dVdt = ((E_L - v) + Delta_T * safe_exp((v - V_T)/Delta_T) + R*(I - w)) / tau_m
    dwdt = (a*(v - E_L) - w) / tau_w
    return dVdt, dwdt

def simulate_with_pulse(params, pulse, T=T_TOTAL, dt=DT):
    tau_m = params["tau_m_ms"]*1e-3
    tau_w = params["tau_w_ms"]*1e-3
    a     = params["a_nS"]*1e-9
    b     = params["b_pA"]*1e-12
    Vrst  = params["Vreset_mV"]*1e-3

    t = np.arange(0.0, T+dt, dt)
    V = np.empty_like(t); w = np.empty_like(t)
    V[0] = E_L; w[0] = 0.0

    def I_of_t(tt):
        return (pulse["amp_pA"]*1e-12) if (pulse["t_on"] <= tt < pulse["t_off"]) else 0.0

    spiked = False
    for k in range(len(t)-1):
        v, u = V[k], w[k]
        if v >= V_peak:
            v = Vrst
            u = u + b
            spiked = True

        I = I_of_t(t[k])
        dVdt, dwdt = rhs(v, u, I, tau_m, tau_w, a)

        dv = np.clip(dVdt*dt, -DV_LIMIT, DV_LIMIT)
        du = dwdt*dt

        v_next = v + dv
        u_next = u + du

        if v_next >= V_peak:
            v_next = Vrst
            u_next = u_next + b
            spiked = True

        V[k+1] = v_next
        w[k+1] = u_next

    return t, V, w, spiked

def nullclines_I0(params, V_grid):
    a = params["a_nS"]*1e-9
    w_V = (E_L - V_grid)/R + (Delta_T/R)*safe_exp((V_grid - V_T)/Delta_T)   # dV/dt=0 (I=0)
    w_w = a*(V_grid - E_L)                                                  # dw/dt=0
    return w_V, w_w

def find_fixed_points(params, V_min=-0.085, V_max=-0.035, n_samples=2000):
    a = params["a_nS"]*1e-9
    Vs = np.linspace(V_min, V_max, n_samples)
    def F(V):
        return (E_L - V)/R + (Delta_T/R)*safe_exp((V - V_T)/Delta_T) - a*(V - E_L)
    vals = F(Vs)
    roots = []
    for i in range(len(Vs)-1):
        y1, y2 = vals[i], vals[i+1]
        if np.isnan(y1) or np.isnan(y2): continue
        if y1 == 0.0:
            Vstar = Vs[i]
        elif y1*y2 < 0.0:
            aV, bV = Vs[i], Vs[i+1]; fa, fb = y1, y2
            for _ in range(60):
                m = 0.5*(aV+bV); fm = F(m)
                if fa*fm <= 0: bV, fb = m, fm
                else:          aV, fa = m, fm
            Vstar = 0.5*(aV+bV)
        else:
            continue
        wstar = a*(Vstar - E_L)
        if not roots or abs(Vstar - roots[-1][0]) > 1e-6:
            roots.append((Vstar, wstar))
    return roots

def classify_fixed_point(Vs, ws, params):
    tau_m = params["tau_m_ms"]*1e-3
    tau_w = params["tau_w_ms"]*1e-3
    a     = params["a_nS"]*1e-9
    exp_term = safe_exp((Vs - V_T)/Delta_T)
    dFdV = (-1.0 + exp_term)/tau_m
    dFdW = (-R)/tau_m
    dGdV = (a)/tau_w
    dGdW = (-1.0)/tau_w
    J = np.array([[dFdV, dFdW],[dGdV, dGdW]], float)
    eig = np.linalg.eigvals(J)
    re, im = np.real(eig), np.imag(eig)
    if np.all(im == 0):
        typ = "stable node" if np.all(re < 0) else ("saddle" if np.any(re > 0) and np.any(re < 0) else "unstable node")
    else:
        typ = "stable focus" if np.max(re) < 0 else "unstable focus"
    return typ, eig

if __name__ == "__main__":
    # ---- compute nullclines (I=0) in a zoomed window ----
    Vmin, Vmax = -0.080, -0.040  # [-80, -40] mV
    V_grid = np.linspace(Vmin, Vmax, 800)
    wV, ww = nullclines_I0(PARAMS, V_grid)

    # fixed points
    fps = find_fixed_points(PARAMS, V_min=Vmin, V_max=Vmax)
    fp_info = [(*fp, *classify_fixed_point(fp[0], fp[1], PARAMS)) for fp in fps]

    # simulate pulse
    t, V, w, spiked = simulate_with_pulse(PARAMS, PULSE)

    # vector field (downsampled grid)
    Vq = np.linspace(Vmin, Vmax, 20)
    wq = np.linspace(-40e-12, 80e-12, 20)  # [-40,80] pA
    VV, WW = np.meshgrid(Vq, wq)
    tau_m = PARAMS["tau_m_ms"]*1e-3 # type: ignore
    tau_w = PARAMS["tau_w_ms"]*1e-3 # type: ignore
    a     = PARAMS["a_nS"]*1e-9 # type: ignore
    dVdt, dwdt = rhs(VV, WW, 0.0, tau_m, tau_w, a)
    # normalize for nice arrows
    mag = np.hypot(dVdt, dwdt) + 1e-20
    dVn, dwn = dVdt/mag, dwdt/mag

    # ---- plots ----
    fig = plt.figure(figsize=(11,8))

    # phase plane
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1.quiver(VV*1e3, WW*1e12, dVn, dwn, angles='xy', scale_units='xy', scale=20)
    ax1.plot(V_grid*1e3, wV*1e12, label="V-nullcline (I=0)")
    ax1.plot(V_grid*1e3, ww*1e12, label="w-nullcline")
    ax1.plot(V*1e3, w*1e12, lw=1.5, label="trajectory (pulse)")

    for Vstar, wstar, typ, eig in fp_info:
        ax1.plot([Vstar*1e3], [wstar*1e12], 'o')
        ax1.text(Vstar*1e3, wstar*1e12, f" {typ}", fontsize=9)

    ax1.set_xlim(Vmin*1e3, Vmax*1e3)
    ax1.set_ylim(-40, 80)  # pA
    ax1.set_xlabel("V (mV)")
    ax1.set_ylabel("w (pA)")
    ax1.set_title(f"Phase plane — {PARAMS['name']} (I=0) — spiked={spiked}")
    ax1.legend(loc="upper right", fontsize=8)

    # V(t)
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2.plot(t*1e3, V*1e3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("V (mV)")
    ax2.set_title("Membrane potential with pulse")

    # w(t)
    ax3 = plt.subplot2grid((2,2),(1,0), colspan=2)
    ax3.plot(t*1e3, w*1e12)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("w (pA)")
    ax3.set_title("Adaptation current with pulse")
    plt.tight_layout()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_7/figures/ex_2bi.png", dpi=300, bbox_inches="tight")
