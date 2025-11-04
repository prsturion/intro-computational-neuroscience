#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdEx — Phase-plane with constant step current (part b-ii).

- Use constant I_dc != previous pulse to elicit spiking.
- Plot w- and V-nullclines for I=I_dc, mark fixed points and stability.
- Simulate trajectory; mark reset points after each spike.
- Plot V(t) and w(t).

Model (tau_m, R) form:
  tau_m dV/dt = (E_L - V) + Delta_T*exp((V - V_T)/Delta_T) + R*(I - w)
  tau_w dw/dt = a*(V - E_L) - w
Reset: if V >= V_peak -> V = V_reset, w = w + b
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Base constants (SI) ----------------
E_L     = -70e-3   # V
R       = 500e6    # Ohm
V_T     = -50e-3   # V
Delta_T =  2e-3    # V
V_peak  = 20e-3    # V

DT       = 1e-5    # s (0.01 ms)
EXP_CLIP = 40.0
DV_LIMIT = 5e-3    # V per step (safety)

# ------------- Choose a parameter set (Table 6.1-like) -------------
PARAMS = dict(
    name="tonic",
    tau_m_ms=20.0,   # ms
    a_nS=0.0,        # nS (set >0 for adapting)
    tau_w_ms=30.0,   # ms
    b_pA=60.0,       # pA
    Vreset_mV=-55.0  # mV
)

# Constant step current (different from previous item)
I_DC_pA = 60    # pA   <-- change if you want more/less spiking
T_TOTAL = 0.5       # s of simulation

# ------------- Helpers -------------
def safe_exp(x): return np.exp(np.minimum(x, EXP_CLIP))

def rhs(v, w, I, tau_m, tau_w, a):
    dVdt = ((E_L - v) + Delta_T*safe_exp((v - V_T)/Delta_T) + R*(I - w)) / tau_m
    dwdt = (a*(v - E_L) - w) / tau_w
    return dVdt, dwdt

def simulate_constant_I(params, I_dc_pA, T=T_TOTAL, dt=DT):
    tau_m = params["tau_m_ms"]*1e-3
    tau_w = params["tau_w_ms"]*1e-3
    a     = params["a_nS"]*1e-9
    b     = params["b_pA"]*1e-12
    Vrst  = params["Vreset_mV"]*1e-3
    I_dc  = I_dc_pA*1e-12

    t = np.arange(0.0, T+dt, dt)
    V = np.empty_like(t); w = np.empty_like(t)
    V[0] = E_L; w[0] = 0.0

    resets_V = []  # points (V_reset, w_after_reset) to plot on phase plane
    resets_w = []

    for k in range(len(t)-1):
        v, u = V[k], w[k]
        # pre-reset (numerical safety)
        if v >= V_peak:
            v = Vrst
            u = u + b
            resets_V.append(v); resets_w.append(u)

        dVdt, dwdt = rhs(v, u, I_dc, tau_m, tau_w, a)
        dv = np.clip(dVdt*dt, -DV_LIMIT, DV_LIMIT)
        du = dwdt*dt

        v_next = v + dv
        u_next = u + du

        # post-reset + record reset point
        if v_next >= V_peak:
            v_next = Vrst
            u_next = u_next + b
            resets_V.append(v_next); resets_w.append(u_next)

        V[k+1] = v_next
        w[k+1] = u_next

    return t, V, w, np.array(resets_V), np.array(resets_w)

def nullclines_I(params, V_grid, I_dc_pA):
    """Nullclines for a given constant I."""
    a   = params["a_nS"]*1e-9
    I_A = I_dc_pA*1e-12
    # dV/dt=0 => w = I + (E_L - V)/R + (Delta_T/R) exp((V-V_T)/Delta_T)
    w_V = I_A + (E_L - V_grid)/R + (Delta_T/R)*safe_exp((V_grid - V_T)/Delta_T)
    # dw/dt=0 => w = a (V - E_L)
    w_w = a*(V_grid - E_L)
    return w_V, w_w

def find_fixed_points_I(params, I_dc_pA, V_min=-0.09, V_max=-0.03, n_samples=3000):
    a   = params["a_nS"]*1e-9
    I_A = I_dc_pA*1e-12
    Vs = np.linspace(V_min, V_max, n_samples)
    def F(V):
        return I_A + (E_L - V)/R + (Delta_T/R)*safe_exp((V - V_T)/Delta_T) - a*(V - E_L)
    vals = F(Vs)
    roots = []
    for i in range(len(Vs)-1):
        y1, y2 = vals[i], vals[i+1]
        if np.isnan(y1) or np.isnan(y2): continue
        if y1 == 0.0: Vstar = Vs[i]
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

def classify_fixed_point(Vs, ws, params, I_dc_pA):
    tau_m = params["tau_m_ms"]*1e-3
    tau_w = params["tau_w_ms"]*1e-3
    a     = params["a_nS"]*1e-9
    # Jacobian does not depend on I explicitly (I is constant) except via V*
    exp_term = safe_exp((Vs - V_T)/Delta_T)
    dFdV = (-1.0 + exp_term)/tau_m
    dFdW = (-R)/tau_m
    dGdV = (a)/tau_w
    dGdW = (-1.0)/tau_w
    J = np.array([[dFdV, dFdW],[dGdV, dGdW]], float)
    eig = np.linalg.eigvals(J); re = np.real(eig); im = np.imag(eig)
    if np.all(im == 0):
        typ = "stable node" if np.all(re < 0) else ("saddle" if (re.min()<0<re.max()) else "unstable node")
    else:
        typ = "stable focus" if np.max(re) < 0 else "unstable focus"
    return typ, eig

# ---------------- Main ----------------
if __name__ == "__main__":
    # nullclines for this I
    Vmin, Vmax = -0.085, -0.035  # zoom window [-85, -35] mV
    V_grid = np.linspace(Vmin, Vmax, 1000)
    wV, ww = nullclines_I(PARAMS, V_grid, I_DC_pA)

    # fixed points + classification
    fps = find_fixed_points_I(PARAMS, I_DC_pA, V_min=Vmin, V_max=Vmax)
    fp_info = [(*fp, *classify_fixed_point(fp[0], fp[1], PARAMS, I_DC_pA)) for fp in fps]

    # simulate
    t, V, w, rV, rW = simulate_constant_I(PARAMS, I_DC_pA, T=T_TOTAL, dt=DT)

    # vector field (I=I_dc)
    Vq = np.linspace(Vmin, Vmax, 20)
    wq = np.linspace(-40e-12, 120e-12, 20)
    VV, WW = np.meshgrid(Vq, wq)
    tau_m = PARAMS["tau_m_ms"]*1e-3 # type: ignore
    tau_w = PARAMS["tau_w_ms"]*1e-3 # type: ignore
    a     = PARAMS["a_nS"]*1e-9 # type: ignore
    I_A   = I_DC_pA*1e-12
    dVdt, dwdt = rhs(VV, WW, I_A, tau_m, tau_w, a)
    mag = np.hypot(dVdt, dwdt) + 1e-20
    dVn, dwn = dVdt/mag, dwdt/mag

    # -------- plots --------
    fig = plt.figure(figsize=(11,8))

    # phase plane
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1.quiver(VV*1e3, WW*1e12, dVn, dwn, angles='xy', scale_units='xy', scale=20)
    ax1.plot(V_grid*1e3, wV*1e12, label="V-nullcline (I=I_dc)")
    ax1.plot(V_grid*1e3, ww*1e12, label="w-nullcline")
    ax1.plot(V*1e3, w*1e12, lw=1.5, label="trajectory")
    if rV.size:
        ax1.plot(rV*1e3, rW*1e12, 'x', label="reset points", ms=6)

    for Vstar, wstar, typ, eig in fp_info:
        ax1.plot([Vstar*1e3], [wstar*1e12], 'o')
        ax1.text(Vstar*1e3, wstar*1e12, f" {typ}", fontsize=9)

    ax1.set_xlim(Vmin*1e3, Vmax*1e3)
    ax1.set_ylim(-40, 120)  # pA
    ax1.set_xlabel("V (mV)")
    ax1.set_ylabel("w (pA)")
    ax1.set_title(f"Phase plane — {PARAMS['name']} with I_dc={I_DC_pA:.0f} pA")
    ax1.legend(loc="upper right", fontsize=8)

    # V(t)
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2.plot(t*1e3, V*1e3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("V (mV)")
    ax2.set_title("Membrane potential (constant current)")

    # w(t)
    ax3 = plt.subplot2grid((2,2),(1,0), colspan=2)
    ax3.plot(t*1e3, w*1e12)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("w (pA)")
    ax3.set_title("Adaptation current (constant current)")

    plt.tight_layout()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_7/figures/ex_2bii.png", dpi=300, bbox_inches="tight")

    # console summary
    print(f"Fixed points for I_dc={I_DC_pA:.1f} pA:")
    for Vstar, wstar, typ, eig in fp_info:
        print(f"  V*={Vstar*1e3:.2f} mV, w*={wstar*1e12:.2f} pA -> {typ}, eig={eig}")
