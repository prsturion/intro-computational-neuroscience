import numpy as np
import matplotlib.pyplot as plt

# ===================== Morris–Lecar modificado (do item d) ==================
gCa, gK, gL = 1.0, 2.0, 0.5           # mS/cm^2
ECa, EK, EL = 100.0, -70.0, -50.0     # mV
Cm = 1.0                               # uF/cm^2

# ------------------ steady states and time constant ------------------
def sech(x): 
    return 1.0 / np.cosh(x)

def m_inf(V):
    # Ca activation (unchanged)
    return 0.5 * (1.0 + np.tanh((V + 1.0) / 15.0))

def n_inf(V):
    # Modified K activation
    return 0.5 * (1.0 + np.tanh((V - 10.0) / 14.5))

def tau_n(V):
    # Modified K time constant (ms)
    return 3.0 / np.cosh((V - 10.0) / 29.0)

# ------------------ vector field ------------------
def rhs(V, n, J):
    # Vector field with injected current J
    dVdt = (-gCa * m_inf(V) * (V - ECa)
            - gK * n * (V - EK)
            - gL * (V - EL)
            + J) / Cm
    dndt = (n_inf(V) - n) / tau_n(V)
    return dVdt, dndt

def rk4_step(V, n, dt, J):
    # One RK4 step
    k1V, k1n = rhs(V, n, J)
    k2V, k2n = rhs(V + 0.5*dt*k1V, n + 0.5*dt*k1n, J)
    k3V, k3n = rhs(V + 0.5*dt*k2V, n + 0.5*dt*k2n, J)
    k4V, k4n = rhs(V + dt*k3V,     n + dt*k3n,     J)
    V += (dt/6.0)*(k1V + 2*k2V + 2*k3V + k4V)
    n += (dt/6.0)*(k1n + 2*k2n + 2*k3n + k4n)
    return V, n

# ------------------ nullclines ------------------
def n_nullcline(V): 
    return n_inf(V)

def V_nullcline(V, J):
    # Solve dV/dt = 0 for n(V; J)
    return (-gCa * m_inf(V) * (V - ECa) - gL * (V - EL) + J) / (gK * (V - EK))

# ------------------ fixed points (utility to start near equilibrium) --------
def fixed_points(J):
    # Find all intersections of V-nullcline and n-nullcline
    Vs = np.linspace(-80, 60, 40001)
    H = V_nullcline(Vs, J) - n_inf(Vs)
    mask = np.abs(Vs - EK) > 1e-2
    idxs = np.where(mask[:-1] & mask[1:] & (np.sign(H[:-1]) * np.sign(H[1:]) < 0))[0]
    roots = []
    for idx in idxs:
        a, b = Vs[idx], Vs[idx+1]
        for _ in range(60):  # bisection
            c = 0.5*(a + b)
            Ha = V_nullcline(a, J) - n_inf(a)
            Hc = V_nullcline(c, J) - n_inf(c)
            if Ha * Hc <= 0: b = c
            else: a = c
        V_star = 0.5*(a + b)
        n_star = n_inf(V_star)
        roots.append((V_star, n_star))
    return roots

def pick_stable_or_leftmost(J):
    # Pick a stable fixed point if present; otherwise the leftmost one
    fps = fixed_points(J)
    if not fps:
        return -50.0, n_inf(-50.0)
    # classify via trace/determinant (only sign needed)
    def classify(Vs, ns):
        FV = (-gCa*(0.5*sech((Vs + 1.0)/15.0)**2/15.0*(Vs - ECa) + m_inf(Vs)) - gK*ns - gL) / Cm
        Fn = (-gK * (Vs - EK)) / Cm
        GV = (1.0/29.0)*sech((Vs - 10.0)/14.5)**2 / tau_n(Vs)
        Gn = -1.0 / tau_n(Vs)
        tr = FV + Gn
        det = FV*Gn - Fn*GV
        return tr, det
    for Vs, ns in fps:
        tr, det = classify(Vs, ns)
        if det > 0 and tr < 0:
            return Vs, ns
    # fallback: leftmost
    return min(fps, key=lambda z: z[0])

# ------------------ frequency estimator ------------------
def estimate_frequency(V, t, trans_frac=0.5):
    # Peak-based estimate after a transient fraction of the signal
    n = len(V); start = int(trans_frac*n)
    idx = np.arange(1, n-1); idx = idx[idx > start]
    peaks = idx[(V[idx] > V[idx-1]) & (V[idx] > V[idx+1])]
    if len(peaks) < 2:
        return 0.0
    isi = np.diff(t[peaks])
    isi = isi[isi > 0]
    if len(isi) == 0:
        return 0.0
    return 1.0 / np.mean(isi)

# ===================== 1) f–I curve (linha sem marcadores) ===================
J_vals = np.linspace(0.0, 20.0, 81)
dt = 0.01
T = 800.0
steps = int(T/dt) + 1
t = np.linspace(0.0, T, steps)

freqs = np.zeros_like(J_vals)

# adiabatic continuation across J for robustness
V0, n0 = pick_stable_or_leftmost(J_vals[0])
V0 += 0.5  # small kick

for i, J in enumerate(J_vals):
    V = np.empty(steps); n = np.empty(steps)
    V[0], n[0] = V0, n0
    for k in range(steps-1):
        V0, n0 = rk4_step(V0, n0, dt, J)
        V[k+1], n[k+1] = V0, n0
    freqs[i] = estimate_frequency(V, t, trans_frac=0.5)

mask = freqs > 0.0
plt.figure(figsize=(8,5))
plt.plot(J_vals[mask], freqs[mask], lw=2)  # line only, no markers
plt.xlabel("Corrente injetada J (µA/cm²)")
plt.ylabel("Frequência (1/ms)")
plt.title("Curva f–I (Morris–Lecar modificado)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2f_fI.png", dpi=300, bbox_inches="tight")

# ===================== 2) Vm × t (três Js) ==================================
J_three = [7.90, 8.33, 8.35]
T_vt = 400.0
steps_vt = int(T_vt/dt) + 1
t_vt = np.linspace(0.0, T_vt, steps_vt)

plt.figure(figsize=(9,5))
for J in J_three:
    Vst, nst = pick_stable_or_leftmost(J)
    V, n = Vst + 0.5, nst                 # small perturbation
    Vtraj = np.empty(steps_vt)
    Vtraj[0] = V
    for k in range(steps_vt-1):
        V, n = rk4_step(V, n, dt, J)
        Vtraj[k+1] = V
    plt.plot(t_vt, Vtraj, label=f"J = {J:.2f} µA/cm²")

plt.xlabel("Tempo (ms)")
plt.ylabel("V (mV)")
plt.title("Morris–Lecar modificado: V × t para três valores de J")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2f_Vt.png", dpi=300, bbox_inches="tight")

# ===================== 3) Plano de fase com MESMAS nulclinas + 3 trajetórias =
Vgrid = np.linspace(-80, 60, 2500)
mask_left  = Vgrid < EK - 1e-2
mask_right = Vgrid > EK + 1e-2

J_null = 8.33  # per request: single set of nullclines (representative J)
plt.figure(figsize=(7.8,6.4))

# Single pair of nullclines
plt.plot(Vgrid, n_nullcline(Vgrid), 'k--', lw=2, label="Nulclina de n")
plt.plot(Vgrid[mask_left],  V_nullcline(Vgrid[mask_left],  J_null), 'k-', lw=2, label=f"Nulclina de V (J={J_null:.2f})")
plt.plot(Vgrid[mask_right], V_nullcline(Vgrid[mask_right], J_null), 'k-', lw=2)

# Three trajectories on the same plot
colors = ["tab:blue", "tab:orange", "tab:green"]
for J, col in zip(J_three, colors):
    Vst, nst = pick_stable_or_leftmost(J)
    V, n = Vst + 0.5, nst
    Tpp = 400.0
    steps_pp = int(Tpp/dt) + 1
    Vt = np.empty(steps_pp); nt = np.empty(steps_pp)
    Vt[0], nt[0] = V, n
    for k in range(steps_pp-1):
        V, n = rk4_step(V, n, dt, J)
        Vt[k+1], nt[k+1] = V, n
    plt.plot(Vt, nt, color=col, lw=1.8, alpha=0.95, label=f"trajetória (J={J:.2f})")

plt.xlim(-80, 60)
plt.ylim(-.2, 1)
plt.xlabel("V (mV)")
plt.ylabel("n")
plt.title("Plano de fase n × V: nulclinas (J=8.33) e trajetórias (J=7.90, 8.33, 8.35)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2f_phase.png", dpi=300, bbox_inches="tight")
# plt.show()
