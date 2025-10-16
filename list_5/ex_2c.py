import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parameters (Rinzel–Ermentrout) ------------------
gCa, gK, gL = 1.1, 2.0, 0.5          # mS/cm^2
ECa, EK, EL = 100.0, -70.0, -50.0    # mV
Cm = 1.0                              # uF/cm^2

# ------------------ Steady-state and time-constant functions ----------------
def sech(x): return 1.0 / np.cosh(x)

def m_inf(V):
    # Ca activation steady state
    return 0.5 * (1.0 + np.tanh((V + 1.0) / 15.0))

def n_inf(V):
    # K activation steady state
    return 0.5 * (1.0 + np.tanh(V / 30.0))

def tau_n(V):
    # K activation time constant (ms)
    return 5.0 * sech(V / 60.0)

# ------------------ Vector field (depends on J) -----------------------------
def rhs(V, n, J):
    # Morris–Lecar right-hand side for a given injected current J
    dVdt = (-gCa * m_inf(V) * (V - ECa)
            - gK * n * (V - EK)
            - gL * (V - EL)
            + J) / Cm
    dndt = (n_inf(V) - n) / tau_n(V)
    return dVdt, dndt

def rk4_step(V, n, dt, J):
    # One RK4 step at fixed J
    k1V, k1n = rhs(V, n, J)
    k2V, k2n = rhs(V + 0.5*dt*k1V, n + 0.5*dt*k1n, J)
    k3V, k3n = rhs(V + 0.5*dt*k2V, n + 0.5*dt*k2n, J)
    k4V, k4n = rhs(V + dt*k3V,     n + dt*k3n,     J)
    V += (dt/6.0)*(k1V + 2*k2V + 2*k3V + k4V)
    n += (dt/6.0)*(k1n + 2*k2n + 2*k3n + k4n)
    return V, n

# ------------------ Nullclines for plotting (given J) -----------------------
def n_nullcline(V):  # n-nullcline
    return n_inf(V)

def V_nullcline(V, J):  # V-nullcline: solve for n(V;J)
    return (J - gCa*m_inf(V)*(V - ECa) - gL*(V - EL)) / (gK*(V - EK))

# ------------------ Fixed point for given J ---------------------------------
def fixed_point(J):
    # Solve H(V) = V_nullcline(V,J) - n_inf(V) = 0 by bisection
    Vs = np.linspace(-80, 60, 40001)
    H = V_nullcline(Vs, J) - n_inf(Vs)
    mask = np.abs(Vs - EK) > 1e-2  # avoid asymptote at V=EK
    idx = np.where(mask[:-1] & mask[1:] & (np.sign(H[:-1]) * np.sign(H[1:]) < 0))[0]
    if len(idx) == 0:
        # No crossing found (e.g., purely oscillatory regime); return a reasonable seed
        return -30.0, n_inf(-30.0)
    a, b = Vs[idx[0]], Vs[idx[0]+1]
    for _ in range(60):  # bisection
        c = 0.5 * (a + b)
        if (V_nullcline(a, J) - n_inf(a)) * (V_nullcline(c, J) - n_inf(c)) <= 0:
            b = c
        else:
            a = c
    V_star = 0.5 * (a + b)
    n_star = n_inf(V_star)
    return V_star, n_star

# ------------------ Frequency estimation ------------------------------------
def estimate_frequency(V, t, v_min_peak=0.0, trans_frac=0.4):
    """
    Detect spikes by local maxima of V above v_min_peak.
    Use only the post-transient portion defined by trans_frac.
    Returns frequency in 1/ms (model time units).
    """
    n = len(V)
    start = int(trans_frac * n)
    idx = np.arange(1, n-1)
    idx = idx[idx > start]
    peaks = idx[(V[idx] > V[idx-1]) & (V[idx] > V[idx+1]) & (V[idx] > v_min_peak)]
    if len(peaks) < 2:
        return 0.0
    isi = np.diff(t[peaks])
    isi = isi[isi > 0]
    if len(isi) == 0:
        return 0.0
    return 1.0 / np.mean(isi)

# ------------------ Sweep J: f–I curve --------------------------------------
J_vals = np.linspace(20.0, 40.0, 81)  # 20 → 40
dt = 0.01
T  = 800.0
steps = int(T/dt) + 1
t = np.linspace(0.0, T, steps)

freqs = np.zeros_like(J_vals)

# Adiabatic continuation across J (use last state as initial guess)
V, n = fixed_point(J_vals[0])
V += 1e-3  # tiny perturbation to escape unstable fixed points

for i, J in enumerate(J_vals):
    Vtraj = np.empty(steps); ntraj = np.empty(steps)
    Vtraj[0], ntraj[0] = V, n
    for k in range(steps-1):
        V, n = rk4_step(V, n, dt, J)
        Vtraj[k+1], ntraj[k+1] = V, n
    # Frequency from peaks after transient
    freqs[i] = estimate_frequency(Vtraj, t, v_min_peak=0.0, trans_frac=0.5)

# Plot f–I (only f>0)
mask = freqs > 0.0
plt.figure(figsize=(8,5))
plt.plot(J_vals[mask], freqs[mask], lw=2)
plt.xlabel("Corrente injetada J (µA/cm²)")
plt.ylabel("Frequência (1/ms)")
plt.title("Curva f–I do Morris–Lecar (20 ≤ J ≤ 40)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2c_fI.png", dpi=300, bbox_inches="tight")

# Estimate onset Jc and classify Type I vs II (heuristic)
pos_idx = np.where(mask)[0]
if len(pos_idx) > 0:
    Jc_idx = pos_idx[0]
    Jc = J_vals[Jc_idx]
    f_on = freqs[Jc_idx]
    kind = "Tipo I (f → 0⁺)" if f_on < 1e-3 else "Tipo II (salto em f)"
    print(f"Estimativa: J_c ≈ {Jc:.3f} µA/cm²; f_on ≈ {f_on:.4f} (1/ms) → {kind}")
else:
    print("Nenhuma oscilação encontrada no intervalo 20–40 µA/cm².")
    Jc = 25.0  # fallback for the next plots

# ------------------ Vm vs t for an oscillatory case -------------------------
J_plot = max(Jc + 0.5, 25.0)  # choose a clearly oscillatory value
V, n = fixed_point(J_plot)
V += 1.0  # small kick to start
T_vt = 300.0
steps_vt = int(T_vt/dt) + 1
t_vt = np.linspace(0.0, T_vt, steps_vt)
Vtraj = np.empty(steps_vt); ntraj = np.empty(steps_vt)
Vtraj[0], ntraj[0] = V, n
for k in range(steps_vt-1):
    V, n = rk4_step(V, n, dt, J_plot)
    Vtraj[k+1], ntraj[k+1] = V, n

plt.figure(figsize=(9,5))
plt.plot(t_vt, Vtraj, lw=1.8)
plt.xlabel("Tempo (ms)")
plt.ylabel("V (mV)")
plt.title(f"Morris–Lecar: V × t para J = {J_plot:.2f} µA/cm²")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2c_Vt.png", dpi=300, bbox_inches="tight")

# ------------------ Phase planes for J = 24.8 and J = 25.0 ------------------
def phase_plane_with_traj(J, fname):
    # Build nullclines and a trajectory converging to the attractor
    V0, n0 = fixed_point(J)
    V0 += 10  # small perturbation
    Tpp = 300.0
    steps_pp = int(Tpp/dt) + 1
    Vt = np.empty(steps_pp); nt = np.empty(steps_pp)
    Vt[0], nt[0] = V0, n0
    for k in range(steps_pp-1):
        V0, n0 = rk4_step(V0, n0, dt, J)
        Vt[k+1], nt[k+1] = V0, n0

    Vgrid = np.linspace(-80, 60, 2500)
    mask_left  = Vgrid < EK - 1e-2
    mask_right = Vgrid > EK + 1e-2

    plt.figure(figsize=(7.2,6.2))
    plt.plot(Vgrid[mask_left],  V_nullcline(Vgrid[mask_left], J),  'k-', lw=2, label="Nulclina de V")
    plt.plot(Vgrid[mask_right], V_nullcline(Vgrid[mask_right], J), 'k-', lw=2)
    plt.plot(Vgrid, n_nullcline(Vgrid), 'k--', lw=2, label="Nulclina de n")
    plt.plot(Vt, nt, lw=1.8, label="trajetória")
    plt.xlim(-80, 60); plt.ylim(0, 1)
    plt.xlabel("V (mV)"); plt.ylabel("n")
    plt.title(f"Plano de fase n × V (J = {J:.1f} µA/cm²)")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(fname, dpi=300, bbox_inches="tight")

phase_plane_with_traj(24.8, "intro-computational-neuroscience/list_5/figures/ex_2c_phase_J24p8.png")
phase_plane_with_traj(25.0, "intro-computational-neuroscience/list_5/figures/ex_2c_phase_J25p0.png")
# plt.show()  # (intencionalmente não exibido)
