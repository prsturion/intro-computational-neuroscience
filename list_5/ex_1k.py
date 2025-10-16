import numpy as np
import matplotlib.pyplot as plt

# -------------------- Parâmetros do modelo --------------------
a = 0.7
b = 0.8
phi = 0.08

# -------------------- Dinâmica de FHN --------------------
def rhs(v, w, I):
    # FitzHugh–Nagumo right-hand side
    dv = v - (v**3)/3.0 - w + I
    dw = phi * (v + a - b*w)
    return dv, dw

def rk4_step(v, w, dt, I):
    # One RK4 step
    k1v, k1w = rhs(v, w, I)
    k2v, k2w = rhs(v + 0.5*dt*k1v, w + 0.5*dt*k1w, I)
    k3v, k3w = rhs(v + 0.5*dt*k2v, w + 0.5*dt*k2w, I)
    k4v, k4w = rhs(v + dt*k3v,     w + dt*k3w,     I)
    v += (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    w += (dt/6.0)*(k1w + 2*k2w + 2*k3w + k4w)
    return v, w

# -------------------- Equilíbrio para dado I --------------------
def equilibrium_v(I):
    # Solve cubic for v*: -(b/3)v^3 + (b-1)v + (b*I - a) = 0
    coeffs = [-b/3.0, 0.0, (b - 1.0), (b*I - a)]
    roots = np.roots(coeffs)
    real_roots = np.real(roots[np.isclose(np.imag(roots), 0.0)])
    return float(real_roots[np.argmin(np.abs(real_roots))])

def equilibrium(I):
    # Return (v*, w*) from nullclines
    v_star = equilibrium_v(I)
    w_star = v_star - (v_star**3)/3.0 + I
    return v_star, w_star

# -------------------- Estimativa de frequência --------------------
def estimate_frequency(v, t, v_min_peak=0.5, trans_frac=0.4):
    """
    Estimate spikes by local maxima of v above v_min_peak, ignoring the first
    trans_frac fraction of the record as transient. Returns frequency (Hz in model time units).
    """
    n = len(v)
    start = int(trans_frac * n)
    idx = np.arange(1, n-1)
    idx = idx[idx > start]
    peaks = idx[(v[idx] > v[idx-1]) & (v[idx] > v[idx+1]) & (v[idx] > v_min_peak)]
    if len(peaks) < 2:
        return 0.0
    isi = np.diff(t[peaks])
    isi = isi[isi > 0]
    if len(isi) == 0:
        return 0.0
    return 1.0 / np.mean(isi)

# -------------------- Varredura e curva f–I --------------------
I_vals = np.linspace(0.0, 1.0, 81)  # 0 ≤ I ≤ 1
dt = 0.002
T = 300.0
steps = int(T/dt) + 1
t = np.linspace(0.0, T, steps)

freqs = np.zeros_like(I_vals)

# Warm start from I=0 equilibrium (tiny perturbation)
v, w = equilibrium(0.0)
v += 1e-3

for i, I in enumerate(I_vals):
    v_traj = np.empty(steps); w_traj = np.empty(steps)
    v_traj[0], w_traj[0] = v, w
    for k in range(steps-1):
        v, w = rk4_step(v, w, dt, I)
        v_traj[k+1], w_traj[k+1] = v, w
    freqs[i] = estimate_frequency(v_traj, t, v_min_peak=0.5, trans_frac=0.4)

# -------------------- Plot: apenas pontos com f > 0 --------------------
mask = freqs > 0.0  # only positive frequencies
plt.figure(figsize=(8,5))
plt.plot(I_vals[mask], freqs[mask], lw=2)  # plot points and connect them
plt.xlabel("Corrente injetada I")
plt.ylabel("Frequência (1/unidade de tempo)")
plt.title("Curva f–I do FitzHugh–Nagumo (apenas f>0)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_1k.png", dpi=300, bbox_inches="tight")
# plt.show()
