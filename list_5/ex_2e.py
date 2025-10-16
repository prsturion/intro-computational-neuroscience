import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parâmetros (modelo modificado do 2(d)) ------------------
gCa, gK, gL = 1.0, 2.0, 0.5           # mS/cm^2
ECa, EK, EL = 100.0, -70.0, -50.0     # mV
Cm = 1.0                               # uF/cm^2
J = 0.0                                # sem corrente de ajuste (item e)

# ------------------ Funções de estado estacionário e tau_n ------------------
def sech(x):  # numerical helper
    return 1.0 / np.cosh(x)

def m_inf(V):
    # steady-state activation for Ca (unchanged)
    return 0.5 * (1.0 + np.tanh((V + 1.0) / 15.0))

def n_inf(V):
    # modified steady-state for K
    return 0.5 * (1.0 + np.tanh((V - 10.0) / 14.5))

def tau_n(V):
    # modified time constant for K (ms)
    return 3.0 / np.cosh((V - 10.0) / 29.0)  # = 3*sech((V-10)/29)

# (optional) derivatives used to classify equilibria
def m_inf_prime(V):
    # 0.5 * (1/15) * sech^2((V+1)/15) = 1/30 * sech^2
    return 0.5 * sech((V + 1.0) / 15.0)**2 / 15.0

def n_inf_prime(V):
    # = (1/29) * sech^2((V-10)/14.5)
    return (1.0 / 29.0) * sech((V - 10.0) / 14.5)**2

# ------------------ Campo vetorial e integrador -----------------------------
def rhs(V, n):
    # Right-hand side of modified Morris–Lecar
    dVdt = (-gCa * m_inf(V) * (V - ECa)
            - gK * n * (V - EK)
            - gL * (V - EL)
            + J) / Cm
    dndt = (n_inf(V) - n) / tau_n(V)
    return dVdt, dndt

def rk4_step(V, n, dt):
    # One RK4 step
    k1V, k1n = rhs(V, n)
    k2V, k2n = rhs(V + 0.5*dt*k1V, n + 0.5*dt*k1n)
    k3V, k3n = rhs(V + 0.5*dt*k2V, n + 0.5*dt*k2n)
    k4V, k4n = rhs(V + dt*k3V,     n + dt*k3n)
    V += (dt/6.0)*(k1V + 2*k2V + 2*k3V + k4V)
    n += (dt/6.0)*(k1n + 2*k2n + 2*k3n + k4n)
    return V, n

# ------------------ Nulclinas ------------------------------------------------
def n_nullcline(V):
    # n-nullcline: n = n_inf(V)
    return n_inf(V)

def V_nullcline(V):
    # V-nullcline: solve dV/dt = 0 for n(V)
    return (-gCa * m_inf(V) * (V - ECa) - gL * (V - EL) + J) / (gK * (V - EK))

# ------------------ Encontrar e classificar pontos fixos --------------------
def fixed_points():
    # Find all sign changes of H(V) = V_nullcline(V) - n_inf(V)
    Vs = np.linspace(-80, 60, 40001)
    H = V_nullcline(Vs) - n_inf(Vs)
    mask = np.abs(Vs - EK) > 1e-2  # avoid asymptote at V=EK
    idxs = np.where(mask[:-1] & mask[1:] & (np.sign(H[:-1]) * np.sign(H[1:]) < 0))[0]
    roots = []
    for idx in idxs:
        a, b = Vs[idx], Vs[idx+1]
        for _ in range(60):  # bisection refinement
            c = 0.5*(a + b)
            Ha = V_nullcline(a) - n_inf(a)
            Hc = V_nullcline(c) - n_inf(c)
            if Ha * Hc <= 0:
                b = c
            else:
                a = c
        V_star = 0.5*(a + b)
        n_star = n_inf(V_star)
        roots.append((V_star, n_star))
    return roots

def classify_point(Vs, ns):
    # Classify stability via Jacobian trace/determinant
    FV = (-gCa * (m_inf_prime(Vs) * (Vs - ECa) + m_inf(Vs))
          - gK * ns - gL) / Cm
    Fn = (-gK * (Vs - EK)) / Cm
    GV = n_inf_prime(Vs) / tau_n(Vs)
    Gn = -1.0 / tau_n(Vs)

    tr = FV + Gn
    det = FV*Gn - Fn*GV
    return tr, det  # stable if det>0 and tr<0

fps = fixed_points()
# Pick the stable fixed point to set n(0) = n*
stable_fp = None
for (Vst, nst) in fps:
    tr, det = classify_point(Vst, nst)
    if det > 0 and tr < 0:
        stable_fp = (Vst, nst)
        break
if stable_fp is None:
    raise RuntimeError("Nenhum equilíbrio estável encontrado para definir n(0).")

Vstar, nstar = stable_fp
print(f"Equilíbrio estável usado: V*={Vstar:.3f} mV, n*={nstar:.4f}")

# ------------------ Simulações (t=0..200 ms) --------------------------------
T = 20.0
dt = 0.01
steps = int(T/dt) + 1
t = np.linspace(0.0, T, steps)

V_inits = [-40.0, -15.0, -12.0, -2.0]  # mV
sims = []

for V0 in V_inits:
    V = np.empty(steps); n = np.empty(steps)
    V[0], n[0] = V0, nstar  # n(0) = n* (stable equilibrium)
    for k in range(steps-1):
        V[k+1], n[k+1] = rk4_step(V[k], n[k], dt)
    sims.append({"V": V, "n": n, "label": f"V(0) = {V0:.0f} mV"})

# ------------------ (i) V × t -----------------------------------------------
plt.figure(figsize=(9,5))
for s in sims:
    plt.plot(t, s["V"], label=s["label"])
plt.axhline(Vstar, color="k", ls="--", lw=1, label="V* (equilíbrio estável)")
plt.xlabel("Tempo (ms)")
plt.ylabel("V (mV)")
plt.title("Morris–Lecar modificado (J=0): V × t para diferentes V(0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2e_1.png", dpi=300, bbox_inches="tight")

# ------------------ (ii) n × t ----------------------------------------------
plt.figure(figsize=(9,5))
for s in sims:
    plt.plot(t, s["n"], label=s["label"])
plt.axhline(nstar, color="k", ls="--", lw=1, label="n* (equilíbrio estável)")
plt.xlabel("Tempo (ms)")
plt.ylabel("n")
plt.title("Morris–Lecar modificado (J=0): n × t para diferentes V(0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2e_2.png", dpi=300, bbox_inches="tight")

# ------------------ (iii) Plano de fase n × V --------------------------------
Vgrid = np.linspace(-80, 60, 2500)
mask_left  = Vgrid < EK - 1e-2
mask_right = Vgrid > EK + 1e-2

plt.figure(figsize=(7.6,6.2))
# Nullclines (avoid vertical asymptote at V=EK)
plt.plot(Vgrid[mask_left],  V_nullcline(Vgrid[mask_left]),  'k-', lw=2, label="Nulclina de V")
plt.plot(Vgrid[mask_right], V_nullcline(Vgrid[mask_right]), 'k-', lw=2)
plt.plot(Vgrid, n_nullcline(Vgrid), 'k--', lw=2, label="Nulclina de n")

# Trajectories and initial points
markers = ["o", "s", "^", "d"]
for s, mk in zip(sims, markers):
    plt.plot(s["V"], s["n"], lw=1.8, label=f"trajetória ({s['label']})")
    plt.scatter(s["V"][0], s["n"][0], s=50, marker=mk) # type: ignore

# Mark the stable fixed point
plt.scatter([Vstar], [nstar], c="k", s=60, zorder=5, label="Ponto fixo estável")

plt.xlim(-80, 60)
plt.ylim(-.2, 1)
plt.xlabel("V (mV)")
plt.ylabel("n")
plt.title("Plano de fase n × V com nulclinas e trajetórias (modificado, J=0)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2e_3.png", dpi=300, bbox_inches="tight")
# plt.show()
