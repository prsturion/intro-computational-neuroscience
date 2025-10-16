import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parâmetros ------------------
gCa, gK, gL = 1.1, 2.0, 0.5
ECa, EK, EL = 100.0, -70.0, -50.0
Cm = 1.0
J = 15.0  # corrente constante

# ------------------ Funções do modelo ------------------
def sech(x):
    return 1.0 / np.cosh(x)

def m_inf(V):
    return 0.5 * (1.0 + np.tanh((V + 1.0) / 15.0))

def n_inf(V):
    return 0.5 * (1.0 + np.tanh(V / 30.0))

def tau_n(V):
    return 5.0 * sech(V / 60.0)

def rhs(V, n):
    dVdt = (-gCa * m_inf(V) * (V - ECa)
            - gK * n * (V - EK)
            - gL * (V - EL)
            + J) / Cm
    dndt = (n_inf(V) - n) / tau_n(V)
    return dVdt, dndt

def rk4_step(V, n, dt):
    # Single RK4 step
    k1V, k1n = rhs(V, n)
    k2V, k2n = rhs(V + 0.5*dt*k1V, n + 0.5*dt*k1n)
    k3V, k3n = rhs(V + 0.5*dt*k2V, n + 0.5*dt*k2n)
    k4V, k4n = rhs(V + dt*k3V, n + dt*k3n)
    V_next = V + (dt/6.0)*(k1V + 2*k2V + 2*k3V + k4V)
    n_next = n + (dt/6.0)*(k1n + 2*k2n + 2*k3n + k4n)
    return V_next, n_next

# ------------------ Nulclinas ------------------
def n_nullcline(V):
    return n_inf(V)

def V_nullcline(V):
    return (J - gCa*m_inf(V)*(V - ECa) - gL*(V - EL)) / (gK*(V - EK))

# ------------------ Ponto de equilíbrio ------------------
def fixed_point():
    Vs = np.linspace(-80, 60, 40001)
    H = V_nullcline(Vs) - n_inf(Vs)
    mask = np.abs(Vs - EK) > 1e-2
    idx = np.where(mask[:-1] & mask[1:] & (np.sign(H[:-1]) * np.sign(H[1:]) < 0))[0]
    a, b = Vs[idx[0]], Vs[idx[0]+1]
    for _ in range(60):
        c = 0.5*(a + b)
        if (V_nullcline(a) - n_inf(a)) * (V_nullcline(c) - n_inf(c)) <= 0:
            b = c
        else:
            a = c
    V_star = 0.5*(a + b)
    n_star = n_inf(V_star)
    return V_star, n_star

V_star, n_star = fixed_point()
print(f"Ponto fixo: V* = {V_star:.3f} mV, n* = {n_star:.4f}")

# ------------------ Simulação ------------------
T = 20.0
dt = 0.01
steps = int(T/dt) + 1
t = np.linspace(0.0, T, steps)

V_inits = [-18.0, -14.8, -14.7, -12.0]
n0 = n_star
sims = []

for V0 in V_inits:
    V = np.empty(steps); n = np.empty(steps)
    V[0], n[0] = V0, n0
    for k in range(steps-1):
        V[k+1], n[k+1] = rk4_step(V[k], n[k], dt)
    sims.append({"V": V, "n": n, "label": f"V(0) = {V0:.1f} mV"})

# ------------------ (i) V × t ------------------
plt.figure(figsize=(9,5))
for s in sims:
    plt.plot(t, s["V"], label=s["label"])
plt.axhline(V_star, color="k", ls="--", lw=1, label="V* (equilíbrio)") #type: ignore
plt.xlabel("Tempo (ms)")
plt.ylabel("V (mV)")
plt.title("Morris–Lecar (J=15): V × t para diferentes V(0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2b_1.png")

# ------------------ (ii) n × t ------------------
plt.figure(figsize=(9,5))
for s in sims:
    plt.plot(t, s["n"], label=s["label"])
plt.axhline(n_star, color="k", ls="--", lw=1, label="n* (equilíbrio)")
plt.xlabel("Tempo (ms)")
plt.ylabel("n")
plt.title("Morris–Lecar (J=15): n × t para diferentes V(0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2b_2.png")

# ------------------ (iii) Plano de fase n × V ------------------
Vgrid = np.linspace(-80, 60, 2000)
mask_left  = Vgrid < EK - 1e-2
mask_right = Vgrid > EK + 1e-2

plt.figure(figsize=(7.2,5.2))
plt.plot(Vgrid[mask_left],  V_nullcline(Vgrid[mask_left]),  'k-',  lw=2, label="Nulclina de V")
plt.plot(Vgrid[mask_right], V_nullcline(Vgrid[mask_right]), 'k-',  lw=2)
plt.plot(Vgrid, n_nullcline(Vgrid), 'k--', lw=2, label="Nulclina de n")

markers = ["o", "s", "^", "d"]
for s, mk in zip(sims, markers):
    plt.plot(s["V"], s["n"], lw=1.8, label=f"trajetória ({s['label']})")
    plt.scatter(s["V"][0], s["n"][0], s=50, marker=mk) # type: ignore

plt.scatter([V_star], [n_star], c="k", s=60, zorder=5, label="Ponto fixo")
plt.xlim(-80, 60)
plt.ylim(0, 1)
plt.xlabel("V (mV)")
plt.ylabel("n")
plt.title("Plano de fase n × V com nulclinas e trajetórias (J=15)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2b_3.png")
