import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parâmetros ------------------
gCa, gK, gL = 1.1, 2.0, 0.5
ECa, EK, EL = 100.0, -70.0, -50.0
Cm = 1.0
J = 15.0  # corrente constante

# ------------------ Funções do modelo ------------------
def sech(x):  # numerical helper
    return 1.0 / np.cosh(x)

def m_inf(V):
    # steady-state activation for Ca
    return 0.5 * (1.0 + np.tanh((V + 1.0) / 15.0))

def n_inf(V):
    # steady-state activation for K
    return 0.5 * (1.0 + np.tanh(V / 30.0))

def tau_n(V):
    # activation time-constant for K (ms)
    return 5.0 * sech(V / 60.0)

# Nullclines
def n_nullcline(V):
    # n-nullcline: dn/dt = 0 -> n = n_inf(V)
    return n_inf(V)

def V_nullcline(V):
    # V-nullcline: dV/dt = 0 -> solve for n(V)
    return (J - gCa * m_inf(V) * (V - ECa) - gL * (V - EL)) / (gK * (V - EK))

# ------------------ Equilíbrio (interseção) ------------------
def fixed_point():
    # Solve H(V) = n_Vnull(V) - n_inf(V) = 0 (avoid V ≈ EK)
    Vs = np.linspace(-80, 60, 20001)
    H = V_nullcline(Vs) - n_inf(Vs)
    # ignore discontinuity near EK
    mask = np.abs(Vs - EK) > 1e-2
    idx = np.where(mask[:-1] & mask[1:] & (np.sign(H[:-1]) * np.sign(H[1:]) < 0))[0]
    a, b = Vs[idx[0]], Vs[idx[0] + 1]
    # bisection
    for _ in range(60):
        c = 0.5 * (a + b)
        if (V_nullcline(a) - n_inf(a)) * (V_nullcline(c) - n_inf(c)) <= 0:
            b = c
        else:
            a = c
    V_star = 0.5 * (a + b)
    n_star = n_inf(V_star)
    return V_star, n_star

V_star, n_star = fixed_point()
print(f"Equilíbrio ~ V*={V_star:.4f} mV, n*={n_star:.4f}")

# ------------------ Plot das nulclinas ------------------
Vgrid = np.linspace(-80, 60, 2000)
plt.figure(figsize=(7.2, 5.2))

# evita plotar a descontinuidade perto de V=EK
mask_left  = Vgrid < EK - 1e-2
mask_right = Vgrid > EK + 1e-2

plt.plot(Vgrid[mask_left],  V_nullcline(Vgrid[mask_left]),  'k-',  lw=2, label="Nulclina de V")
plt.plot(Vgrid[mask_right], V_nullcline(Vgrid[mask_right]), 'k-',  lw=2)
plt.plot(Vgrid, n_nullcline(Vgrid), 'k--', lw=2, label="Nulclina de n")

plt.scatter([V_star], [n_star], c="k", s=60, zorder=5, label="Ponto fixo")

plt.xlim(-80, 60)
plt.ylim(0, 1)
plt.xlabel("V (mV)")
plt.ylabel("n")
plt.title("Plano de fase n × V com nulclinas (Morris–Lecar, J=15)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2a.png", dpi=300)

