import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parameters ------------------
gCa, gK, gL = 1.0, 2.0, 0.5           # mS/cm^2 (gCa changed to 1.0)
ECa, EK, EL = 100.0, -70.0, -50.0     # mV
Cm = 1.0                               # uF/cm^2
J = 0.0                                # no adjustment current in item (d)

# ------------------ Helpers ---------------------
def sech(x):  # numerical helper
    return 1.0 / np.cosh(x)

def m_inf(V):
    # unchanged from previous items
    return 0.5 * (1.0 + np.tanh((V + 1.0) / 15.0))

def n_inf(V):
    # modified n_infty
    return 0.5 * (1.0 + np.tanh((V - 10.0) / 14.5))

def tau_n(V):
    # modified tau_n
    return 3.0 / np.cosh((V - 10.0) / 29.0)  # = 3 * sech((V-10)/29)

# derivatives for the Jacobian
def m_inf_prime(V):
    # d/dV tanh((V+1)/15) = sech^2((V+1)/15) * (1/15)
    return 0.5 * sech((V + 1.0) / 15.0)**2 / 15.0  # = 1/30 * sech^2

def n_inf_prime(V):
    # 0.5 * (1/14.5) * sech^2((V-10)/14.5) = 1/29 * sech^2(...)
    return (1.0 / 29.0) * sech((V - 10.0) / 14.5)**2

# ------------------ Vector field ----------------
def F(V, n):
    # dV/dt (times Cm)
    return (-gCa * m_inf(V) * (V - ECa)
            - gK * n * (V - EK)
            - gL * (V - EL)
            + J) / Cm

def G(V, n):
    # dn/dt
    return (n_inf(V) - n) / tau_n(V)

# nullclines
def n_nullcline(V): return n_inf(V)
def V_nullcline(V):
    # dV/dt = 0 -> solve for n(V)
    return (-gCa * m_inf(V) * (V - ECa) - gL * (V - EL) + J) / (gK * (V - EK))

# ------------------ Fixed points (all intersections) ---------------
def fixed_points():
    # find all sign changes of H(V) = V_nullcline(V) - n_inf(V)
    Vs = np.linspace(-80, 60, 40001)
    H = V_nullcline(Vs) - n_inf(Vs)
    mask = np.abs(Vs - EK) > 1e-2  # skip the vertical asymptote at V=EK
    idxs = np.where(mask[:-1] & mask[1:] & (np.sign(H[:-1]) * np.sign(H[1:]) < 0))[0]

    roots = []
    for idx in idxs:
        a, b = Vs[idx], Vs[idx+1]
        # bisection to high precision
        for _ in range(60):
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
    # Build Jacobian at (Vs, ns) and classify by trace/determinant
    FV = (-gCa * (m_inf_prime(Vs) * (Vs - ECa) + m_inf(Vs))
          - gK * ns - gL) / Cm
    Fn = (-gK * (Vs - EK)) / Cm
    GV = n_inf_prime(Vs) / tau_n(Vs)       # since n = n_inf(V) at equilibrium
    Gn = -1.0 / tau_n(Vs)

    Jmat = np.array([[FV, Fn], [GV, Gn]])
    tr = np.trace(Jmat)
    det = np.linalg.det(Jmat)
    disc = tr**2 - 4*det

    if det < 0:
        kind = "sela (instável)"
    else:
        if disc > 0:
            # real eigenvalues
            kind = "nó estável" if tr < 0 else "nó instável"
        elif disc < 0:
            # complex (focus)
            kind = "foco estável" if tr < 0 else "foco instável"
        else:
            kind = "limiar (traço=±2√Δ)"
    # eigenvalues (for completeness)
    eigs = np.linalg.eigvals(Jmat)
    return kind, tr, det, eigs

# ------------------ Compute and print fixed points ------------------
fps = fixed_points()
print(f"Número de pontos fixos encontrados: {len(fps)}")
for i, (Vst, nst) in enumerate(fps, 1):
    kind, tr, det, eigs = classify_point(Vst, nst)
    print(f"Eq.{i}: V*={Vst:.3f} mV, n*={nst:.4f}  |  tipo: {kind}  "
          f"|  traço={tr:.4f}, det={det:.4f}, autovalores={eigs}")

# ------------------ Plot nullclines + fixed points ------------------
Vgrid = np.linspace(-80, 60, 3000)
mask_left  = Vgrid < EK - 1e-2
mask_right = Vgrid > EK + 1e-2

plt.figure(figsize=(7.2, 5.2))
plt.plot(Vgrid[mask_left],  V_nullcline(Vgrid[mask_left]),  'k-', lw=2, label="Nulclina de V")
plt.plot(Vgrid[mask_right], V_nullcline(Vgrid[mask_right]), 'k-', lw=2)
plt.plot(Vgrid, n_nullcline(Vgrid), 'k--', lw=2, label="Nulclina de n")

# mark fixed points
for i, (Vst, nst) in enumerate(fps, 1):
    plt.scatter([Vst], [nst], s=60, zorder=5)
    plt.text(Vst+1.5, nst+0.02, f"Eq.{i}", fontsize=10)

plt.xlim(-80, 60)
plt.ylim(-.2, 1)
plt.xlabel("V (mV)")
plt.ylabel("n")
plt.title("Morris–Lecar modificado: nulclinas no plano n × V (J=0)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_2d.png", dpi=300, bbox_inches="tight")
# plt.show()
