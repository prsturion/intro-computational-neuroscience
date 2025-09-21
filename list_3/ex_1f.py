import numpy as np
import matplotlib.pyplot as plt
import utils_A as ut  # expects: rk4_solve, connor_stevens_rhs, gating funcs, constants

# ----- Modified RHS: scale db/dt by 0.25 (slower inactivation of I_A) -----
def connor_stevens_rhs_slow_b(t: float, y: np.ndarray, J_of_t) -> np.ndarray:
    """
    Same as ut.connor_stevens_rhs, but the 'b' equation is slowed down:
    db/dt <- 0.25 * db/dt
    """
    V, n, m, h, a, b = y

    # Ionic currents
    Iion = (
        ut.I_Na(V, m, h) +
        ut.I_K(V, n) +
        ut.I_A(V, a, b) +
        ut.I_L(V)
    )

    # Membrane equation
    dVdt = -(Iion - J_of_t(t)) / ut.C_m

    # Gating kinetics
    dndt = (ut.n_inf(V) - n) / ut.tau_n(V)
    dmdt = (ut.m_inf(V) - m) / ut.tau_m(V)
    dhdt = (ut.h_inf(V) - h) / ut.tau_h(V)
    dadt = (ut.a_inf(V) - a) / ut.tau_a(V)

    # <-- key change here: slow inactivation of A-current (b gate)
    dbdt = 0.25 * (ut.b_inf(V) - b) / ut.tau_b(V)

    return np.array([dVdt, dndt, dmdt, dhdt, dadt, dbdt], dtype=float)

# ----- Simulation setup -----
T_ms = 200.0
dt_ms = 0.01

# Initial conditions (Ermentrout)
y0 = np.array([
    -67.976,  # V(0) mV
    0.1558,   # n(0)
    0.01,     # m(0)
    0.965,    # h(0)
    0.5404,   # a(0)
    0.2885,   # b(0)
], dtype=float)

# Constant current J = 15 µA/cm² for the whole interval
J_const = lambda t: 15.0

# Integrate
t, Y = ut.rk4_solve(connor_stevens_rhs_slow_b, 0.0, T_ms, dt_ms, y0, J_const) # type: ignore
V = Y[:, 0]

# Plot and SAVE (no directory creation checks)
plt.figure(figsize=(9, 4.8))
plt.plot(t, V, lw=1.6)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("Connor–Stevens with slower b inactivation (db/dt × 0.25), J = 15 µA/cm²")
plt.grid(True, alpha=0.35)
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_1f.png", dpi=300)
# plt.show()
