import numpy as np
import matplotlib.pyplot as plt
import utils_A as ut  # must provide: rk4_solve, connor_stevens_rhs, piecewise_constant_protocol

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

# ----- Current protocol -----
knots = [0.0, 60.0, 65.0, T_ms]
amps  = [0.0, -50.0, 20.0]
J_of_t = ut.piecewise_constant_protocol(knots, amps)

# ----- Integrate -----
t, Y = ut.rk4_solve(ut.connor_stevens_rhs, 0.0, T_ms, dt_ms, y0, J_of_t) # type: ignore
V = Y[:, 0]
J = np.array([J_of_t(tt) for tt in t])

# ----- Plots -----
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={'height_ratios': [1, 2]})

# Current protocol
axes[0].plot(t, J, color="tab:red", linewidth=1.5)
axes[0].set_ylabel("J (µA/cm²)")
# axes[0].set_title("Current protocol and membrane response")
axes[0].grid(True, alpha=0.3)

# Membrane potential
axes[1].plot(t, V, color="tab:blue", linewidth=1.5)
axes[1].set_xlabel("t (ms)")
axes[1].set_ylabel("V (mV)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_1e.png", dpi=300)
# plt.show()
