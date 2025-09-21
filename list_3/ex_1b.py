
import os
import numpy as np
import matplotlib.pyplot as plt
import utils_A as ut

def simulate_once(J_amp: float, T_ms: float = 200.0, dt_ms: float = 0.01):
    # Initial conditions from the statement (Ermentrout)
    y0 = np.array([
        -67.976,   # V(0) mV
        0.1558,    # n(0)
        0.01,      # m(0)
        0.965,     # h(0)
        0.5404,    # a(0)
        0.2885,    # b(0)
    ], dtype=float)

    # Current protocol: inject from t=60 ms to end
    J_func = ut.step_current_protocol(t_on=60.0, t_off=T_ms, amp=J_amp)

    # Integrate with generic solver
    t, Y = ut.rk4_solve(ut.connor_stevens_rhs, 0.0, T_ms, dt_ms, y0, J_func) #type: ignore

    return t, Y  # Y columns: [V, n, m, h, a, b]

def save_plots(t: np.ndarray, Y: np.ndarray, J_amp: float, f_path: str):

    labels = ["V (mV)", "n", "m", "h", "a", "b"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, Y[:, i])
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("t (ms)")
    fig.suptitle(f"J = {J_amp} µA/cm²")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plt.savefig(f_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    for i, J in enumerate([5.0, 10.0, 15.0, 20.0]):
        t, Y = simulate_once(J_amp=J, T_ms=200.0, dt_ms=0.01)
        save_plots(t, Y, J_amp=J, f_path=f"./intro-computational-neuroscience/list_3/figures/ex_1b_{i+1}.png")
