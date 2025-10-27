# lif_dt_roughness_same_T.py
# Noisy LIF subthreshold trajectories with identical duration for all Δt.
# Euler–Maruyama integration; no reset/refractory; optional clipping below threshold.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Model parameters (SI units)
# ----------------------------
V_rest  = -70e-3   # V
R       = 40e6     # Ohm
C       = 250e-12  # F
V_th    = -50e-3   # V (threshold, used only if clipping enabled)

# Subthreshold setup
I_fixed = 300e-12  # A (choose below rheobase)
sigma   = 0.5      # V / sqrt(s)  (try 0.2 or 0.6)

# Common total duration for all traces
T_total = 0.05     # s  (e.g., 50 ms)

# Time steps to compare: Δt, Δt/10, Δt/100
dt0     = 0.01e-3  # s (0.01 ms)
DT_LIST = [dt0/i for i in [10, 100, 1e3]]

# If True, keep trajectory strictly subthreshold by clipping just below V_th
CLIP_TO_THRESHOLD = False

def simulate_em(I, sigma, dt, T, rng, clip=False):
    """
    Euler–Maruyama for dV = ((V_rest - V)/R + I)/C dt + sigma dW.
    Returns t (0..T with step dt) and V with same length.
    """
    n = int(np.round(T / dt))             # ensures identical end time T for this dt
    t = np.linspace(0.0, n * dt, n + 1)   # exactly spans [0, T]
    V = np.empty(n + 1, dtype=float)
    V[0] = V_rest

    for k in range(n):
        drift = ((V_rest - V[k]) / R + I) / C
        noise = sigma * np.sqrt(dt) * rng.normal()
        V[k + 1] = V[k] + dt * drift + noise
        if clip and V[k + 1] >= V_th:
            # keep strictly below threshold without altering duration
            V[k + 1] = np.nextafter(V_th, -np.inf)

    return t, V

def main():
    plt.figure()
    for i, dt in enumerate(DT_LIST):
        rng = np.random.default_rng(2025_10_26 + i)  # different RNG per Δt
        t, V = simulate_em(I_fixed, sigma, dt, T_total, rng, clip=CLIP_TO_THRESHOLD)
        plt.plot(t * 1e3, V * 1e3, label=f"Δt = {dt*1e3:.5f} ms (N={len(t)-1})")

    # Portuguese labels on the figure
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Voltagem V (mV)")
    plt.title("Trajetórias V(t) para diferentes passos Δt (mesma duração)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_6/figures/ex_2b.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
