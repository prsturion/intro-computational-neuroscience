# lif_cv_isi_fig.py
# LIF with Gaussian white noise (Euler–Maruyama).
# Single 2x2 figure mimicking Dayan & Abbott Fig. 5.21:
# (A) above rheobase (regular) — top: no reset, bottom: with spikes
# (B) below rheobase (irregular) — top: no reset, bottom: with spikes
# Spike lines are dashed and extended up to a configurable "spike peak".

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Model parameters (SI units)
# ----------------------------
V_rest  = -70e-3   # V
R       = 40e6     # Ohm
C       = 250e-12  # F
V_th    = -50e-3   # V (threshold)
V_reset = -65e-3   # V (reset)
t_ref   = 2e-3     # s (absolute refractory)

dt      = 0.01e-3  # s (0.01 ms)
T_total = 2.0      # s (simulate for 2 s, as requested)
tau     = R * C

# ----------------------------
# Noise and plotting settings
# ----------------------------
sigma = 0.04                   # V / sqrt(s)  (increase/decrease as needed)
SPIKE_PEAK = -10e-3           # V (top of dashed spike lines in bottom plots)
SHOW_WINDOW = 1.0             # s (plot only first 1000 ms like the book figure)

# Currents: one above and one below rheobase
I_rheo   = (V_th - V_rest) / R
I_acima  = 570e-12            # A (example suggested)
I_abaixo = 430e-12            # A (example suggested)

# Seeds for reproducibility (different per trace)
SEED_FREE_ABOVE  = 12345
SEED_SPIKE_ABOVE = 12346
SEED_FREE_BELOW  = 67890
SEED_SPIKE_BELOW = 67891

# ----------------------------
# Euler–Maruyama integrators
# ----------------------------
def simulate_free(I, sigma, dt, T, rng):
    """No reset/refractory; used for 'top' panels."""
    n = int(np.floor(T / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    V = np.empty(n + 1)
    V[0] = V_rest
    for k in range(n):
        drift = ((V_rest - V[k]) / R + I) / C
        noise = sigma * np.sqrt(dt) * rng.normal()
        V[k + 1] = V[k] + dt * drift + noise
    return t, V

def simulate_with_spikes(I, sigma, dt, T, rng):
    """With reset and absolute refractory; returns times, voltages, and spike times."""
    n = int(np.floor(T / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    V = np.empty(n + 1)
    V[0] = V_rest
    spikes = []
    ref_left = 0.0

    for k in range(n):
        if ref_left > 0.0:
            V[k + 1] = V_reset
            ref_left -= dt
            continue

        drift = ((V_rest - V[k]) / R + I) / C
        noise = sigma * np.sqrt(dt) * rng.normal()
        V_next = V[k] + dt * drift + noise

        # threshold crossing from below with linear interpolation for t*
        if (V[k] < V_th) and (V_next >= V_th):
            denom = (V_next - V[k])
            frac  = (V_th - V[k]) / denom if denom != 0.0 else 1.0
            frac  = float(np.clip(frac, 0.0, 1.0))
            t_star = t[k] + dt * frac
            spikes.append(t_star)
            V[k + 1] = V_reset
            ref_left = t_ref
        else:
            V[k + 1] = V_next

    return t, V, np.asarray(spikes)

def isi_and_cv(spike_times):
    """Compute ISIs and CV_ISI (std/mean)."""
    if spike_times.size < 2:
        return np.array([]), np.nan
    isis = np.diff(spike_times)
    mean = float(np.mean(isis))
    std  = float(np.std(isis, ddof=1)) if isis.size > 1 else 0.0
    cv   = std / mean if mean > 0 else np.nan
    return isis, cv

# ----------------------------
# Build the 2x2 figure
# ----------------------------
def main():
    # Simulate all four traces
    tA_top,  VA_top  = simulate_free(I_acima,  sigma, dt, T_total, np.random.default_rng(SEED_FREE_ABOVE))
    tA_bot,  VA_bot, spkA = simulate_with_spikes(I_acima,  sigma, dt, T_total, np.random.default_rng(SEED_SPIKE_ABOVE))
    tB_top,  VB_top  = simulate_free(I_abaixo, sigma, dt, T_total, np.random.default_rng(SEED_FREE_BELOW))
    tB_bot,  VB_bot, spkB = simulate_with_spikes(I_abaixo, sigma, dt, T_total, np.random.default_rng(SEED_SPIKE_BELOW))

    # Window to display (e.g., first 1000 ms)
    def window_mask(t):
        return t <= SHOW_WINDOW

    mA_top = window_mask(tA_top)
    mA_bot = window_mask(tA_bot)
    mB_top = window_mask(tB_top)
    mB_bot = window_mask(tB_bot)
    # spikes inside window
    spkA_w = spkA[spkA <= SHOW_WINDOW]
    spkB_w = spkB[spkB <= SHOW_WINDOW]

    # Compute CVs on full 2 s (for printing)
    _, cvA = isi_and_cv(spkA)
    _, cvB = isi_and_cv(spkB)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    (axA_top, axB_top), (axA_bot, axB_bot) = axes

    # Panel A (regular) — top (no reset)
    axA_top.plot(tA_top[mA_top]*1e3, VA_top[mA_top]*1e3, lw=1.0, color="k")
    axA_top.axhline(V_th*1e3, ls="--", lw=1.0, color="gray")
    axA_top.set_ylabel("V (mV)")
    axA_top.set_title("A (acima do limiar) — sem reset")
    axA_top.grid(True)

    # Panel A (regular) — bottom (with spikes)
    axA_bot.plot(tA_bot[mA_bot]*1e3, VA_bot[mA_bot]*1e3, lw=1.0, color="k")
    axA_bot.axhline(V_th*1e3, ls="--", lw=1.0, color="gray")
    if spkA_w.size > 0:
        axA_bot.vlines(spkA_w*1e3, ymin=V_th*1e3, ymax=SPIKE_PEAK*1e3,
                       colors="r", linestyles="--", linewidth=0.9)
    axA_bot.set_xlabel("t (ms)")
    axA_bot.set_ylabel("V (mV)")
    axA_bot.set_title("A (acima do limiar) — com spikes")
    axA_bot.set_ylim(-75, -8)  # similar to the book panel
    axA_bot.grid(True)

    # Panel B (irregular) — top (no reset)
    axB_top.plot(tB_top[mB_top]*1e3, VB_top[mB_top]*1e3, lw=1.0, color="k")
    axB_top.axhline(V_th*1e3, ls="--", lw=1.0, color="gray")
    axB_top.set_ylabel("V (mV)")
    axB_top.set_title("B (abaixo do limiar) — sem reset")
    axB_top.grid(True)

    # Panel B (irregular) — bottom (with spikes)
    axB_bot.plot(tB_bot[mB_bot]*1e3, VB_bot[mB_bot]*1e3, lw=1.0, color="k")
    axB_bot.axhline(V_th*1e3, ls="--", lw=1.0, color="gray")
    if spkB_w.size > 0:
        axB_bot.vlines(spkB_w*1e3, ymin=V_th*1e3, ymax=SPIKE_PEAK*1e3,
                       colors="r", linestyles="--", linewidth=0.9)
    axB_bot.set_xlabel("t (ms)")
    axB_bot.set_ylabel("V (mV)")
    axB_bot.set_title("B (abaixo do limiar) — com spikes")
    axB_bot.set_ylim(-75, -8)
    axB_bot.grid(True)

    fig.suptitle(f"Modos regular (A) e irregular (B) — σ={sigma:g}  |  CV_A={cvA:.3f}, CV_B={cvB:.3f}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore
   #  plt.show()
    plt.savefig(f"intro-computational-neuroscience/list_6/figures/ex_2c.png", dpi=300, bbox_inches="tight")

    # Also print CVs in the console
    print(f"CV_ISI (acima)  = {cvA:.3f}")
    print(f"CV_ISI (abaixo) = {cvB:.3f}")

if __name__ == "__main__":
    main()
