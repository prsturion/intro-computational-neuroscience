import numpy as np
import matplotlib.pyplot as plt
import utils_A as ut 

# ---------- Spike counting ----------
def count_spikes(t: np.ndarray, V: np.ndarray,
                 threshold_mV: float = 0.0,
                 refractory_ms: float = 2.0) -> int:
    """
    Count spikes by upward crossings of 'threshold_mV' with a refractory window.
    Returns the number of spikes in V(t).
    """
    dt = t[1] - t[0]
    refr_idx = int(np.round(refractory_ms / dt))
    crossings = np.where((V[:-1] < threshold_mV) & (V[1:] >= threshold_mV))[0]

    # Enforce refractory period to avoid double counting
    if crossings.size == 0:
        return 0
    count = 1
    last = crossings[0]
    for idx in crossings[1:]:
        if idx - last >= refr_idx:
            count += 1
            last = idx
    return count

# ---------- Single simulation ----------
def simulate_connor_stevens(J_amp_uAcm2: float,
                            T_ms: float = 200.0,
                            dt_ms: float = 0.01):
    """
    Simulate the Connor–Stevens model for a step current J starting at t=60 ms.
    Returns (t, Y) where Y columns are [V, n, m, h, a, b].
    """
    # Initial conditions (Ermentrout)
    y0 = np.array([-67.976, 0.1558, 0.01, 0.965, 0.5404, 0.2885], dtype=float)

    # Current protocol: 0 until 60 ms, then J until T_ms
    J_of_t = ut.step_current_protocol(t_on=60.0, t_off=T_ms, amp=J_amp_uAcm2)

    # Integrate with generic RK4 solver from utils_A.py
    t, Y = ut.rk4_solve(ut.connor_stevens_rhs, 0.0, T_ms, dt_ms, y0, J_of_t) # type: ignore

    return t, Y

# ---------- Sweep and build f–I ----------
if __name__ == "__main__":
    # Current sweep: 8 to 10 µA/cm² in steps of 0.2
    I_vals = np.round(np.arange(8.0, 100.0 + 1e-9, 0.2), 3)

    f_vals_hz = []      # spikes per second (Hz)
    f_vals_per_ms = []  # spikes per ms (as asked: spikes/140 ms)

    for J in I_vals:
        t, Y = simulate_connor_stevens(J_amp_uAcm2=J, T_ms=200.0, dt_ms=0.01)
        V = Y[:, 0]

        # Count spikes only within the stimulation window [60 ms, 200 ms)
        stim_mask = (t >= 60.0)
        spikes = count_spikes(t[stim_mask], V[stim_mask],
                              threshold_mV=0.0, refractory_ms=2.0)

        # Rate as specified (spikes / 140 ms)
        rate_per_ms = spikes / 140.0
        f_vals_per_ms.append(rate_per_ms)

        # Also compute in Hz for convenience (spikes per second)
        f_vals_hz.append(spikes / 0.140)

    I_vals = np.asarray(I_vals)
    f_vals_hz = np.asarray(f_vals_hz)
    f_vals_per_ms = np.asarray(f_vals_per_ms)

    # ---------- Plot and save (no directory checks, fixed path) ----------
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(I_vals, f_vals_hz, linewidth=1.5)
    plt.xlabel("Injected current density J (µA/cm²)")
    plt.ylabel("Firing rate f (Hz)")
    plt.title("Connor–Stevens f–I curve (step: 60–200 ms)")
    plt.grid(True, alpha=0.4)

    # Save
    plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_1c.png", dpi=300)

    # If you also want the exact definition from the statement (spikes per ms),
    # uncomment the block below to save an alternative plot:
    """
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(I_vals, f_vals_per_ms, marker="o", linewidth=1.5)
    plt.xlabel("Injected current density J (µA/cm²)")
    plt.ylabel("Firing rate f (spikes/ms)")
    plt.title("Connor–Stevens f–I curve (spikes per 140 ms)")
    plt.grid(True, alpha=0.4)
    plt.savefig("./intro-computational-neuroscience/list_3/figures/fi_connor_stevens_spikes_per_ms.png", dpi=300)
    """
