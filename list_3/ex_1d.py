import numpy as np
import matplotlib.pyplot as plt
import utils_A as ut  # expects rk4_solve, step_current_protocol, connor_stevens_rhs

# --------- Helpers ---------
def first_spike_latency_ms(t: np.ndarray,
                           V: np.ndarray,
                           stim_onset_ms: float = 60.0,
                           threshold_mV: float = 0.0,
                           refractory_ms: float = 2.0) -> float:
    """
    Return latency (ms) from stimulus onset to the first *upward* threshold crossing.
    If no spike occurs, return np.inf.
    Uses linear interpolation between samples for sub-step timing.
    """
    # Only analyze the post-stimulus window
    mask = t >= stim_onset_ms
    t_w = t[mask]
    V_w = V[mask]
    if t_w.size < 2:
        return np.inf

    # Find all upward crossings
    up = np.where((V_w[:-1] < threshold_mV) & (V_w[1:] >= threshold_mV))[0]
    if up.size == 0:
        return np.inf

    # Enforce a refractory window to avoid counting the same spike twice
    dt = t[1] - t[0]
    refr_idx = int(np.round(refractory_ms / dt))
    idx0 = up[0]
    for k in up[1:]:
        if k - idx0 >= refr_idx:
            idx0 = k
            break  # but for latency we only need the very first spike

    # Linear interpolation for precise crossing time
    t1, t2 = t_w[idx0], t_w[idx0 + 1]
    v1, v2 = V_w[idx0], V_w[idx0 + 1]
    if v2 == v1:
        t_cross = t1
    else:
        frac = (threshold_mV - v1) / (v2 - v1)
        t_cross = t1 + frac * (t2 - t1)

    return float(t_cross - stim_onset_ms)

def simulate_once(J_amp_uAcm2: float,
                  T_ms: float = 200.0,
                  dt_ms: float = 0.01):
    """
    Simulate Connor–Stevens with a step current J from 60 ms to T_ms.
    Returns (t, Y) with columns [V, n, m, h, a, b].
    """
    y0 = np.array([-67.976, 0.1558, 0.01, 0.965, 0.5404, 0.2885], dtype=float)
    J_of_t = ut.step_current_protocol(t_on=60.0, t_off=T_ms, amp=J_amp_uAcm2)
    t, Y = ut.rk4_solve(ut.connor_stevens_rhs, 0.0, T_ms, dt_ms, y0, J_of_t) # type: ignore
    return t, Y

# --------- Sweep and plot f–I (latency method) ---------
if __name__ == "__main__":
    I_vals = np.round(np.arange(8.0, 100.0 + 1e-12, 0.2), 3)  # 8 → 10 µA/cm² step 0.2
    f_vals_hz = []

    for J in I_vals:
        t, Y = simulate_once(J, T_ms=200.0, dt_ms=0.01)
        V = Y[:, 0]

        lat_ms = first_spike_latency_ms(t, V, stim_onset_ms=60.0,
                                        threshold_mV=0.0, refractory_ms=2.0)
        # f = 1 / latency. If no spike, set f = 0.
        f_hz = 0.0 if not np.isfinite(lat_ms) or lat_ms <= 0 else 1000.0 / lat_ms
        f_vals_hz.append(f_hz)

    # Plot and SAVE (no directory checks; fixed path as requested)
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(I_vals, f_vals_hz, linewidth=1.5)
    plt.xlabel("Injected current density J (µA/cm²)")
    plt.ylabel("Firing rate f = 1/latency (Hz)")
    # plt.title("Connor–Stevens f–I via first-spike latency (step: 60–200 ms)")
    plt.grid(True, alpha=0.4)
    plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_1d.png", dpi=300)
