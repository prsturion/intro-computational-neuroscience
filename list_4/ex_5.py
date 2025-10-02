import numpy as np
import matplotlib.pyplot as plt

from parameters import PARAMS
from initial_values import Y0
from utils import rhs_full, integrate_rk4

# ----------------------------
# Simulation setup
# ----------------------------
t0, tf = 0.0, 2.0       # seconds
h = 3e-5            # time step (reduce if you see numerical artifacts)

# (optinal) modify parameters here:
PARAMS["gc"] = 50e-9 # nS
PARAMS["k"] = PARAMS["k"] * 2

# (optional) modify injected currents here:    
Iinj = 200 # pA
PARAMS["Iinj_S"] = Iinj * 1e-12 # type: ignore
PARAMS["Iinj_D"] = 0 * 1e-12 # type: ignore

# ----------------------------
# Integrate with RK4 (fixed step)
# ----------------------------
t, Y = integrate_rk4(rhs_full, (t0, tf), Y0, h, PARAMS) #type: ignore
Vs = Y[:, 0]  # soma voltage [V]
Vd = Y[:, 1]  # dendrite voltage [V]

# ----------------------------
# Somatic spike detection (Miller)
#   - upward crossing of -10 mV
#   - refractory until Vs < -30 mV
# ----------------------------
def detect_spikes(t, Vs, thr_up=-0.010, thr_reset=-0.030):
    """Return spike times using hysteresis: upward crossing at thr_up,
    then wait until Vs < thr_reset before accepting another spike."""
    spikes = []
    ready = True
    for i in range(1, len(Vs)):
        if ready and (Vs[i-1] < thr_up) and (Vs[i] >= thr_up):
            spikes.append(t[i])
            ready = False
        if not ready and (Vs[i] <= thr_reset):
            ready = True
    return np.asarray(spikes, dtype=float)

spike_times = detect_spikes(t, Vs, thr_up=-0.010, thr_reset=-0.030)

# Spike metrics
if len(spike_times) >= 2:
    isi = np.diff(spike_times)
    mean_isi = float(np.mean(isi))
    mean_freq = 1.0 / mean_isi
else:
    isi = np.array([])
    mean_isi = np.nan
    mean_freq = 0.0

print(f"# somatic spikes: {len(spike_times)}")
if len(isi) > 0:
    print(f"Mean ISI: {mean_isi:.4f} s | Mean freq: {mean_freq:.2f} Hz")

# ----------------------------
# Plots (match the style of the statement’s figure)
# ----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
plt.suptitle(f"$Iinj = {Iinj}$ pA")

# Panel A: Vs (soma)
ax1.plot(t, Vs * 1e3, lw=1.2)
ax1.set_ylabel(r"$V_s$ (mV)")
ax1.set_title("A", loc="left")
ax1.grid(True, ls="--", alpha=0.4)
# mark spikes
for ts in spike_times:
    continue
    ax1.axvline(ts, color="k", ls=":", lw=0.9, alpha=0.7)

# Panel B: Vd (dendrite)
ax2.plot(t, Vd * 1e3, lw=1.2)
ax2.set_ylabel(r"$V_d$ (mV)")
ax2.set_xlabel("Time (s)")
ax2.set_title("B", loc="left")
ax2.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"intro-computational-neuroscience/list_4/figures/ex_5_I{Iinj}_soma.png", dpi=150)

# plt.show()
