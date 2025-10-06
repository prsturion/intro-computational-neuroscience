
import numpy as np
import matplotlib.pyplot as plt

from parameters import PARAMS
from initial_values import Y0
from utils import rhs_full, integrate_rk4, chi

# ----------------------------
# Simulation setup
# ----------------------------
t0, tf = 0.0, 2.0       # seconds
h = 3e-5            # time step (reduce if you see numerical artifacts)

# (optional) modify injected currents here:
def Iinj(t):
    I = 0
    if 0.3 <= t <= 0.33:
        return I
    elif .97 <= t <= 1:
        return I
    elif 1.5 <= t <= 1.53:
        return I
    else:
        return 0
    
PARAMS["Iinj_S"] = 0 # type: ignore
PARAMS["Iinj_D"] = Iinj # type: ignore

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
plt.savefig("intro-computational-neuroscience/list_4/figures/ex_2_1.png", dpi=150)
plt.show()



# Unpack states we need for currents and [Ca]
Ca   = Y[:, 2]    # [M]
mca  = Y[:, 6]
mkca = Y[:, 7]
mkahp = Y[:, 8]

# Coupling current: Ic > 0 when Vd > Vs
Ic = PARAMS["gc"] * (Vd - Vs)                 # [A]

# Dendritic K currents
IKCa  = PARAMS["gKCa"]  * mkca  * chi(Ca) * (Vd - PARAMS["EK"])  # [A]
IKahp = PARAMS["gKahp"] * mkahp *           (Vd - PARAMS["EK"])  # [A]

# Time windows (as in the caption): 
# A,B: 0.86–0.91 s (zoom on voltages and Ic)
# C,D: 0.85–1.15 s ([Ca] and K-currents during a burst)
tA0, tA1 = 0.86, 0.91
tC0, tC1 = 0.85, 1.15
winA = (t >= tA0) & (t <= tA1)
winC = (t >= tC0) & (t <= tC1)

# Unit conversions for plotting
mV  = 1e3
nA  = 1e9
mM  = 1e3

fig2, axs = plt.subplots(2, 2, figsize=(9.2, 6.2))
(axA, axC), (axB, axD) = axs   # layout to match A/C top, B/D bottom

# Panel A: Vs (solid) and Vd (dashed), zoomed
axA.plot(t[winA], (Vs[winA] * mV), lw=1.2, label=r"$V_s$")
axA.plot(t[winA], (Vd[winA] * mV), lw=1.2, ls="--", label=r"$V_d$")
axA.set_ylabel("V (mV)")
axA.set_title("A", loc="left")
axA.grid(True, ls="--", alpha=0.35)
axA.legend(loc="upper right", frameon=False, fontsize=9)

# Panel B: coupling current Ic, same zoom as A
axB.plot(t[winA], (Ic[winA] * nA), lw=1.2)
axB.set_xlabel("Time (s)")
axB.set_ylabel(r"$I_c$ (nA)")
axB.set_title("B", loc="left")
axB.grid(True, ls="--", alpha=0.35)

# Panel C: calcium concentration [Ca], wider window
axC.plot(t[winC], (Ca[winC] * mM), lw=1.2)
axC.set_ylabel(r"[Ca] (mM)")
axC.set_title("C", loc="left")
axC.grid(True, ls="--", alpha=0.35)

# Panel D: dendritic K-currents; IKCa dotted; 100*IKahp dashed
axD.plot(t[winC], (IKCa[winC] * nA), lw=1.2, ls=":",  label=r"$I_{KCa}$")
axD.plot(t[winC], (100.0 * IKahp[winC] * nA), lw=1.2, ls="--", label=r"$100\times I_{KAHp}$")
axD.set_xlabel("Time (s)")
axD.set_ylabel("Dendritic K-currents (nA)")
axD.set_title("D", loc="left")
axD.grid(True, ls="--", alpha=0.35)
axD.legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_4/figures/ex_2_2.png", dpi=150)
# plt.show()  # optional, if you want to show it immediately