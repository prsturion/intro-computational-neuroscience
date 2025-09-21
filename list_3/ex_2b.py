import numpy as np
import matplotlib.pyplot as plt
import utils_T as ut

# -----------------------------
# Grids
# -----------------------------
base_pA_grid = np.arange(-200.0, 220.0, 40.0)
step_pA_grid = np.arange(0.0, 120.0, 20.0)

T_MS, DT_MS = 750.0, 0.01
t_on_ms, t_off_ms = 250.0, 500.0
thr_mV = 0.0

# -----------------------------
# Spike detection helper
# -----------------------------
def upcrossing_times_ms(t_s, V_V, thr_V):
    idx = np.where((V_V[:-1] < thr_V) & (V_V[1:] >= thr_V))[0]
    if idx.size == 0:
        return np.empty(0)
    t1, t2 = t_s[idx], t_s[idx+1]
    v1, v2 = V_V[idx], V_V[idx+1]
    denom = np.where(np.abs(v2-v1)<1e-15, 1e-15, v2-v1)
    frac = (thr_V - v1)/denom
    return (t1 + frac*(t2-t1)) / ut.ms

def simulate(base_pA, step_pA):
    J_of_t = ut.base_plus_step_protocol(base_pA, step_pA,
                                        t_on_ms, t_off_ms, T_MS)
    y0 = ut.initial_state_at_rest()
    t_s, Y = ut.rk4_solve(ut.thalamic_rhs, 0.0, T_MS*ut.ms, DT_MS*ut.ms, y0, J_of_t) # type: ignore
    return t_s/ut.ms, Y

# -----------------------------
# Sweep
# -----------------------------
spike_count = np.zeros((len(base_pA_grid), len(step_pA_grid)))
min_ISI = np.full_like(spike_count, np.nan, dtype=float)

thr_V = thr_mV*ut.mV

for ib, base in enumerate(base_pA_grid):
    for is_, step in enumerate(step_pA_grid):
        t_ms, Y = simulate(base, step)
        V = Y[:,0]
        # Restrict to step window
        mask = (t_ms>=t_on_ms)&(t_ms<t_off_ms)
        ts = upcrossing_times_ms(t_ms[mask]*ut.ms, V[mask], thr_V)
        spike_count[ib,is_] = len(ts)
        if len(ts)>=2:
            isi = np.diff(ts)
            min_ISI[ib,is_] = np.min(isi)

# -----------------------------
# 2D Heatmaps
# -----------------------------
X, Ygrid = np.meshgrid(step_pA_grid, base_pA_grid)

plt.figure(figsize=(7,5))
plt.pcolormesh(X, Ygrid, spike_count, shading="nearest")
plt.colorbar(label="# spikes")
plt.xlabel("Step current (pA)")
plt.ylabel("Base current (pA)")
plt.title("Number of spikes (250–500 ms)")
plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_2b_spikecount_heatmap.png", dpi=300)

plt.figure(figsize=(7,5))
plt.pcolormesh(X, Ygrid, min_ISI, shading="nearest")
plt.colorbar(label="Min ISI (ms)")
plt.xlabel("Step current (pA)")
plt.ylabel("Base current (pA)")
plt.title("Minimum ISI (250–500 ms)")
plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_2b_minISI_heatmap.png", dpi=300)

# -----------------------------
# 3D surfaces
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Ygrid, spike_count, cmap="viridis")
ax.set_xlabel("Step (pA)")
ax.set_ylabel("Base (pA)")
ax.set_zlabel("# spikes")
ax.set_title("3D surface: spike count")
plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_2b_spikecount_surface.png", dpi=300)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Ygrid, min_ISI, cmap="viridis")
ax.set_xlabel("Step (pA)")
ax.set_ylabel("Base (pA)")
ax.set_zlabel("Min ISI (ms)")
ax.set_title("3D surface: min ISI")
plt.tight_layout()
plt.savefig("./intro-computational-neuroscience/list_3/figures/ex_2b_minISI_surface.png", dpi=300)

# -----------------------------
# Sample traces for distinct regimes
# -----------------------------
cases = [
    (-200, 100),  # strongly hyperpolarized base
    (0, 40),      # moderate excitation
    (200, 80),    # strongly depolarized base
]

for base, step in cases:
    t_ms, Y = simulate(base, step)
    V_mV = Y[:,0]/ut.mV
    n, h, hT = Y[:,2], Y[:,1], Y[:,3]

    fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
    axes[0].plot(t_ms, V_mV); axes[0].set_ylabel("V (mV)")
    axes[1].plot(t_ms, n); axes[1].set_ylabel("n")
    axes[2].plot(t_ms, h); axes[2].set_ylabel("h")
    axes[3].plot(t_ms, hT); axes[3].set_ylabel("h_T"); axes[3].set_xlabel("t (ms)")
    fig.suptitle(f"Base={base} pA, Step={step} pA")
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./intro-computational-neuroscience/list_3/figures/ex_2b_traces_base{base}_step{step}.png", dpi=300)
