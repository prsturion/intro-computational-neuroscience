import numpy as np
import matplotlib.pyplot as plt

# Import framework components
# Note the use of the corrected initial values file
from parameters_ex_6 import PARAMS
from initial_values_ex_6 import Y0
from utils_ex_6 import rhs_full, integrate_rk4

def detect_bursts(t: np.ndarray, Vd: np.ndarray, thr_start: float = 0.0, thr_end: float = -0.050) -> np.ndarray:
    """
    Detects the start times of somatic bursts based on dendritic voltage criteria.
    A burst starts when Vd crosses thr_start upwards and ends when Vd falls below thr_end.
    
    Args:
        t (np.ndarray): Time vector [s].
        Vd (np.ndarray): Dendritic membrane potential vector [V].
        thr_start (float): Dendritic voltage threshold for burst initiation [V].
        thr_end (float): Dendritic voltage threshold for burst termination [V].

    Returns:
        np.ndarray: Array of burst start times [s].
    """
    burst_starts = []
    in_burst = False
    for i in range(1, len(Vd)):
        # Condition for burst start: not in a burst and Vd crosses start threshold upwards
        if not in_burst and Vd[i-1] < thr_start and Vd[i] >= thr_start:
            burst_starts.append(t[i])
            in_burst = True
        # Condition for burst end: in a burst and Vd crosses end threshold downwards
        elif in_burst and Vd[i-1] >= thr_end and Vd[i] < thr_end:
            in_burst = False
    return np.asarray(burst_starts, dtype=float)

# --- Simulation setup ---
t0, tf = 0.0, 6.0   # seconds
h = 3e-5            # time step [s]

# This current is used to elicit bursting behavior
I_inj_soma = 0*200e-12 # 200 pA
PARAMS["Iinj_S"] = I_inj_soma
PARAMS["Iinj_D"] = 0.0

# Values for Gh to be tested
gh_values = [0e-9, 5e-9, 10e-9, 15e-9]

# --- Plotting setup ---
fig, axes = plt.subplots(len(gh_values), 2, figsize=(10, 12), sharey='row')
fig.suptitle(r'Efeito da Condutância $\bar{G}_h$ no Disparo em Rajadas', fontsize=16)

# --- Main loop for simulation and plotting ---
for i, gh in enumerate(gh_values):
    print(f"Simulating with gh = {gh*1e9:.1f} nS...")
    
    # Update the parameter for the current simulation run
    PARAMS["gh"] = gh
    
    # Integrate the ODE system
    t, Y = integrate_rk4(rhs_full, (t0, tf), Y0, h, PARAMS) # type: ignore
    Vs = Y[:, 0]  # Soma voltage [V]
    Vd = Y[:, 1]  # Dendrite voltage [V]
    
    # --- Left Column Plot (Vs from 2s to 6s) ---
    ax_left = axes[i, 0]
    ax_left.plot(t, Vs * 1e3, lw=1.0, color='forestgreen')
    ax_left.set_xlim(2.0, 6.0)
    ax_left.set_ylabel(r'$V_s$ (mV)')
    ax_left.grid(True, linestyle='--', alpha=0.5)
    
    
    # --- Right Column Plot (Zoom on penultimate burst) ---
    ax_right = axes[i, 1]
    
    # Detect all burst start times
    burst_times = detect_bursts(t, Vd)
    
    # Check if at least two bursts were detected
    if len(burst_times) >= 2:
        t_penultimate = burst_times[-2] # Time of the penultimate burst start
        
        # Define the time window for the zoom plot
        t_start_zoom = t_penultimate - 0.025
        t_end_zoom = t_penultimate + 0.025
        
        # Find the indices corresponding to this window
        idx_zoom = (t >= t_start_zoom) & (t <= t_end_zoom)
        
        # Plot Vs in the window, with time centered at 0 ms
        t_zoom_ms = (t[idx_zoom] - t_penultimate) * 1e3
        Vs_zoom_mv = Vs[idx_zoom] * 1e3
        
        ax_right.plot(t_zoom_ms, Vs_zoom_mv, lw=1.2, color='forestgreen')
        ax_right.set_xlim(-25, 25)
        ax_right.set_title(fr'$\bar{{G}}_h = {gh*1e9:.0f}$ nS', loc='right', y=0.75)
    else:
        # If not enough bursts are found, display a message on the plot
        ax_right.text(0.5, 0.5, 'Penúltima rajada\nnão encontrada', 
                      ha='center', va='center', transform=ax_right.transAxes)

    ax_right.grid(True, linestyle='--', alpha=0.5)

# --- Final adjustments for labels ---
axes[-1, 0].set_xlabel('Tempo (s)')
axes[-1, 1].set_xlabel('Tempo (ms)')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.savefig("intro-computational-neuroscience/list_4/figures/ex_6b.png", dpi=300)
# To display the plot, uncomment the line below
# plt.show()