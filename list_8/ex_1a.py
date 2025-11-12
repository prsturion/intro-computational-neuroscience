
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    EIONetwork, rates_E, rates_I, x_inf
)

# ---------- Problem configuration ----------
DT = 0.01       # ms
T_TOTAL = 30.0  # ms
TINJ_ON  = 5.0  # ms
TINJ_OFF = 15.0 # ms
G_EI = 0.05     # mS/cm^2 (E -> I AMPA)

def set_initial_conditions(net, V0=-65.0):
    """Set both neurons to V=-65 mV and set gates to steady state at that V."""
    net.V[:] = V0
    rE = rates_E(V0); rI = rates_I(V0)
    net.h[0] = x_inf(rE['ah'], rE['bh']); net.n[0] = x_inf(rE['an'], rE['bn'])
    net.h[1] = x_inf(rI['ah'], rI['bh']); net.n[1] = x_inf(rI['an'], rI['bn'])

def targeted_step_current(I0, t_on, t_off, target_index):
    def f(t, i):
        return I0 if (i == target_index and t_on <= t <= t_off) else 0.0
    return f

def count_spikes(t, V, threshold=0.0):
    V = np.asarray(V)
    above = V >= threshold
    return np.where((~above[:-1]) & (above[1:]))[0].size

def simulate_for_J0(J0, record_gates=False):
    net = EIONetwork(['E','I'])
    net.add_connection(0, 1, gbar=G_EI)  # E -> I (AMPA)
    set_initial_conditions(net, V0=-65.0)
    inj = targeted_step_current(J0, TINJ_ON, TINJ_OFF, target_index=0)  # only on E
    net.set_injection(inj)
    out = net.simulate(T=T_TOTAL, dt=DT, record_gates=record_gates)
    return net, inj, out

def find_J0():
    # smallest amplitude in [0.5, 20] µA/cm^2 that yields exactly 1 spike
    for J0 in np.arange(0.5, 20.5, 0.25):
        _, _, out = simulate_for_J0(J0, record_gates=False)
        nspk = count_spikes(out['t'], out['V'][:,0], threshold=0.0)
        if nspk == 1:
            return J0
    return None

def main():
    J0 = find_J0()
    if J0 is None:
        raise RuntimeError("No J0 in the search range produced exactly one spike. "
                           "Widen the search or tweak DT/T_TOTAL.")
    print(f"Found J0 = {J0:.3f} µA/cm^2 (single spike in E).")

    net, inj, out = simulate_for_J0(J0, record_gates=True)

    t  = out['t']
    VE = out['V'][:,0]
    VI = out['V'][:,1]

    # J_inj(t) for neuron E
    Jinj_t = np.array([inj(tt, 0) for tt in t])

    # Synaptic current into I (edge 0: E->I)
    gbar = net.syn[0].gbar; Erev = net.syn[0].Erev
    s = out['s'][:,0]
    Jsyn_t = gbar * s * (Erev - VI)

    # ---- Single stacked figure ----
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(t, Jinj_t)
    ax1.set_ylabel(r'$J_{\mathrm{inj}}$ ($\mu$A/cm$^2$)')
    ax1.set_title(f'Q1(a): E→I, $\\bar g_{{EI}}={G_EI}$ mS/cm$^2$, single spike with $J_0={J0:.3f}$')

    ax2.plot(t, VE)
    ax2.set_ylabel(r'$V_E$ (mV)')

    ax3.plot(t, Jsyn_t)
    ax3.set_ylabel(r'$J_{\mathrm{syn}}$ ($\mu$A/cm$^2$)')

    ax4.plot(t, VI)
    ax4.set_ylabel(r'$V_I$ (mV)')
    ax4.set_xlabel('t (ms)')

    fig.tight_layout()
    fig.savefig('intro-computational-neuroscience/list_8/figures/ex_1a.png', dpi=150)
    # plt.show()

if __name__ == "__main__":
    main()
