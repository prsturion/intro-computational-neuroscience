
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    EIONetwork, rates_E, rates_I, x_inf
)

# ---------- Problem configuration ----------
DT = 0.01        # ms
T_TOTAL = 60.0   # ms (a bit longer as suggested)
TINJ_ON  = 5.0   # ms
TINJ_OFF = 15.0  # ms
G_IE = 0.1       # mS/cm^2 (I -> E, GABAA_I)

# ---------- Helpers ----------

def set_initial_conditions_generic(net, V0=-65.0):
    """Initialize V to V0 and set gates (h,n) at steady state for each neuron type."""
    net.V[:] = V0
    for i, tcode in enumerate(net.neuron_types):
        if tcode.upper() == 'E':
            r = rates_E(V0)
        elif tcode.upper() == 'I':
            r = rates_I(V0)
        else:
            # O neuron not present in this problem; default to E rates if ever used
            r = rates_E(V0)
        net.h[i] = x_inf(r['ah'], r['bh'])
        net.n[i] = x_inf(r['an'], r['bn'])

def targeted_step_current(I0, t_on, t_off, target_index):
    def f(t, i):
        return I0 if (i == target_index and t_on <= t <= t_off) else 0.0
    return f

def count_spikes(t, V, threshold=0.0):
    """Count upward crossings of 'threshold' (simple spike detector)."""
    V = np.asarray(V)
    above = V >= threshold
    return np.where((~above[:-1]) & (above[1:]))[0].size

def simulate_for_J0(J0, record_gates=False):
    """Build fresh I->E network with given J0 injected to I (index 0)."""
    net = EIONetwork(['I','E'])
    net.add_connection(0, 1, gbar=G_IE)  # I -> E (GABAA_I)
    set_initial_conditions_generic(net, V0=-65.0)
    inj = targeted_step_current(J0, TINJ_ON, TINJ_OFF, target_index=0)  # inject on I only
    net.set_injection(inj)
    out = net.simulate(T=T_TOTAL, dt=DT, record_gates=record_gates)
    return net, inj, out

def find_J0():
    """Find smallest J0 in a grid that yields exactly one spike in I within T_TOTAL."""
    for J0 in np.arange(0.5, 25.5, 0.25):  # slightly wider range
        _, _, out = simulate_for_J0(J0, record_gates=False)
        nspk = count_spikes(out['t'], out['V'][:,0], threshold=0.0)  # spikes of I (index 0)
        if nspk == 1:
            return J0
    return None

def main():
    J0 = find_J0()
    if J0 is None:
        raise RuntimeError("No J0 in the search range produced exactly one spike in I. "
                           "Consider widening the J0 range or T_TOTAL.")
    print(f"Found J0 = {J0:.3f} µA/cm^2 (single spike in I).")

    net, inj, out = simulate_for_J0(J0, record_gates=True)

    t  = out['t']
    VI = out['V'][:,0]  # presynaptic I
    VE = out['V'][:,1]  # postsynaptic E

    # J_inj(t) applied to I (index 0)
    Jinj_t = np.array([inj(tt, 0) for tt in t])

    # Synaptic current into E (edge 0: I->E, GABAA_I)
    gbar = net.syn[0].gbar; Erev = net.syn[0].Erev
    s = out['s'][:,0]
    Jsyn_t = gbar * s * (Erev - VE)

    # ---- Single stacked figure ----
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(t, Jinj_t)
    ax1.set_ylabel(r'$J_{\mathrm{inj}}$ ($\mu$A/cm$^2$)')
    ax1.set_title(f'Q1(b): I→E, $\\bar g_{{IE}}={G_IE}$ mS/cm$^2$, single spike with $J_0={J0:.3f}$')

    ax2.plot(t, VI)
    ax2.set_ylabel(r'$V_I$ (mV)')

    ax3.plot(t, Jsyn_t)
    ax3.set_ylabel(r'$J_{\mathrm{syn}}$ ($\mu$A/cm$^2$)')

    ax4.plot(t, VE)
    ax4.set_ylabel(r'$V_E$ (mV)')
    ax4.set_xlabel('t (ms)')

    fig.tight_layout()
    fig.savefig('intro-computational-neuroscience/list_8/figures/ex_1b.png', dpi=150)
    # plt.show()

if __name__ == "__main__":
    main()
