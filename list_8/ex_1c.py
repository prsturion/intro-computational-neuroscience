
import numpy as np
import matplotlib.pyplot as plt

from utils import EIONetwork, rates_E, rates_I, x_inf

# ---------- Problem configuration ----------
DT = 0.01        # ms
T_TOTAL = 500.0  # ms
TINJ_ON  = 5.0   # ms
TINJ_OFF = 450.0 # ms
J0 = 1.5         # µA/cm^2 (constant step on E)
G_LIST = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]  # mS/cm^2

# ---------- Helpers ----------

def targeted_step_current(I0, t_on, t_off, target_index):
    def f(t, i):
        return I0 if (i == target_index and t_on <= t <= t_off) else 0.0
    return f

def set_initial_conditions(net, V0=-65.0):
    net.V[:] = V0
    # E at index 0, I at index 1
    rE = rates_E(V0); rI = rates_I(V0)
    net.h[0] = x_inf(rE['ah'], rE['bh']); net.n[0] = x_inf(rE['an'], rE['bn'])
    net.h[1] = x_inf(rI['ah'], rI['bh']); net.n[1] = x_inf(rI['an'], rI['bn'])

def spike_times(t, V, threshold=0.0):
    V = np.asarray(V)
    above = V >= threshold
    idx = np.where((~above[:-1]) & (above[1:]))[0]
    # Linear interpolation for better timing
    times = []
    for k in idx:
        t0, t1 = t[k], t[k+1]
        v0, v1 = V[k], V[k+1]
        if v1 != v0:
            frac = (threshold - v0)/(v1 - v0)
            times.append(t0 + frac*(t1 - t0))
        else:
            times.append(t0)
    return np.array(times)

def mean_ISI(times, tmin=None, tmax=None):
    if tmin is not None:
        times = times[times >= tmin]
    if tmax is not None:
        times = times[times <= tmax]
    if len(times) < 2:
        return np.nan
    return np.mean(np.diff(times))

def simulate_once(g_ei):
    net = EIONetwork(['E','I'])
    net.add_connection(0, 1, gbar=g_ei)  # E -> I (AMPA)
    set_initial_conditions(net, V0=-65.0)
    net.set_injection(targeted_step_current(J0, TINJ_ON, TINJ_OFF, target_index=0))
    out = net.simulate(T=T_TOTAL, dt=DT, record_gates=True)
    return net, out

def main():
    # First, get the firing period T of E (any g_ei since I doesn't feed back to E)
    _, out0 = simulate_once(G_LIST[0])
    t = out0['t']
    VE = out0['V'][:,0]
    spike_t_E = spike_times(t, VE, threshold=0.0)
    T_est = mean_ISI(spike_t_E, tmin=100.0, tmax=440.0)
    print(f"Estimated interspike interval of E with J0={J0} µA/cm²: T ≈ {T_est:.3f} ms "
          f"(mean ISI from {len(spike_t_E)} spikes).")

    # Now sweep g_EI values and check whether I fires a spike train
    results = []
    for g in G_LIST:
        _, out = simulate_once(g)
        VI = out['V'][:,1]
        spike_t_I = spike_times(out['t'], VI, threshold=0.0)
        # define "train" as >= 3 spikes in [100, 440] ms (avoid transients & off period)
        n_spikes_mid = np.sum((spike_t_I >= 100.0) & (spike_t_I <= 440.0))
        is_train = n_spikes_mid >= 3
        results.append((g, n_spikes_mid, is_train))

    # Print summary
    print("\nResults for g_EI (mS/cm²): spikes in I and whether it forms a train (>=3 spikes).")
    for g, nspk, ok in results:
        print(f"  g_EI={g:>4.2f} -> I spikes: {nspk:>3d}  -> train? {'YES' if ok else 'NO'}")

    # Which values produced a train?
    winners = [g for g, nspk, ok in results if ok]
    print("\nValues of g_EI for which I emits a spike train:", winners)
'''
    # Optional: quick plot of a representative case (largest g that produced a train, if any)
    if winners:
        g = winners[-1]
        _, out = simulate_once(g)
        t = out['t']; VE = out['V'][:,0]; VI = out['V'][:,1]
        # separate figures as per simple style
        plt.figure(); plt.plot(t, VE); plt.xlabel('t (ms)'); plt.ylabel(r'$V_E$ (mV)'); plt.title(f'E voltage (g_EI={g:.2f})')
        plt.figure(); plt.plot(t, VI); plt.xlabel('t (ms)'); plt.ylabel(r'$V_I$ (mV)'); plt.title(f'I voltage (g_EI={g:.2f})')
        plt.savefig('intro-computational-neuroscience/list_8/ex_1c.png', dpi=150)  # save last one
        plt.show()
'''
if __name__ == "__main__":
    main()
