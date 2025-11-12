# ex_1d.py — item (d) no estilo da Figura 3 (corrigido)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Evita avisos de glifo/fonte
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

from utils import EIONetwork, rates_E, rates_I, x_inf

# ---------- Configuração (como no item c) ----------
DT = 0.01        # ms
T_TOTAL = 500.0  # ms
TINJ_ON  = 5.0   # ms
TINJ_OFF = 450.0 # ms
J0 = 1.5         # µA/cm^2 (degrau constante em E)
G_LIST = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]  # mS/cm^2

# Janela só para deixar o gráfico bonito (opcional)
T_MIN = 0.0
T_MAX = 480.0

def targeted_step_current(I0, t_on, t_off, target_index):
    def f(t, i):
        return I0 if (i == target_index and t_on <= t <= t_off) else 0.0
    return f

def set_initial_conditions(net, V0=-65.0):
    net.V[:] = V0
    rE = rates_E(V0); rI = rates_I(V0)
    net.h[0] = x_inf(rE['ah'], rE['bh']); net.n[0] = x_inf(rE['an'], rE['bn'])
    net.h[1] = x_inf(rI['ah'], rI['bh']); net.n[1] = x_inf(rI['an'], rI['bn'])

def spike_times(t, V, threshold=0.0):
    """Tempos de cruzamento ascendente do limiar com interpolação linear."""
    V = np.asarray(V)
    above = V >= threshold
    idx = np.where((~above[:-1]) & (above[1:]))[0]
    times = []
    for k in idx:
        t0, t1 = t[k], t[k+1]
        v0, v1 = V[k], V[k+1]
        frac = 0.0 if v1 == v0 else (threshold - v0)/(v1 - v0)
        times.append(t0 + frac*(t1 - t0))
    return np.array(times)

def simulate_once(g_ei):
    net = EIONetwork(['E','I'])
    net.add_connection(0, 1, gbar=g_ei)  # E -> I (AMPA)
    set_initial_conditions(net, V0=-65.0)
    net.set_injection(targeted_step_current(J0, TINJ_ON, TINJ_OFF, target_index=0))
    out = net.simulate(T=T_TOTAL, dt=DT, record_gates=True)
    return net, out

def fig3_for_g(g):
    _, out = simulate_once(g)
    t = out['t']; VE = out['V'][:,0]; VI = out['V'][:,1]
    spikes_E = spike_times(t, VE, threshold=0.0)
    spikes_I = spike_times(t, VI, threshold=0.0)

    # pareamento 1:1 por índice (coerente com a figura proposta)
    m = min(len(spikes_E), len(spikes_I))
    deltas = spikes_I[:m] - spikes_E[:m]

    # ---- Figura 3 (topo: V, baixo: δ por disparo) ----
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7), sharex=False)

    mask = (t >= T_MIN) & (t <= T_MAX)
    ax_top.plot(t[mask], VE[mask], linewidth=0.8, alpha=0.6, label='V_E (ref)')
    ax_top.plot(t[mask], VI[mask], 'k', linewidth=1.2, label='V_I')  # traço preto
    for ts in spikes_E[(spikes_E >= T_MIN) & (spikes_E <= T_MAX)]:
        ax_top.axvline(ts, color='r', linestyle='--', alpha=0.5)     # linhas vermelhas nos spikes de E
    ax_top.set_xlim(T_MIN, T_MAX)
    ax_top.set_ylabel('V (mV)')
    # >>> título corrigido (use f-string e escape das chaves do LaTeX) <<<
    ax_top.set_title(rf'$\bar g_{{EI}} = {g:.2f}$ mS/cm$^2$')
    ax_top.legend(loc='upper right', frameon=False)

    n_show = min(15, len(deltas))
    ax_bot.plot(np.arange(1, n_show+1), deltas[:n_show], 'k.', markersize=6)
    ax_bot.set_xlim(1, max(2, n_show))
    ax_bot.set_xlabel('disparo #')
    ax_bot.set_ylabel(r'$\delta$ (ms)')

    fig.tight_layout()
    fname = f'intro-computational-neuroscience/list_8/figures/ex_1d_g_{int(round(100*g)):03d}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    # resumo no terminal
    if len(spikes_E) >= 2 and len(spikes_I) >= 2:
        fE = len(spikes_E) / ((t[-1]-t[0])/1000.0)
        fI = len(spikes_I) / ((t[-1]-t[0])/1000.0)
    else:
        fE = fI = 0.0
    n_est = np.round(fE/fI) if fI > 0 else np.nan
    print(f"g_EI={g:.2f}: spikes E={len(spikes_E)}, I={len(spikes_I)}, "
          f"fE={fE:.2f} Hz, fI={fI:.2f} Hz, n≈{n_est}, "
          f"mean δ={np.mean(deltas[:n_show]) if n_show>0 else np.nan:.2f} ms")

def main():
    for g in G_LIST:
        fig3_for_g(g)

if __name__ == '__main__':
    main()
