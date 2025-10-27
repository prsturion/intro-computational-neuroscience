# lif_fi_ruido.py
# LIF com ruído aditivo gaussiano (Euler–Maruyama) + refratário absoluto.
# Curvas f–I para dois valores de sigma especificados em SIGMAS.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parâmetros do modelo (SI)
# ----------------------------
V_rep   = -70e-3   # V (repouso)
R       = 40e6     # Ohm
C       = 250e-12  # F
V_th    = -50e-3   # V (limiar V_L)
V_reset = -65e-3   # V (reset V_ref)
t_ref   = 2e-3     # s (refratário absoluto)

dt      = 0.01e-3  # s  (0.01 ms)
t_max   = 2.0      # s  (tempo total)
tau     = R * C

# ----------------------------
# Ruído branco w(t)
#   dV = f(V) dt + sigma * dW
# Na discretização de Euler–Maruyama:
#   V_{n+1} = V_n + f(V_n)*dt + sigma * sqrt(dt) * N(0,1)
# sigma tem unidade V / sqrt(s). O termo sigma*sqrt(dt) tem unidade de V.
# Escolha valores de sigma para ver diferença visível na f–I.
#   Dica: com dt=1e-5 s, sqrt(dt) ≈ 0.00316; então
#   sigma=0.2 -> passo de ruído ~0.63 mV; sigma=0.6 -> ~1.9 mV.
# ----------------------------
SIGMAS    = [.06, 1.6]  # V / sqrt(s)   <-- ajuste aqui os dois valores pedidos
N_TRIALS  = 5           # média sobre ensaios para estabilizar f (por causa do ruído)
SEED_BASE = 12345       # base para sementes; correntes diferentes usam sementes diferentes

# Faixa de correntes: de abaixo do reobase até ~150 Hz
I_L   = (V_th - V_rep) / R          # reobase ≈ 0.5 nA
I_min = 0.40e-9                     # abaixo do reobase
# Encontrar corrente ~150 Hz (analítico, sem ruído) para delimitar topo da faixa
def fI_analitico(I):
    V_inf = V_rep + R * I
    if np.isscalar(I):
        if V_inf <= V_th:
            return 0.0
        T = t_ref + tau * np.log((V_inf - V_reset) / (V_inf - V_th))
        return 1.0 / T
    f = np.zeros_like(I, dtype=float)
    mask = (V_rep + R*I) > V_th
    T = t_ref + tau * np.log(((V_rep + R*I[mask]) - V_reset) / ((V_rep + R*I[mask]) - V_th))
    f[mask] = 1.0 / T
    return f

def I_para_f_alvo(f_alvo, iters=50):
    lo = I_L * 1.001
    hi = I_L + 5e-9
    while fI_analitico(hi) < f_alvo:
        hi *= 1.5
        if hi > 1e-6: break
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if fI_analitico(mid) < f_alvo:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

I_max = I_para_f_alvo(150.0)       # ~1.13 nA
N_pts = 25
I_vals = np.linspace(I_min, I_max, N_pts)

# ----------------------------
# Simulador LIF com ruído
# ----------------------------
def simular_spikes(I, sigma, rng):
    """
    Simula V(t) com Euler–Maruyama.
    - Limiar: V_k < V_th <= V_{k+1} (vindo de baixo) com interpolação linear para t*.
    - Refratário absoluto: mantém V = V_reset por t_ref após spike.
    """
    n_steps = int(t_max / dt)
    t = 0.0
    V = V_rep        # condição inicial: V(0) = V_rep
    ref_restante = 0.0
    spikes = []

    for _ in range(n_steps):
        if ref_restante > 0.0:
            V_next = V_reset
            ref_restante -= dt
        else:
            drift = ((V_rep - V) / R + I) / C
            noise = sigma * np.sqrt(dt) * rng.normal()
            V_next = V + dt * drift + noise

            if (V < V_th) and (V_next >= V_th):
                # interpolação linear para estimar t* (mesmo com ruído, dentro do passo)
                frac = (V_th - V) / (V_next - V) if V_next != V else 1.0
                t_star = t + dt * np.clip(frac, 0.0, 1.0)
                spikes.append(t_star)
                V_next = V_reset
                ref_restante = t_ref

        V = V_next
        t += dt

    return np.array(spikes)

def freq_media(I, sigma, i_I, i_sig):
    """Frequência média (Hz) sobre N_TRIALS, com sementes diferentes por corrente/ensaio."""
    freqs = []
    for rep in range(N_TRIALS):
        rng = np.random.default_rng(SEED_BASE + 10_000*i_sig + 100*i_I + rep)
        spikes = simular_spikes(I, sigma, rng)
        freqs.append(spikes.size / t_max)
    return float(np.mean(freqs))

# ----------------------------
# Executa varredas e plota
# ----------------------------
def main():
    # Curva analítica sem ruído (para referência)
    f_an = fI_analitico(I_vals)

    plt.figure()
    plt.plot(I_vals*1e9, f_an, label="Analítico (σ=0)")

    # Curvas com ruído para os dois sigmas
    for i_sig, sigma in enumerate(SIGMAS):
        f_num = np.array([freq_media(I, sigma, i_I=i, i_sig=i_sig) for i, I in enumerate(I_vals)])
        plt.plot(I_vals*1e9, f_num, "o-", label=f"Numérico com ruído (σ={sigma:g})")

    plt.xlabel("Corrente I (nA)")
    plt.ylabel("Frequência f (Hz)")
    plt.title("Curvas f–I do LIF (Euler–Maruyama, refratário absoluto)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_6/figures/ex_2a.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    print(f"Reobase (analítico): I_L = {(I_L*1e9):.3f} nA")
    main()
