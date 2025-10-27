# lif_fi.py
# Curva f–I do neurônio LIF com período refratário absoluto
# Euler explícito + detecção de limiar com interpolação linear

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parâmetros do modelo (SI)
# ----------------------------
V_rep   = -70e-3   # V (potencial de repouso)
R       = 40e6     # Ohm
C       = 250e-12  # F
V_th    = -50e-3   # V (limiar V_L)
V_reset = -65e-3   # V (V_ref)
t_ref   = 2e-3     # s (refratário absoluto)

dt      = 0.01e-3  # s  (0.01 ms)
t_max   = 2.0      # s  (tempo total de simulação)
tau     = R * C    # constante de tempo da membrana

# ----------------------------
# Curva analítica f(I)
# ----------------------------
def fI_analitico(I):
    """
    f(I) = 1 / [ t_ref + tau * ln( (V_inf - V_reset) / (V_inf - V_th) ) ], se V_inf>V_th
    senão f=0. Aceita escalar ou array.
    """
    V_inf = V_rep + R * I
    if np.isscalar(I):
        if V_inf <= V_th:
            return 0.0
        T = t_ref + tau * np.log((V_inf - V_reset) / (V_inf - V_th))
        return 1.0 / T
    f = np.zeros_like(I, dtype=float)
    mask = V_inf > V_th
    Vin = V_inf[mask]
    T = t_ref + tau * np.log((Vin - V_reset) / (Vin - V_th))
    f[mask] = 1.0 / T
    return f

# ----------------------------
# Simulador LIF (numérico)
# ----------------------------
def simular_spikes(I, dt=dt, t_max=t_max):
    """
    Simula V(t) com Euler. Retorna tempos de spikes (lista).
    - Limiar: cruzamento de V_th vindo de baixo.
    - Instante do disparo t*: interpolação linear entre k e k+1.
    - Refratário absoluto: mantém V_reset por t_ref após o spike.
    """
    n_steps = int(t_max / dt)
    V = V_rep
    t = 0.0
    ref_restante = 0.0
    spikes = []

    for _ in range(n_steps):
        if ref_restante > 0.0:
            V_next = V_reset
            ref_restante -= dt
        else:
            dVdt = ((V_rep - V) / R + I) / C
            V_next = V + dt * dVdt

            # detecta cruzamento: V_k < V_th <= V_{k+1}
            if (V < V_th) and (V_next >= V_th):
                frac = (V_th - V) / (V_next - V)  # fração do passo até o limiar
                t_star = t + dt * frac
                spikes.append(t_star)
                V_next = V_reset
                ref_restante = t_ref

        V = V_next
        t += dt

    return np.array(spikes)

def freq_numerica(I):
    """Frequência média em Hz ao longo de t_max."""
    spikes = simular_spikes(I)
    return spikes.size / t_max

# ----------------------------
# Faixa de correntes: de < reobase até ~150 Hz
# ----------------------------
I_L = (V_th - V_rep) / R  # reobase = 0.5 nA
# Encontrar I que dá ~150 Hz via bisseção na fórmula analítica
def I_para_f_alvo(f_alvo, I_lo=None, I_hi=None, iters=50):
    if I_lo is None: I_lo = I_L * 1.001
    if I_hi is None:
        I_hi = I_L + 5e-9
        while fI_analitico(I_hi) < f_alvo:
            I_hi *= 1.5
            if I_hi > 1e-6: break  # segurança
    lo, hi = I_lo, I_hi
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if fI_analitico(mid) < f_alvo:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

I_min = 0.40e-9                # abaixo do reobase (0.5 nA)
I_max = I_para_f_alvo(150.0)   # ~1.13 nA para 150 Hz
N_pts = 200

# ----------------------------
# Varredura e gráficos
# ----------------------------
def main():
    I_vals = np.linspace(I_min, I_max, N_pts)
    f_num  = np.array([freq_numerica(I) for I in I_vals[::4]])
    f_ana  = fI_analitico(I_vals)

    plt.figure()
    plt.plot(I_vals*1e9, f_ana, label="Analítico", color='r')
    plt.plot(I_vals[::4]*1e9, f_num, "x", label="Numérico (simulação)", markersize=4, color='k')
    plt.xlabel("Corrente I (nA)")
    plt.ylabel("Frequência f (Hz)")
    plt.title("Curva f–I do neurônio LIF (refratário absoluto)")
    plt.grid(True)
    plt.legend()
   #  plt.show()
    plt.savefig("intro-computational-neuroscience/list_6/figures/ex_1a.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
