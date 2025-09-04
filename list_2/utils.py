from typing import Callable, List, Tuple
import numpy as np

# Método de Euler
def euler(
    f_list: List[Callable[..., float]],
    y0_list: List[float],
    t0: float,
    tf: float,
    h: float
) -> Tuple[List[float], List[List[float]]]:
    """
    Método de Euler para sistemas dy/dt = f(t, y), onde
    y = [y0, y1, ..., y_{n-1}] e f_list = [f0, f1, ..., f_{n-1}],
    cada f_i(t, *y) -> dy_i/dt.

    Parâmetros:
        f_list : list de funções derivadas (f_i(t, *y))
        y0_list: list de condições iniciais (mesmo tamanho de f_list)
        t0     : tempo inicial
        tf     : tempo final
        h      : passo base da integração

    Retorna:
        ts : list dos tempos
        ys : list de lists; cada linha é a série temporal de uma variável
             (ys[i][j] = valor da i-ésima variável no j-ésimo instante de tempo)
    """

    if len(f_list) != len(y0_list):
        raise ValueError("f_list e y0_list devem ter o mesmo tamanho.")

    # Inicialização
    ts: List[float] = [t0]
    ys: List[List[float]] = [list(y0_list)]  # cada linha = estado no tempo t

    t = t0
    y = list(y0_list)

    # Integra até tf, ajustando o último passo se necessário
    while t < tf:
        next_t = t + h
        if next_t > tf:
            next_t = tf
        dt = next_t - t

        # Calcula derivadas no ponto atual
        dydt = [fi(t, *y) for fi in f_list]

        # Passo de Euler
        y = [yi + dt * dyi for yi, dyi in zip(y, dydt)]
        t = next_t

        ts.append(t)
        ys.append(list(y))

    # Transpor para que cada linha seja uma variável ao longo do tempo
    ys_transposed: List[List[float]] = list(map(list, zip(*ys)))

    return ts, ys_transposed


# Achar cruzamentos de curvas
def find_crossings(values, L, direction='both'):
    """
    Encontra os índices onde a curva dada pelos valores cruza o valor L,
    com a opção de selecionar o tipo de transição (negativo-para-positivo, positivo-para-negativo, ou ambos).
    
    Parâmetros:
        values    : list de valores (por exemplo, resultado de uma função de t)
        L         : valor de referência (o nível que estamos verificando os cruzamentos)
        direction: tipo de transição a ser considerado:
                    'both'  : considera transições de ambos os sentidos
                    'up'    : considera transições de negativo para positivo
                    'down'  : considera transições de positivo para negativo

    Retorna:
        indices  : list de índices onde a curva cruza o valor L
    """
    # Subtrai L de cada valor da list
    diff = [v - L for v in values]
    
    # Inicializa list para armazenar os índices de cruzamento
    indices = []
    
    for i in range(1, len(diff)):
        # Verifica a mudança de sinal
        if diff[i-1] * diff[i] < 0:
            # Transição de negativo para positivo
            if direction == 'up' and diff[i-1] < 0 and diff[i] > 0:
                indices.append(i)
            # Transição de positivo para negativo
            elif direction == 'down' and diff[i-1] > 0 and diff[i] < 0:
                indices.append(i)
            # Ambas as transições
            elif direction == 'both':
                indices.append(i)
    
    return indices


def HH_alpha(V, gating):
    
    if gating == 'n':
        if V != -55: # Avoiding 0/0 problems
            alpha = 0.01*((V + 55)/(1 - np.exp(-(V + 55)/10))) 
        else:
            alpha = 0.1 # lim V -> -55

    elif gating == 'm':
        if V != -40: # Avoiding 0/0 problems
            alpha = 0.1*((V + 40)/(1 - np.exp(-(V + 40)/10))) 
        else:
            alpha = 1 # lim V -> -40

    elif gating == 'h':
        alpha = 0.07*np.exp(-(V + 65)/20)

    else:
        raise ValueError("Gating must be 'n', 'm' or 'h'")

    return alpha


def HH_beta(V, gating):
    
    if gating == 'n':
        beta = 0.125*np.exp(-(V + 65)/80)

    elif gating == 'm':
        beta = 4*np.exp(-(V + 65)/18)

    elif gating == 'h':
        beta = 1/(1 + np.exp(-(V + 35)/10))

    else:
        raise ValueError("Gating must be 'n', 'm' or 'h'")

    return beta


def HH_gating_eq(t, V, n, m, h, gating):

    if gating == 'n':
        dydt = HH_alpha(V, 'n')*(1 - n) - HH_beta(V, 'n')*n

    elif gating == 'm':
        dydt = HH_alpha(V, 'm')*(1 - m) - HH_beta(V, 'm')*m

    elif gating == 'h':
        dydt = HH_alpha(V, 'h')*(1 - h) - HH_beta(V, 'h')*h

    else:
        raise ValueError("Gating must be 'n', 'm' or 'h'")

    return dydt



def HH_equation(t, V, n, m, h, C_m, g_Na, g_K, g_V, E_Na, E_K, E_V, J_inj):

    dVdt = (1/C_m) * (
        - g_Na * m**3 * h * (V - E_Na)
        - g_K * n**4 * (V - E_K)
        - g_V * (V - E_V)
        + J_inj
    )

    return dVdt