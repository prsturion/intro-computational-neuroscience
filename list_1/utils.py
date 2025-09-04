import numpy as np

# Método de Euler
def euler(f, t0, t1, y0, h):
    """Método de Euler explícito para resolver equações diferenciais
    ordinárias.

    Parameters
    ----------
    f : callable
        Função que define a equação diferencial, que deve ter a seguinte
        assinatura: f(t, y), onde t é um escalar e y é um vetor (ou array)
        de valores.
    t0 : float
        Tempo inicial.
    t1 : float
        Tempo final.
    y0 : array_like
        Vetor de condições iniciais.
    h : float
        Passo de integração.

    Returns
    -------
    t, y : tuple
        t é um array com os valores de tempo e y é um array com as soluções
        encontradas em cada tempo.
    """
    N = int((t1 - t0)/h)
    t = np.linspace(t0, t1, N+1)
    y = np.zeros((N+1,))
    y[0] = y0
    for k in range(N):
        y[k+1] = y[k] + h * f(t[k], y[k])
    return t, y


# EDO Equação da membrana
def eq_membrana(t, V_m, g_m, C_m, E, J_inj, SI = True):
    """Equação da membrana.

    Parameters
    ----------
    t : float
        Tempo.
    V_m : float
        Tensão da membrana.
    g_m : float
        Condutância específica da membrana em [uS/cm^2].
    C_m : float
        Capacitância espec fica da membrana em [uF/cm^2].
    E : float
        Potencial de equilíbrio em [mV].
    J_inj : float
        Densidade de corrente injetada em [uA/cm^2].

    Returns
    -------
    dydt : float
        Taxa de varia o temporal da tensão da membrana em [mV/ms].
    """
    fator = 1 if SI else 1e-3
    dydt = (1/C_m) * (g_m * (E - V_m) * fator + J_inj)
    return dydt

def analytic_solution(t, V_0, g_m, C_m, E, J_inj, SI = True):

    """Solu o analítica da equa o da membrana.

    Parameters
    ----------
    t : array_like
        Vetor de tempos.
    V_0 : float
        Valor inicial da tens o da membrana em [mV].
    g_m : float
        Condutância espec fica da membrana em [uS/cm^2].
    C_m : float
        Capacitância espec fica da membrana em [uF/cm^2].
    E : float
        Potencial de equil brio em [mV].
    J_inj : float
        Densidade de corrente injetada em [uA/cm^2].
    SI : bool, optional
        Se True, as unidades são dadas no sistema internacional (SI) e
        os valores de entrada devem estar nesse sistema. Caso contrário, as
        unidades são dadas em uS/cm^2, uF/cm^2, mV, uA/cm^2 e ms.

    Returns
    -------
    V_m : array_like
        Tensão da membrana em cada tempo t, em [mV].
    """
    fator = 1 if SI else 1e-3
    V_m = (V_0 - E - J_inj/(g_m*fator)) * np.exp(-(g_m*fator)/C_m * t) + (E + J_inj/(g_m*fator))
    return V_m