# gates.py
import numpy as np
from typing import Callable, Iterable, Tuple, Optional, Any

# Maximal conductances (mS/cm²)
g_Na = 120.0   # sodium conductance
g_K  = 20.0    # potassium conductance
g_L  = 0.3     # leak conductance
g_A  = 47.7    # A-type potassium conductance

# Reversal potentials (mV)
E_Na = 55.0    # sodium reversal potential
E_K  = -72.0   # potassium reversal potential
E_L  = -17.0   # leak reversal potential
E_A  = -75.0   # A-type potassium reversal potential

# Membrane capacitance (µF/cm²)
C_m = 1.0



EPS = 1e-12  # small value to avoid division by zero at singularities

def _den(x):
    """Helper function to safely handle denominators"""
    return np.where(np.abs(x) < EPS, EPS, x)

# -------------------------
# n-gate (Potassium activation)
# -------------------------
def alpha_n(V):
    """α_n(V) = 0.01 (V + 45.7) / [1 - exp(-0.1 (V + 45.7))]"""
    x = V + 45.7
    return 0.01 * x / _den(1.0 - np.exp(-0.1 * x))

def beta_n(V):
    """β_n(V) = 0.125 * exp(-0.0125 (V + 55.7))"""
    return 0.125 * np.exp(-0.0125 * (V + 55.7))

def n_inf(V):
    """Steady-state value n∞(V)"""
    an, bn = alpha_n(V), beta_n(V)
    return an / _den(an + bn)

def tau_n(V):
    """Time constant τ_n(V)"""
    an, bn = alpha_n(V), beta_n(V)
    return 2.0 / _den(3.8 * (an + bn))

# -------------------------
# m-gate (Sodium activation)
# -------------------------
def alpha_m(V):
    """α_m(V) = 0.1 (V + 29.7) / [1 - exp(-0.1 (V + 29.7))]"""
    x = V + 29.7
    return 0.1 * x / _den(1.0 - np.exp(-0.1 * x))

def beta_m(V):
    """β_m(V) = 4 * exp(-0.0556 (V + 54.7))"""
    return 4.0 * np.exp(-0.0556 * (V + 54.7))

def m_inf(V):
    """Steady-state value m∞(V)"""
    am, bm = alpha_m(V), beta_m(V)
    return am / _den(am + bm)

def tau_m(V):
    """Time constant τ_m(V)"""
    am, bm = alpha_m(V), beta_m(V)
    return 1.0 / _den(3.8 * (am + bm))

# -------------------------
# h-gate (Sodium inactivation)
# -------------------------
def alpha_h(V):
    """α_h(V) = 0.07 * exp(-0.05 (V + 48))"""
    return 0.07 * np.exp(-0.05 * (V + 48.0))

def beta_h(V):
    """β_h(V) = 1 / [1 + exp(-0.1 (V + 18))]"""
    return 1.0 / (1.0 + np.exp(-0.1 * (V + 18.0)))

def h_inf(V):
    """Steady-state value h∞(V)"""
    ah, bh = alpha_h(V), beta_h(V)
    return ah / _den(ah + bh)

def tau_h(V):
    """Time constant τ_h(V)"""
    ah, bh = alpha_h(V), beta_h(V)
    return 1.0 / _den(3.8 * (ah + bh))

# -------------------------
# a-gate (Auxiliary gate)
# -------------------------
def a_inf(V):
    """
    Steady-state value a∞(V) =
    [ 0.0761 * exp(0.0314 (V + 94.22)) / (1 + exp(0.0346 (V + 1.17))) ]^(1/3)
    """
    num = 0.0761 * np.exp(0.0314 * (V + 94.22))
    den = 1.0 + np.exp(0.0346 * (V + 1.17))
    return np.power(num / _den(den), 1.0 / 3.0)

def tau_a(V):
    """
    τ_a(V) = 0.3632 + 1.158 / [1 + exp(0.0497 (V + 55.96))]
    """
    return 0.3632 + 1.158 / (1.0 + np.exp(0.0497 * (V + 55.96)))

# -------------------------
# b-gate (Auxiliary gate)
# -------------------------
def b_inf(V):
    """
    Steady-state value b∞(V) =
    [ 1 / (1 + exp(0.0688 (V + 53.3))) ]^4
    """
    base = 1.0 / (1.0 + np.exp(0.0688 * (V + 53.3)))
    return np.power(base, 4.0)

def tau_b(V):
    """
    τ_b(V) = 1.24 + 2.678 / [1 + exp(0.0624 (V + 50))]
    """
    return 1.24 + 2.678 / (1.0 + np.exp(0.0624 * (V + 50.0)))


# ==========================
# Reusable integration utils
# ==========================
def rk4_step(f: Callable[[float, np.ndarray], np.ndarray],
             t: float,
             y: np.ndarray,
             dt: float,
             *args, **kwargs) -> np.ndarray:
    """
    One Runge–Kutta 4th order step (generic).
    f:  function f(t, y, *args, **kwargs) -> dy/dt (shape like y)
    t:  current time
    y:  current state vector (np.ndarray)
    dt: step size
    *args/**kwargs: extra parameters forwarded to f
    Returns: y(t + dt)
    """
    k1 = f(t, y, *args, **kwargs)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, *args, **kwargs)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, *args, **kwargs)
    k4 = f(t + dt,      y + dt*k3,     *args, **kwargs)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_solve(f: Callable[[float, np.ndarray], np.ndarray],
              t0: float,
              t1: float,
              dt: float,
              y0: Iterable[float],
              *args, callback: Optional[Callable[[float, np.ndarray], Any]] = None,
              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate y' = f(t, y) from t0 to t1 with fixed step RK4.
    Returns (t_array, Y) where Y[i] is the state at t[i].
    If callback is provided, it is called every step as callback(t, y).
    """
    t = np.arange(t0, t1 + 1e-12, dt)
    y = np.array(y0, dtype=float)
    Y = np.empty((t.size, y.size), dtype=float)
    for i, ti in enumerate(t):
        Y[i] = y
        if callback is not None:
            callback(ti, y)
        if i < t.size - 1:
            y = rk4_step(f, ti, y, dt, *args, **kwargs)
    return t, Y

# ==========================
# Current protocol helpers
# ==========================
def step_current_protocol(t_on: float, t_off: float, amp: float) -> Callable[[float], float]:
    """Return J(t) that is amp for t in [t_on, t_off) and 0 otherwise."""
    def J(t: float) -> float:
        return amp if (t >= t_on and t < t_off) else 0.0
    return J

def piecewise_constant_protocol(knots: Iterable[float], amps: Iterable[float]) -> Callable[[float], float]:
    """
    Piecewise-constant J(t). 'knots' are increasing times [t0, t1, ..., tN].
    'amps' has length N: value on [t_i, t_{i+1}).
    """
    knots = np.asarray(list(knots), dtype=float)
    amps  = np.asarray(list(amps), dtype=float)
    assert knots.ndim == 1 and amps.ndim == 1 and amps.size == knots.size - 1
    def J(t: float) -> float:
        idx = np.searchsorted(knots, t, side="right") - 1
        idx = np.clip(idx, 0, amps.size - 1)
        return float(amps[idx])
    return J

# ==========================
# Connor–Stevens model utils
# ==========================
# Reuse your constants (g_Na, g_K, g_A, g_L, E_Na, E_K, E_A, E_L, C_m)
# and the gating functions already defined above in this same utils.py.

def I_Na(V: float, m: float, h: float) -> float:
    """Sodium current density (µA/cm²)."""
    return g_Na * (m**3) * h * (V - E_Na)

def I_K(V: float, n: float) -> float:
    """Delayed rectifier potassium current density (µA/cm²)."""
    return g_K * (n**4) * (V - E_K)

def I_A(V: float, a: float, b: float) -> float:
    """A-type potassium current density (µA/cm²)."""
    return g_A * (a**3) * b * (V - E_A)

def I_L(V: float) -> float:
    """Leak current density (µA/cm²)."""
    return g_L * (V - E_L)

def connor_stevens_rhs(t: float, y: np.ndarray, J_of_t: Callable[[float], float]) -> np.ndarray:
    """
    Right-hand side of the Connor–Stevens ODE system.
    y = [V, n, m, h, a, b]
    Returns dy/dt with units: V in mV, time in ms, currents in µA/cm².
    """
    V, n, m, h, a, b = y

    # Membrane equation: C_m dV/dt = - (I_ion - J)
    Iion = I_Na(V, m, h) + I_K(V, n) + I_A(V, a, b) + I_L(V)
    dVdt = -(Iion - J_of_t(t)) / C_m

    # First-order gating kinetics: dx/dt = (x_inf(V) - x) / tau_x(V)
    dndt = (n_inf(V) - n) / tau_n(V)
    dmdt = (m_inf(V) - m) / tau_m(V)
    dhdt = (h_inf(V) - h) / tau_h(V)
    dadt = (a_inf(V) - a) / tau_a(V)
    dbdt = (b_inf(V) - b) / tau_b(V)

    return np.array([dVdt, dndt, dmdt, dhdt, dadt, dbdt], dtype=float)