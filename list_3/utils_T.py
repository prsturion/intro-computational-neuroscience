from __future__ import annotations
from typing import Callable, Tuple, Optional, Any
import numpy as np

# -----------------------------
# Unit helpers (SI)
# -----------------------------
mV = 1e-3
ms = 1e-3
uS = 1e-6
nS = 1e-9
pF = 1e-12
pA = 1e-12

# -----------------------------
# Model parameters (global, SI)
# From the statement (Miller's thalamic relay neuron).
# -----------------------------
G_L  = 10.0 * nS
G_Na = 3.6  * uS
G_K  = 1.6  * uS
G_T  = 0.22 * uS

E_L  = -70.0 * mV
E_Na =  55.0 * mV
E_K  = -90.0 * mV
E_Ca = 120.0 * mV

C_m  = 100.0 * pF

# -----------------------------
# Numerical safety
# -----------------------------
_EPS = 1e-12
def _den(x: np.ndarray | float) -> np.ndarray | float:
    """Avoid division by zero in rate expressions."""
    if isinstance(x, np.ndarray):
        out = x.copy()
        out[np.abs(out) < _EPS] = np.sign(out[np.abs(out) < _EPS]) * _EPS + (out[np.abs(out) < _EPS] == 0) * _EPS
        return out
    return x if abs(x) >= _EPS else (_EPS if x >= 0 else -_EPS)

# -----------------------------
# Gating rates (V in volts, t in seconds)
# Equations as in the statement (Question 2).
# -----------------------------
def alpha_m(V: np.ndarray | float) -> np.ndarray | float:
    # α_m(V) = 1e5 (V + 0.035) / [1 - exp(-100 (V + 0.035))]
    x = V + 0.035
    return 1.0e5 * x / _den(1.0 - np.exp(-100.0 * x))

def beta_m(V: np.ndarray | float) -> np.ndarray | float:
    # β_m(V) = 4000 * exp(-(V + 0.06)/0.018)
    return 4000.0 * np.exp(-(V + 0.06) / 0.018)

def m_inf(V: np.ndarray | float) -> np.ndarray | float:
    am, bm = alpha_m(V), beta_m(V)
    return am / _den(am + bm)

def alpha_h(V: np.ndarray | float) -> np.ndarray | float:
    # α_h(V) = 350 * exp(-50 (V + 0.058))
    return 350.0 * np.exp(-50.0 * (V + 0.058))

def beta_h(V: np.ndarray | float) -> np.ndarray | float:
    # β_h(V) = 5000 / [1 + exp(-100 (V + 0.028))]
    return 5000.0 / (1.0 + np.exp(-100.0 * (V + 0.028)))

def h_inf(V: np.ndarray | float) -> np.ndarray | float:
    ah, bh = alpha_h(V), beta_h(V)
    return ah / _den(ah + bh)

def tau_h(V: np.ndarray | float) -> np.ndarray | float:
    # τ_h = 1 / (α_h + β_h)
    ah, bh = alpha_h(V), beta_h(V)
    return 1.0 / _den(ah + bh)

def alpha_n(V: np.ndarray | float) -> np.ndarray | float:
    # α_n(V) = 5e4 (V + 0.034) / [1 - exp(-100 (V + 0.034))]
    x = V + 0.034
    return 5.0e4 * x / _den(1.0 - np.exp(-100.0 * x))

def beta_n(V: np.ndarray | float) -> np.ndarray | float:
    # β_n(V) = 625 * exp(-12.5 (V + 0.044))
    return 625.0 * np.exp(-12.5 * (V + 0.044))

def n_inf(V: np.ndarray | float) -> np.ndarray | float:
    an, bn = alpha_n(V), beta_n(V)
    return an / _den(an + bn)

def tau_n(V: np.ndarray | float) -> np.ndarray | float:
    # τ_n = 1 / (α_n + β_n)
    an, bn = alpha_n(V), beta_n(V)
    return 1.0 / _den(an + bn)

def mT_inf(V: np.ndarray | float) -> np.ndarray | float:
    # m_T,∞(V) = 1 / (1 + exp(-(V + 0.052)/0.0074))
    return 1.0 / (1.0 + np.exp(-(V + 0.052) / 0.0074))

def hT_inf(V: np.ndarray | float) -> np.ndarray | float:
    # h_T,∞(V) = 1 / (1 + exp(500 (V + 0.076)))
    return 1.0 / (1.0 + np.exp(500.0 * (V + 0.076)))

def tau_hT(V: np.ndarray | float) -> np.ndarray | float:
    # τ_hT(V) piecewise:
    # if V < -0.080: τ = 0.001 * exp(15 (V + 0.467))
    # else:          τ = 0.028 + 0.001 * exp(-(V + 0.022)/0.0105)
    V = np.asarray(V)
    out = np.empty_like(V, dtype=float)
    mask = V < -0.080
    out[mask]  = 0.001 * np.exp(15.0 * (V[mask] + 0.467))
    out[~mask] = 0.028 + 0.001 * np.exp(-(V[~mask] + 0.022) / 0.0105)
    return out

# -----------------------------
# Ionic currents (A) for a single cell (global conductances)
# -----------------------------
def I_L(V: np.ndarray | float) -> np.ndarray | float:
    return G_L * (V - E_L)

def I_Na(V: np.ndarray | float, h: np.ndarray | float) -> np.ndarray | float:
    return G_Na * (m_inf(V) ** 3) * h * (V - E_Na)

def I_K(V: np.ndarray | float, n: np.ndarray | float) -> np.ndarray | float:
    return G_K * (n ** 4) * (V - E_K)

def I_T(V: np.ndarray | float, hT: np.ndarray | float) -> np.ndarray | float:
    return G_T * (mT_inf(V) ** 2) * hT * (V - E_Ca)

# -----------------------------
# RHS of the ODE system (Eq. 7–10)
# State y = [V, h, n, hT]
# -----------------------------
def thalamic_rhs(t: float, y: np.ndarray, Iinj_of_t: Callable[[float], float]) -> np.ndarray:
    """
    Right-hand side for the thalamic relay neuron with T-type Ca current.
    t: time (s)
    y: [V, h, n, hT] (V is in volts; gating variables are dimensionless)
    Iinj_of_t: function mapping time->current (A)
    """
    V, h, n, hT = y

    # Membrane equation: C dV/dt = - (I_L + I_Na + I_K + I_T) + Iinj
    Iion = I_L(V) + I_Na(V, h) + I_K(V, n) + I_T(V, hT)
    dVdt = -(Iion) / C_m + Iinj_of_t(t) / C_m #type: ignore

    # Gating dynamics
    dhdt  = alpha_h(V)  * (1.0 - h)  - beta_h(V)  * h
    dndt  = alpha_n(V)  * (1.0 - n)  - beta_n(V)  * n
    dhTdt = (hT_inf(V) - hT) / tau_hT(V)

    return np.array([dVdt, dhdt, dndt, dhTdt], dtype=float)

# -----------------------------
# Steady-state initial conditions at V = E_L
# -----------------------------
def initial_state_at_rest(V0: float = E_L) -> np.ndarray:
    """
    Return y0 = [V0, h0, n0, hT0] with h0, n0, hT0 at steady state for V0.
    """
    h0  = h_inf(V0)
    n0  = n_inf(V0)
    hT0 = hT_inf(V0)
    return np.array([V0, h0, n0, hT0], dtype=float)

# -----------------------------
# Current protocols
# -----------------------------
def base_plus_step_protocol(base_pA_val: float,
                            step_pA_val: float,
                            t_on_ms: float = 250.0,
                            t_off_ms: float = 500.0,
                            T_ms: float = 750.0) -> Callable[[float], float]:
    """
    Piecewise-constant current (A):
    base from t=0 to t_on; base+step in [t_on, t_off); base in [t_off, T].
    Times given in ms; converted to seconds internally.
    """
    t_on  = t_on_ms * ms
    t_off = t_off_ms * ms
    baseA = base_pA_val * pA
    stepA = step_pA_val * pA

    def Iinj(t: float) -> float:
        if t < t_on:
            return baseA
        elif t < t_off:
            return baseA + stepA
        else:
            return baseA
    return Iinj

# -----------------------------
# Generic RK4 (fixed step)
# -----------------------------
def rk4_step(f: Callable[[float, np.ndarray], np.ndarray],
             t: float,
             y: np.ndarray,
             dt: float,
             *args, **kwargs) -> np.ndarray:
    """One RK4 step (generic)."""
    k1 = f(t, y, *args, **kwargs)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, *args, **kwargs)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, *args, **kwargs)
    k4 = f(t + dt,      y + dt*k3,     *args, **kwargs)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_solve(f: Callable[[float, np.ndarray], np.ndarray],
              t0: float,
              t1: float,
              dt: float,
              y0: np.ndarray,
              *args,
              callback: Optional[Callable[[float, np.ndarray], Any]] = None,
              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate y' = f(t, y) from t0 to t1 with fixed-step RK4.
    Returns (t_array, Y) where Y[i] is the state at t[i].
    """
    t = np.arange(t0, t1 + 1e-15, dt)
    y = np.array(y0, dtype=float)
    Y = np.empty((t.size, y.size), dtype=float)
    for i, ti in enumerate(t):
        Y[i] = y
        if callback is not None:
            callback(ti, y)
        if i < t.size - 1:
            y = rk4_step(f, ti, y, dt, *args, **kwargs)
    return t, Y

# -----------------------------
# Spike detection helpers (optional for later items)
# -----------------------------
def find_upcrossings(t: np.ndarray, V: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Return indices i where V[i] < threshold and V[i+1] >= threshold (upward crossings).
    """
    return np.where((V[:-1] < threshold) & (V[1:] >= threshold))[0]

def count_spikes(t: np.ndarray,
                 V: np.ndarray,
                 threshold: float = 0.0,
                 refractory_ms: float = 2.0) -> int:
    """
    Count spikes by upward crossings with a refractory window (seconds).
    """
    idx = find_upcrossings(t, V, threshold)
    if idx.size == 0:
        return 0
    dt = t[1] - t[0]
    refr_idx = int(np.round((refractory_ms * ms) / dt))
    count, last = 1, idx[0]
    for k in idx[1:]:
        if k - last >= refr_idx:
            count += 1
            last = k
    return count
