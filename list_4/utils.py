# utils.py
# Gating kinetics for somatic (Vs) and dendritic (Vd) compartments, in volts.
# Calcium-dependent gates use concentration [Ca] in the same units as your model.
# Vectorized with NumPy; accepts float or NumPy arrays.

from __future__ import annotations
import numpy as np
from typing import Dict, Sequence

ArrayLike = np.ndarray | float

# ----------------------------
# m-gate (somatic, Vs)
# ----------------------------
def alpha_m(Vs: ArrayLike) -> ArrayLike:
    """α_m(Vs) = 320e3*(Vs+0.0469) / (1 - exp(-250*(Vs+0.0469)))."""
    x = Vs + 0.0469
    return 320e3 * x / (-np.expm1(-250.0 * x))

def beta_m(Vs: ArrayLike) -> ArrayLike:
    """β_m(Vs) = 280e3*(Vs+0.0199) / (exp(200*(Vs+0.0199)) - 1)."""
    x = Vs + 0.0199
    return 280e3 * x / (np.expm1(200.0 * x))

def tau_m(Vs: ArrayLike) -> ArrayLike:
    """τ_m(Vs) = 1 / (α_m + β_m)."""
    a, b = alpha_m(Vs), beta_m(Vs)
    return 1.0 / (a + b + 1e-12)

def m_inf(Vs: ArrayLike) -> ArrayLike:
    """m∞(Vs) = α_m / (α_m + β_m)."""
    a, b = alpha_m(Vs), beta_m(Vs)
    return a / (a + b + 1e-12)

# ----------------------------
# h-gate (somatic, Vs)
# ----------------------------
def alpha_h(Vs: ArrayLike) -> ArrayLike:
    """α_h(Vs) = 128 * exp(-55.556*(Vs+0.043))."""
    return 128.0 * np.exp(-55.556 * (Vs + 0.043))

def beta_h(Vs: ArrayLike) -> ArrayLike:
    """β_h(Vs) = 4000 / (1 + exp(-200*(Vs+0.020)))."""
    return 4000.0 / (1.0 + np.exp(-200.0 * (Vs + 0.020)))

def tau_h(Vs: ArrayLike) -> ArrayLike:
    """τ_h(Vs) = 1 / (α_h + β_h)."""
    a, b = alpha_h(Vs), beta_h(Vs)
    return 1.0 / (a + b + 1e-12)

def h_inf(Vs: ArrayLike) -> ArrayLike:
    """h∞(Vs) = α_h / (α_h + β_h)."""
    a, b = alpha_h(Vs), beta_h(Vs)
    return a / (a + b + 1e-12)

# ----------------------------
# n-gate (somatic, Vs)
# ----------------------------
def alpha_n(Vs: ArrayLike) -> ArrayLike:
    """α_n(Vs) = 16e3*(Vs+0.0249) / (1 - exp(-200*(Vs+0.0249)))."""
    x = Vs + 0.0249
    return 16e3 * x / (-np.expm1(-200.0 * x))

def beta_n(Vs: ArrayLike) -> ArrayLike:
    """β_n(Vs) = 250 * exp(-25*(Vs+0.040))."""
    return 250.0 * np.exp(-25.0 * (Vs + 0.040))

def tau_n(Vs: ArrayLike) -> ArrayLike:
    """τ_n(Vs) = 1 / (α_n + β_n)."""
    a, b = alpha_n(Vs), beta_n(Vs)
    return 1.0 / (a + b + 1e-12)

def n_inf(Vs: ArrayLike) -> ArrayLike:
    """n∞(Vs) = α_n / (α_n + β_n)."""
    a, b = alpha_n(Vs), beta_n(Vs)
    return a / (a + b + 1e-12)

# ----------------------------
# m_ca gate (dendritic, Vd)
# ----------------------------
def alpha_mca(Vd: ArrayLike) -> ArrayLike:
    """α_mca(Vd) = 1600 / (1 + exp(-72*(Vd - 0.005)))."""
    return 1600.0 / (1.0 + np.exp(-72.0 * (Vd - 0.005)))

def beta_mca(Vd: ArrayLike) -> ArrayLike:
    """β_mca(Vd) = 2e4*(Vd+0.0089) / (exp(200*(Vd+0.0089)) - 1)."""
    x = Vd + 0.0089
    return 2e4 * x / (np.expm1(200.0 * x))

def tau_mca(Vd: ArrayLike) -> ArrayLike:
    """τ_mca(Vd) = 1 / (α_mca + β_mca)."""
    a, b = alpha_mca(Vd), beta_mca(Vd)
    return 1.0 / (a + b + 1e-12)

def mca_inf(Vd: ArrayLike) -> ArrayLike:
    """m_ca∞(Vd) = α_mca / (α_mca + β_mca)."""
    a, b = alpha_mca(Vd), beta_mca(Vd)
    return a / (a + b + 1e-12)

# ----------------------------
# m_kca gate (piecewise, dendritic, Vd)
# ----------------------------
def alpha_mkca(Vd: ArrayLike) -> ArrayLike:
    """Piecewise α_mkca(Vd):
       if Vd > -0.010: 2000 * exp(-37.037 * (Vd + 0.0535))
       else:           exp((Vd+0.050)/0.011 - (Vd+0.0535)/0.027) / 0.018975
    """
    Vd_arr = np.asarray(Vd)
    out = np.empty_like(Vd_arr, dtype=float)
    mask = Vd_arr > -0.010
    out[mask] = 2000.0 * np.exp(-37.037 * (Vd_arr[mask] + 0.0535))
    out[~mask] = np.exp((Vd_arr[~mask] + 0.050)/0.011 - (Vd_arr[~mask] + 0.0535)/0.027) / 0.018975
    return out

def beta_mkca(Vd: ArrayLike) -> ArrayLike:
    """Piecewise β_mkca(Vd):
       if Vd > -0.010: 0
       else:           2000 * exp(-(Vd + 0.0535)/0.027) - α_mkca(Vd)
    """
    Vd_arr = np.asarray(Vd)
    out = np.empty_like(Vd_arr, dtype=float)
    mask = Vd_arr > -0.010
    out[mask] = 0.0
    out[~mask] = 2000.0 * np.exp(-(Vd_arr[~mask] + 0.0535)/0.027) - alpha_mkca(Vd_arr[~mask])
    return out

def tau_mkca(Vd: ArrayLike) -> ArrayLike:
    """τ_mKCa(Vd) = 1 / (α_mkca + β_mkca)."""
    a, b = alpha_mkca(Vd), beta_mkca(Vd)
    return 1.0 / (a + b + 1e-12)

def mkca_inf(Vd: ArrayLike) -> ArrayLike:
    """m_KCa∞(Vd) = α_mkca / (α_mkca + β_mkca)."""
    a, b = alpha_mkca(Vd), beta_mkca(Vd)
    return a / (a + b + 1e-12)

# ----------------------------
# K_AHP (calcium-dependent; uses [Ca])
# ----------------------------
def chi(ca: ArrayLike) -> ArrayLike:
    """χ([Ca]) = min(4000*[Ca], 1)."""
    ca_arr = np.asarray(ca, dtype=float)
    return np.minimum(4000.0 * ca_arr, 1.0)

def alpha_mkahp(ca: ArrayLike) -> ArrayLike:
    """α_mKAHp([Ca]) = min(20, 20000*[Ca])."""
    ca_arr = np.asarray(ca, dtype=float)
    return np.minimum(20.0, 20000.0 * ca_arr)

def beta_mkahp(ca: ArrayLike) -> ArrayLike:
    """β_mKAHp([Ca]) = 4 (constant)."""
    # Broadcast to the same shape as ca
    return np.full_like(np.asarray(ca, dtype=float), 4.0)

def tau_mkahp(ca: ArrayLike) -> ArrayLike:
    """τ_mKAHp([Ca]) = 1 / (α_mKAHp + β_mKAHp)."""
    a, b = alpha_mkahp(ca), beta_mkahp(ca)
    return 1.0 / (a + b + 1e-12)

def mkahp_inf(ca: ArrayLike) -> ArrayLike:
    """m_KAHP∞([Ca]) = α_mKAHp / (α_mKAHp + β_mKAHp)."""
    a, b = alpha_mkahp(ca), beta_mkahp(ca)
    return a / (a + b + 1e-12)



# ============================
# Differential Equations (ODEs) — with detailed docstrings
# ============================

def rhs_somatic(t: float, y: Sequence[float], params: Dict[str, float]) -> ArrayLike:
    """
    Right-hand side of the **somatic membrane voltage** ODE (Eq. 1).

    Model:
        C_S * dVs/dt = - gL_S*(Vs - EL) - gNa*m^2*h*(Vs - ENa) - gK*n^2*(Vs - EK)
                       + gc*(Vd - Vs) + Iinj_S

    Parameters
    ----------
    t : float
        Time [s]. (Not used explicitly here, but included for solver compatibility.)
    y : Sequence[float]
        State vector ordered as:
            [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp]
        Units:
            Vs, Vd in volts [V]; Ca in model's [Ca] units; gates are dimensionless in [0, 1].
    params : Dict[str, float]
        Dictionary of constants (all SI unless noted):
            C_S     : Somatic capacitance [F]
            gL_S    : Somatic leak conductance [S]
            EL      : Leak reversal potential [V]
            gNa     : Max sodium conductance [S]
            ENa     : Sodium reversal potential [V]
            gK      : Max potassium (delayed rectifier) conductance [S]
            EK      : Potassium reversal potential [V]
            gc      : Coupling conductance soma↔dendrite [S]
            Iinj_S  : Injected somatic current [A]

    Returns
    -------
    float
        dVs/dt [V/s]
    """
    Vs, Vd, *_ = y
    m, h, n = y[3], y[4], y[5]
    p = params

    INa = p["gNa"] * (m**2) * h * (Vs - p["ENa"])
    IK  = p["gK"]  * (n**2) * (Vs - p["EK"])
    IL  = p["gL_S"] * (Vs - p["EL"])
    Ic  = p["gc"] * (Vd - Vs)

    Iinj_S = p["Iinj_S"](float(t)) if isinstance(p["Iinj_S"], Callable) else p["Iinj_S"]
    dVsdt = (-IL - INa - IK + Ic + Iinj_S) / p["C_S"]

    return dVsdt


def rhs_dendritic(t: float, y: Sequence[float], params: Dict[str, float]) -> ArrayLike:
    """
    Right-hand side of the **dendritic membrane voltage** ODE (Eq. 2).

    Model:
        C_D * dVd/dt = - gL_D*(Vd - EL) - gCa*mca^2*(Vd - ECa) - gKCa*mkca*chi(Ca)*(Vd - EK)
                       - gKahp*mkahp*(Vd - EK) + gc*(Vs - Vd) + Iinj_D

    Notes
    -----
    - The factor `chi(Ca)` = min(4000*Ca, 1) already exists in utils (function `chi`).
    - Calcium-dependent gates (`mkca`, `mkahp`) are taken from `y`.

    Parameters
    ----------
    t : float
        Time [s]. (Unused; present for solver signature consistency.)
    y : Sequence[float]
        State vector:
            [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp]
    params : Dict[str, float]
        Constants:
            C_D     : Dendritic capacitance [F]
            gL_D    : Dendritic leak conductance [S]
            EL      : Leak reversal [V]
            gCa     : Max Ca²⁺ conductance [S]
            ECa     : Ca²⁺ reversal [V]
            gKCa    : Max KCa conductance [S]
            gKahp   : Max KAHP conductance [S]
            EK      : K⁺ reversal [V]
            gc      : Coupling conductance soma↔dendrite [S]
            Iinj_D  : Injected dendritic current [A]

    Returns
    -------
    float
        dVd/dt [V/s]
    """
    Vs, Vd, Ca, *_ = y
    mca, mkca, mkahp = y[6], y[7], y[8]
    p = params

    ICa   = p["gCa"]   * (mca**2) * (Vd - p["ECa"])
    IKCa  = p["gKCa"]  * mkca * (Vd - p["EK"])
    IKahp = p["gKahp"] * mkahp * (Vd - p["EK"])
    IL    = p["gL_D"] * (Vd - p["EL"])
    Ic    = p["gc"] * (Vs - Vd)

    chi_val = chi(Ca)
    
    Iinj_D = p["Iinj_D"](float(t)) if isinstance(p["Iinj_D"], Callable) else p["Iinj_D"]
    
    dVddt = (-IL - ICa - IKCa * chi_val - IKahp + Ic + Iinj_D) / p["C_D"]
    return dVddt


def rhs_calcium(t: float, y: Sequence[float], params: Dict[str, float]) -> ArrayLike:
    """
    Right-hand side of the **intracellular calcium** ODE (Eq. 3).

    Model:
        d[Ca]/dt = - [Ca]/tau_Ca  -  k * gCa * mca^2 * (Vd - ECa)

    Parameters
    ----------
    t : float
        Time [s]. (Unused directly.)
    y : Sequence[float]
        State vector:
            [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp]
    params : Dict[str, float]
        Constants:
            tau_Ca : Calcium decay time constant [s]
            k      : Scaling factor for Ca²⁺ influx (model-specific units)
            gCa    : Max Ca²⁺ conductance [S]
            ECa    : Ca²⁺ reversal [V]

    Returns
    -------
    float
        d[Ca]/dt in model's [Ca]/s units
    """
    Vd, Ca = y[1], y[2]
    mca = y[6]
    p = params

    dCadt = -Ca / p["tau_Ca"] - p["k"] * p["gCa"] * (mca**2) * (Vd - p["ECa"])
    return dCadt


def rhs_gates(t: float, y: Sequence[float], params: Dict[str, float]) -> tuple[ArrayLike, ...]:
    """
    Right-hand side of the **gating variables** ODEs (Eqs. 4–9).

    Generic form:
        dx/dt = (x_inf(V) - x) / tau_x(V)   for voltage-gated x
        dx/dt = (x_inf(Ca) - x) / tau_x(Ca) for Ca-gated x

    Parameters
    ----------
    t : float
        Time [s]. (Unused; included for API compatibility.)
    y : Sequence[float]
        State vector:
            [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp]
    params : Dict[str, float]
        (Not used here, but included for a uniform callable signature.)

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Derivatives of (m, h, n, mca, mkca, mkahp) in this order.
    """
    Vs, Vd, Ca = y[0], y[1], y[2]
    m, h, n, mca, mkca, mkahp = y[3], y[4], y[5], y[6], y[7], y[8]

    dm     = (m_inf(Vs)     - m)     / tau_m(Vs)
    dh     = (h_inf(Vs)     - h)     / tau_h(Vs)
    dn     = (n_inf(Vs)     - n)     / tau_n(Vs)
    dmca   = (mca_inf(Vd)   - mca)   / tau_mca(Vd)
    dmkca  = (mkca_inf(Vd)  - mkca)  / tau_mkca(Vd)
    dmkahp = (mkahp_inf(Ca) - mkahp) / tau_mkahp(Ca)

    return dm, dh, dn, dmca, dmkca, dmkahp


def rhs_full(t: float, y: Sequence[float], params: Dict[str, float]) -> np.ndarray:
    """
    Full system right-hand side combining Eqs. (1)–(9).

    State and Ordering
    ------------------
    y = [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp]

    Required `params` keys
    ----------------------
    Electrical:
        C_S, gL_S, EL, gNa, ENa, gK, EK, gc, Iinj_S
        C_D, gL_D, gCa, ECa, gKCa, gKahp, Iinj_D
    Calcium:
        tau_Ca, k

    Returns
    -------
    np.ndarray
        dy/dt in the same ordering as `y`:
        [dVs, dVd, dCa, dm, dh, dn, dmca, dmkca, dmkahp]
    """
    dVs = rhs_somatic(t, y, params)
    dVd = rhs_dendritic(t, y, params)
    dCa = rhs_calcium(t, y, params)
    dm, dh, dn, dmca, dmkca, dmkahp = rhs_gates(t, y, params)
    return np.array([dVs, dVd, dCa, dm, dh, dn, dmca, dmkca, dmkahp], dtype=float)



# ============================
# Runge–Kutta 4 (fixed step)
# ============================

from typing import Callable, Tuple, Sequence, Any
import numpy as np

def rk4_step(
    fun: Callable[..., np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
    *fargs: Any,
    **fkwargs: Any
) -> np.ndarray:
    """
    Single RK4 step (explicit, fixed step).

    Parameters
    ----------
    fun : callable
        Right-hand side f(t, y, *fargs, **fkwargs) -> dy/dt.
        Must return an array with the same shape as `y`.
    t : float
        Current time.
    y : array_like
        Current state vector.
    h : float
        Step size (Δt).
    *fargs, **fkwargs :
        Extra parameters forwarded to `fun`. For our model, pass `params`:
        e.g., rk4_step(rhs_full, t, y, h, params).

    Returns
    -------
    np.ndarray
        The state y_{n+1} after one RK4 step of size `h`.
    """
    y = np.asarray(y, dtype=float)

    k1 = np.asarray(fun(t,           y,             *fargs, **fkwargs), dtype=float)
    k2 = np.asarray(fun(t + 0.5*h,   y + 0.5*h*k1,  *fargs, **fkwargs), dtype=float)
    k3 = np.asarray(fun(t + 0.5*h,   y + 0.5*h*k2,  *fargs, **fkwargs), dtype=float)
    k4 = np.asarray(fun(t + h,       y + h*k3,      *fargs, **fkwargs), dtype=float)

    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(
    fun: Callable[..., np.ndarray],
    t_span: Tuple[float, float],
    y0: Sequence[float],
    h: float,
    *fargs: Any,
    **fkwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE system using classic RK4 with a fixed step.

    Parameters
    ----------
    fun : callable
        RHS f(t, y, *fargs, **fkwargs) -> dy/dt.
        For your model: `rhs_full`.
    t_span : (t0, tf)
        Start and end times.
    y0 : sequence
        Initial state vector at t0 (shape = [n_states]).
    h : float
        Base step size. The last step is shortened to land exactly on `tf`.
    *fargs, **fkwargs :
        Extra arguments passed to `fun` on every call (e.g., `params` dict).

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time grid including both endpoints.
    Y : np.ndarray, shape (N, n_states)
        State trajectory where Y[i] corresponds to time t[i].

    Notes
    -----
    - This is a *fixed-step* explicit solver; choose `h` small enough
      for stability/precision desired.
    - If you want to vectorize simulations, you can stack `y0` in a matrix
      (n_states,) per case and adapt `fun` to accept a batch.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    if tf < t0:
        raise ValueError("t_span must satisfy tf >= t0.")

    y = np.asarray(y0, dtype=float).copy()
    t_values = [t0]
    y_values = [y.copy()]

    t = t0
    while t < tf:
        h_step = min(h, tf - t)           # adjust last step
        y = rk4_step(fun, t, y, h_step, *fargs, **fkwargs)
        t = t + h_step
        t_values.append(t)
        y_values.append(y.copy())

    t_arr = np.array(t_values, dtype=float)
    Y_arr = np.vstack(y_values).astype(float)  # shape (N, n_states)
    return t_arr, Y_arr
