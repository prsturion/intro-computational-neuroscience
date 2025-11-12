
# utils.py
# Reusable utilities for HH-style E/I/O neurons + AMPA/GABAA synapses
# Units: V in mV, t in ms, conductances in mS/cm^2, capacitance in µF/cm^2.
# Currents are in µA/cm^2 so that dV/dt = I / C_m is consistent.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np

Array = np.ndarray

# ----------------------------
# Numeric helpers
# ----------------------------

def _vtrap_pos(x: Array) -> Array:
    """Stable compute x/(exp(x)-1). Works for scalar or array x."""
    # np.expm1(x) = exp(x) - 1 with better precision near 0
    y = np.expm1(x)
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-6
    out[~small] = x[~small] / y[~small]
    # series expansion: x/(e^x - 1) ~ 1 - x/2 + x^2/12 - ...
    out[small] = 1.0 - 0.5*x[small] + (x[small]**2)/12.0
    return out

def _vtrap_neg(x: Array) -> Array:
    """Stable compute x/(1 - exp(-x))."""
    y = -np.expm1(-x)  # 1 - exp(-x)
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-6
    out[~small] = x[~small] / y[~small]
    # series: x/(1-e^{-x}) ~ 1 + x/2 + x^2/12 + ...
    out[small] = 1.0 + 0.5*x[small] + (x[small]**2)/12.0
    return out

def sigmoid(x: Array) -> Array:
    return 1.0/(1.0 + np.exp(-x))

# ----------------------------
# Parameter sets
# ----------------------------

@dataclass
class NeuronParams:
    Cm: float
    gNa: float
    gK: float
    gL: float
    ENa: float
    EK: float
    EL: float
    # Optional currents (used by OLM neuron)
    gA: float = 0.0
    EA: float = 0.0
    gh: float = 0.0
    Eh: float = 0.0

def params_E() -> NeuronParams:
    return NeuronParams(
        Cm=1.0, gNa=100.0, gK=80.0, gL=0.1, ENa=50.0, EK=-100.0, EL=-67.0
    )

def params_I() -> NeuronParams:
    return NeuronParams(
        Cm=1.0, gNa=35.0, gK=9.0, gL=0.1, ENa=55.0, EK=-90.0, EL=-65.0
    )

def params_O() -> NeuronParams:
    return NeuronParams(
        Cm=1.3, gNa=30.0, gK=23.0, gL=0.05, ENa=90.0, EK=-100.0, EL=-70.0,
        gA=16.0, EA=-90.0, gh=12.0, Eh=-32.9
    )

# ----------------------------
# Gating rate functions (alpha/beta) for each model
# ----------------------------

def rates_E(V: Array) -> Dict[str, Array]:
    V = np.asarray(V, dtype=float)
    am = 0.32 * _vtrap_neg((V + 54.0)/4.0) * 4.0  # rewrite to use vtrap
    bm = 0.28 * _vtrap_pos((V + 27.0)/5.0) * 5.0
    ah = 0.128 * np.exp(-(V + 50.0)/18.0)
    bh = 4.0 / (1.0 + np.exp(-(V + 27.0)/5.0))
    an = 0.032 * _vtrap_neg((V + 52.0)/5.0) * 5.0
    bn = 0.5 * np.exp(-(V + 57.0)/40.0)
    return {"am": am, "bm": bm, "ah": ah, "bh": bh, "an": an, "bn": bn}

def rates_I(V: Array) -> Dict[str, Array]:
    V = np.asarray(V, dtype=float)
    am = 0.1 * _vtrap_neg((V + 35.0)/10.0) * 10.0
    bm = 4.0 * np.exp(-(V + 60.0)/18.0)
    ah = 0.07 * np.exp(-(V + 58.0)/20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 28.0)/10.0))
    an = 0.01 * _vtrap_neg((V + 34.0)/10.0) * 10.0
    bn = 0.125 * np.exp(-(V + 44.0)/80.0)
    return {"am": am, "bm": bm, "ah": ah, "bh": bh, "an": an, "bn": bn}

def rates_O(V: Array) -> Dict[str, Array]:
    V = np.asarray(V, dtype=float)
    am = -0.1 * _vtrap_pos((V + 38.0)/10.0) * 10.0  # -0.1(V+38)/(exp(-(V+38)/10)-1) = -0.1*10*((V+38)/10)/(exp(-(V+38)/10)-1)
    bm = 4.0 * np.exp(-(V + 65.0)/18.0)
    ah = 0.07 * np.exp(-(V + 63.0)/20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 33.0)/10.0))
    an = 0.018 * _vtrap_neg((V - 25.0)/25.0) * 25.0
    bn = 0.0036 * _vtrap_pos((V - 35.0)/12.0) * 12.0
    return {"am": am, "bm": bm, "ah": ah, "bh": bh, "an": an, "bn": bn}

def m_inf_from_rates(am: Array, bm: Array) -> Array:
    return am/(am + bm)

def x_inf(am: Array, bm: Array) -> Array:
    return am/(am + bm)

def tau_from_rates(am: Array, bm: Array, scale: float = 1.0) -> Array:
    return scale/(am + bm)

# Additional gates for OLM neuron
def a_inf_O(V: Array) -> Array:
    return 1.0/(1.0 + np.exp(-(V + 14.0)/16.6))

def tau_a_O(V: Array) -> Array:
    return np.full_like(np.asarray(V, dtype=float), 5.0)

def b_inf_O(V: Array) -> Array:
    return 1.0/(1.0 + np.exp((V + 71.0)/7.3))

def tau_b_O(V: Array) -> Array:
    V = np.asarray(V, dtype=float)
    # 1 / ( 0.000009/exp((V-26)/18.5) + 0.014/(0.2 + exp(-(V+70)/11)) )
    term1 = 0.000009/np.exp((V - 26.0)/18.5)
    term2 = 0.014/(0.2 + np.exp(-(V + 70.0)/11.0))
    return 1.0/(term1 + term2)

def r_inf_O(V: Array) -> Array:
    return 1.0/(1.0 + np.exp((V + 84.0)/10.2))

def tau_r_O(V: Array) -> Array:
    V = np.asarray(V, dtype=float)
    return 1.0/(np.exp(-14.59 - 0.086*V) + np.exp(-1.87 + 0.0701*V))

# ----------------------------
# Synapses
# ----------------------------

@dataclass
class SynParams:
    gbar: float
    tau_s: float
    tau_d: float
    Erev: float

def ampa(gbar: float) -> SynParams:
    # From text: tau_s=0.1 ms, tau_d=3 ms, Erev=0 mV
    return SynParams(gbar=gbar, tau_s=0.1, tau_d=3.0, Erev=0.0)

def gabaa_I(gbar: float) -> SynParams:
    # For synapses made by I-type inhibitory neuron
    return SynParams(gbar=gbar, tau_s=0.3, tau_d=9.0, Erev=-80.0)

def gabaa_O(gbar: float) -> SynParams:
    # For synapses made by O-type inhibitory neuron
    return SynParams(gbar=gbar, tau_s=0.2, tau_d=20.0, Erev=-80.0)

# ----------------------------
# Network model
# ----------------------------

class EIONetwork:
    """
    Network with Hodgkin–Huxley pyramidal E, fast-spiking I, and OLM O neurons.
    Uses Runge–Kutta 4 integrator with time step dt.
    """
    def __init__(self, neuron_types: List[str]):
        """
        neuron_types: list like ['E', 'I', 'O'] describing each neuron.
        """
        self.neuron_types = neuron_types
        self.N = len(neuron_types)

        # Parameters per neuron
        self.params: List[NeuronParams] = []
        for t in neuron_types:
            if t.upper() == 'E':
                self.params.append(params_E())
            elif t.upper() == 'I':
                self.params.append(params_I())
            elif t.upper() == 'O':
                self.params.append(params_O())
            else:
                raise ValueError(f"Unknown neuron type {t}")

        # Initial states
        self.V = np.array([p.EL for p in self.params], dtype=float)

        # gating variables: allocate for all, even if unused, and fill with steady-state at rest
        self.h = np.zeros(self.N)
        self.n = np.zeros(self.N)
        self.a = np.zeros(self.N)
        self.b = np.zeros(self.N)
        self.r = np.zeros(self.N)
        for i, t in enumerate(self.neuron_types):
            if t.upper() in ('E', 'O'):
                rts = rates_E(self.V[i]) if t.upper() == 'E' else rates_O(self.V[i])
                self.h[i] = x_inf(rts['ah'], rts['bh'])
                self.n[i] = x_inf(rts['an'], rts['bn'])
            else:  # I
                rts = rates_I(self.V[i])
                self.h[i] = x_inf(rts['ah'], rts['bh'])
                self.n[i] = x_inf(rts['an'], rts['bn'])
            if t.upper() == 'O':
                self.a[i] = a_inf_O(self.V[i])
                self.b[i] = b_inf_O(self.V[i])
                self.r[i] = r_inf_O(self.V[i])

        # Edges: list of (pre, post, SynParams, s)
        self.pre: List[int] = []
        self.post: List[int] = []
        self.syn: List[SynParams] = []
        self.s: List[float] = []

        # External current (µA/cm^2): function f(t, i) -> float
        self.Iinj: Optional[Callable[[float, int], float]] = None

    def set_injection(self, func: Optional[Callable[[float, int], float]]):
        """Set external current function f(t, i) in µA/cm^2. Use None for zero injection."""
        self.Iinj = func

    def add_connection(self, i_pre: int, j_post: int, gbar: float, kind: Optional[str] = None):
        """
        Add a synapse from neuron i_pre to neuron j_post.
        If kind is None, it is inferred from pre type: 'AMPA' for E, 'GABAA_I' for I, 'GABAA_O' for O.
        """
        pre_type = self.neuron_types[i_pre].upper()
        if kind is None:
            if pre_type == 'E':
                sp = ampa(gbar)
            elif pre_type == 'I':
                sp = gabaa_I(gbar)
            elif pre_type == 'O':
                sp = gabaa_O(gbar)
            else:
                raise ValueError(f"Unknown pre type {pre_type}")
        else:
            k = kind.upper()
            if k == 'AMPA':
                sp = ampa(gbar)
            elif k in ('GABAA_I', 'GABAA'):
                sp = gabaa_I(gbar)
            elif k == 'GABAA_O':
                sp = gabaa_O(gbar)
            else:
                raise ValueError(f"Unknown synapse kind {kind}")

        self.pre.append(i_pre)
        self.post.append(j_post)
        self.syn.append(sp)
        self.s.append(0.0)  # start inactive

    # -------- system derivatives --------

    def _syn_current(self, V: Array, s: Array) -> Array:
        """Compute total synaptic current into each neuron given V and edge activations s."""
        I = np.zeros(self.N, dtype=float)
        for e, (i, j) in enumerate(zip(self.pre, self.post)):
            sp = self.syn[e]
            I[j] += sp.gbar * s[e] * (sp.Erev - V[j])
        return I

    def _ds_dt(self, t: float, V: Array, s: Array) -> Array:
        """ds/dt for each edge (depends on V_pre and s)."""
        ds = np.zeros_like(s)
        for e, i in enumerate(self.pre):
            sp = self.syn[e]
            drive = 0.5*(1.0 + np.tanh(V[i]/4.0))  # presynaptic V in mV
            ds[e] = drive*(1.0 - s[e])/sp.tau_s - s[e]/sp.tau_d
        return ds

    def _neuron_derivs(self, t: float, V: Array, h: Array, n: Array, a: Array, b: Array, r: Array, s: Array) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """Return (dV, dh, dn, da, db, dr)."""
        # Synaptic current into each neuron
        Isyn = self._syn_current(V, s)

        dV = np.zeros(self.N, dtype=float)
        dh = np.zeros(self.N, dtype=float)
        dn = np.zeros(self.N, dtype=float)
        da = np.zeros(self.N, dtype=float)
        db = np.zeros(self.N, dtype=float)
        dr = np.zeros(self.N, dtype=float)

        for i, tcode in enumerate(self.neuron_types):
            P = self.params[i]
            # External injection
            Iinj_i = 0.0 if self.Iinj is None else float(self.Iinj(t, i))

            if tcode.upper() == 'E':
                rt = rates_E(V[i])
                m_inf = m_inf_from_rates(rt['am'], rt['bm'])
                tau_h = tau_from_rates(rt['ah'], rt['bh'])
                tau_n = tau_from_rates(rt['an'], rt['bn'])
                I_Na = P.gNa * (m_inf**3) * h[i] * (P.ENa - V[i])
                I_K  = P.gK  * (n[i]**4)   * (P.EK  - V[i])
                I_L  = P.gL  * (P.EL - V[i])
                dV[i] = (I_Na + I_K + I_L + Isyn[i] + Iinj_i)/P.Cm
                dh[i] = (x_inf(rt['ah'], rt['bh']) - h[i])/tau_h
                dn[i] = (x_inf(rt['an'], rt['bn']) - n[i])/tau_n

            elif tcode.upper() == 'I':
                rt = rates_I(V[i])
                m_inf = m_inf_from_rates(rt['am'], rt['bm'])
                # Note: tau_x = 0.2/(alpha+beta) for h and n
                tau_h = tau_from_rates(rt['ah'], rt['bh'], scale=0.2)
                tau_n = tau_from_rates(rt['an'], rt['bn'], scale=0.2)
                I_Na = P.gNa * (m_inf**3) * h[i] * (P.ENa - V[i])
                I_K  = P.gK  * (n[i]**4)   * (P.EK  - V[i])
                I_L  = P.gL  * (P.EL - V[i])
                dV[i] = (I_Na + I_K + I_L + Isyn[i] + Iinj_i)/P.Cm
                dh[i] = (x_inf(rt['ah'], rt['bh']) - h[i])/tau_h
                dn[i] = (x_inf(rt['an'], rt['bn']) - n[i])/tau_n

            elif tcode.upper() == 'O':
                rt = rates_O(V[i])
                m_inf = m_inf_from_rates(rt['am'], rt['bm'])
                tau_h = tau_from_rates(rt['ah'], rt['bh'])
                tau_n = tau_from_rates(rt['an'], rt['bn'])
                # additional gates for O: a, b, r
                a_inf = a_inf_O(V[i]); ta = tau_a_O(V[i])
                b_inf = b_inf_O(V[i]); tb = tau_b_O(V[i])
                r_inf = r_inf_O(V[i]); tr = tau_r_O(V[i])
                I_Na = P.gNa * (m_inf**3) * h[i] * (P.ENa - V[i])
                I_K  = P.gK  * (n[i]**4)   * (P.EK  - V[i])
                I_A  = P.gA  * a[i]*b[i]   * (P.EA  - V[i])
                I_h  = P.gh  * r[i]        * (P.Eh  - V[i])
                I_L  = P.gL  * (P.EL - V[i])
                dV[i] = (I_Na + I_K + I_A + I_h + I_L + Isyn[i] + Iinj_i)/P.Cm
                dh[i] = (x_inf(rt['ah'], rt['bh']) - h[i])/tau_h
                dn[i] = (x_inf(rt['an'], rt['bn']) - n[i])/tau_n
                da[i] = (a_inf - a[i])/ta
                db[i] = (b_inf - b[i])/tb
                dr[i] = (r_inf - r[i])/tr

            else:
                raise ValueError(f"Unknown neuron type {tcode}")

        return dV, dh, dn, da, db, dr

    # -------- integration (RK4) --------

    def _rk4_step(self, t: float, dt: float):
        """Advance all states by one RK4 step of size dt."""
        V0, h0, n0, a0, b0, r0 = self.V.copy(), self.h.copy(), self.n.copy(), self.a.copy(), self.b.copy(), self.r.copy()
        s0 = np.array(self.s, dtype=float)

        # k1
        dV1, dh1, dn1, da1, db1, dr1 = self._neuron_derivs(t, V0, h0, n0, a0, b0, r0, s0)
        ds1 = self._ds_dt(t, V0, s0)

        # k2
        V2 = V0 + 0.5*dt*dV1
        h2 = h0 + 0.5*dt*dh1
        n2 = n0 + 0.5*dt*dn1
        a2 = a0 + 0.5*dt*da1
        b2 = b0 + 0.5*dt*db1
        r2 = r0 + 0.5*dt*dr1
        s2 = s0 + 0.5*dt*ds1
        dV2, dh2, dn2, da2, db2, dr2 = self._neuron_derivs(t + 0.5*dt, V2, h2, n2, a2, b2, r2, s2)
        ds2 = self._ds_dt(t + 0.5*dt, V2, s2)

        # k3
        V3 = V0 + 0.5*dt*dV2
        h3 = h0 + 0.5*dt*dh2
        n3 = n0 + 0.5*dt*dn2
        a3 = a0 + 0.5*dt*da2
        b3 = b0 + 0.5*dt*db2
        r3 = r0 + 0.5*dt*dr2
        s3 = s0 + 0.5*dt*ds2
        dV3, dh3, dn3, da3, db3, dr3 = self._neuron_derivs(t + 0.5*dt, V3, h3, n3, a3, b3, r3, s3)
        ds3 = self._ds_dt(t + 0.5*dt, V3, s3)

        # k4
        V4 = V0 + dt*dV3
        h4 = h0 + dt*dh3
        n4 = n0 + dt*dn3
        a4 = a0 + dt*da3
        b4 = b0 + dt*db3
        r4 = r0 + dt*dr3
        s4 = s0 + dt*ds3
        dV4, dh4, dn4, da4, db4, dr4 = self._neuron_derivs(t + dt, V4, h4, n4, a4, b4, r4, s4)
        ds4 = self._ds_dt(t + dt, V4, s4)

        # combine
        self.V += (dt/6.0)*(dV1 + 2*dV2 + 2*dV3 + dV4)
        self.h += (dt/6.0)*(dh1 + 2*dh2 + 2*dh3 + dh4)
        self.n += (dt/6.0)*(dn1 + 2*dn2 + 2*dn3 + dn4)
        self.a += (dt/6.0)*(da1 + 2*da2 + 2*da3 + da4)
        self.b += (dt/6.0)*(db1 + 2*db2 + 2*db3 + db4)
        self.r += (dt/6.0)*(dr1 + 2*dr2 + 2*dr3 + dr4)
        s_new = s0 + (dt/6.0)*(ds1 + 2*ds2 + 2*ds3 + ds4)
        self.s = [float(x) for x in s_new]

    # -------- Public API --------

    def simulate(self, T: float, dt: float = 0.01, record_every: int = 1, record_gates: bool = False) -> Dict[str, Array]:
        """
        Run the simulation.
        T: total time in ms.
        dt: time step in ms (default 0.01 ms as requested).
        record_every: store every k steps to reduce memory (default 1 = store all).
        record_gates: if True, also return h,n,a,b,r and per-edge s over time.
        """
        steps = int(np.round(T/dt))
        nrec = steps//record_every + 1
        t_out = np.empty(nrec, dtype=float)
        V_out = np.empty((nrec, self.N), dtype=float)
        if record_gates:
            h_out = np.empty_like(V_out); n_out = np.empty_like(V_out)
            a_out = np.empty_like(V_out); b_out = np.empty_like(V_out); r_out = np.empty_like(V_out)
            s_out = np.empty((nrec, len(self.s)), dtype=float)

        idx = 0
        t = 0.0
        t_out[idx] = t; V_out[idx] = self.V
        if record_gates:
            h_out[idx]=self.h; n_out[idx]=self.n; a_out[idx]=self.a; b_out[idx]=self.b; r_out[idx]=self.r; s_out[idx]=np.array(self.s)
        idx += 1

        for k in range(steps):
            self._rk4_step(t, dt)
            t += dt
            if (k+1) % record_every == 0:
                t_out[idx] = t; V_out[idx] = self.V
                if record_gates:
                    h_out[idx]=self.h; n_out[idx]=self.n; a_out[idx]=self.a; b_out[idx]=self.b; r_out[idx]=self.r; s_out[idx]=np.array(self.s)
                idx += 1

        out = {"t": t_out, "V": V_out}
        if record_gates:
            out.update({"h": h_out, "n": n_out, "a": a_out, "b": b_out, "r": r_out, "s": s_out,
                        "edges": np.array(list(zip(self.pre, self.post)), dtype=int)})
        return out

# ----------------------------
# Convenience builders
# ----------------------------

def build_default_EIO_network() -> EIONetwork:
    """
    Create a minimal E-I-O network (one neuron of each type) with typical connections:
    E -> I (AMPA), E -> O (AMPA), E -> E (AMPA, optional not added here),
    I -> E (GABAA_I), I -> O (GABAA_I),
    O -> E (GABAA_O).
    Conductances are left to be user-tuned; this returns an empty network ready for add_connection.
    """
    return EIONetwork(['E', 'I', 'O'])

def constant_current(I: float) -> Callable[[float, int], float]:
    """Return f(t, i)=I (µA/cm^2)."""
    return lambda t, i: I

def step_current(I0: float, t_on: float, t_off: float) -> Callable[[float, int], float]:
    """Step current: I0 between t_on and t_off (ms), else 0."""
    def f(t: float, i: int) -> float:
        return I0 if (t_on <= t <= t_off) else 0.0
    return f

if __name__ == "__main__":
    # Tiny smoke test (no plotting): three neurons, a few synapses, short run.
    net = build_default_EIO_network()
    # Typical connections (example values you will likely adjust per exercise):
    net.add_connection(0, 1, gbar=0.1)  # E -> I (AMPA)
    net.add_connection(0, 2, gbar=0.05) # E -> O (AMPA)
    net.add_connection(1, 0, gbar=0.2)  # I -> E (GABAA_I)
    net.add_connection(1, 2, gbar=0.05) # I -> O (GABAA_I)
    net.add_connection(2, 0, gbar=0.05) # O -> E (GABAA_O)

    net.set_injection(step_current(I0=1.0, t_on=10.0, t_off=50.0))  # inject into all neurons equally

    out = net.simulate(T=5.0, dt=0.01, record_every=10, record_gates=False)
    print("Simulated", out["t"][-1], "ms; V shape:", out["V"].shape)
