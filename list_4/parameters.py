# parameters.py
# Constants for the two-compartment neuron model
# Units: S (siemens), F (farads), V (volts), A (amperes), s (seconds)

# ----------------------------
# Fractions of total membrane area
# ----------------------------
p = 1.0 / 3.0        # soma fraction
q = 1.0 - p          # dendrite fraction

# ----------------------------
# Leak conductances
# ----------------------------
gL_S = p * 5e-9      # S, soma leak (p × 5 nS)
gL_D = q * 5e-9      # S, dendrite leak ((1-p) × 5 nS)

# ----------------------------
# Maximum conductances
# ----------------------------
gNa    = p * 3e-6        # S, sodium current (p × 3 µS)
gK     = p * 2e-6        # S, delayed rectifier K⁺ (p × 2 µS)
gCa    = q * 2e-6        # S, calcium ( (1-p) × 2 µS )
gKCa   = q * 2.5e-6      # S, KCa ( (1-p) × 2.5 µS )
gKahp  = q * 40e-9       # S, KAHP ( (1-p) × 40 nS )
gc     = 20e-9           # S, coupling conductance (20 nS)

# ----------------------------
# Reversal potentials
# ----------------------------
ENa =  0.060   # V
EK  = -0.075   # V
ECa =  0.080   # V
EL  = -0.060   # V

# ----------------------------
# Capacitances
# ----------------------------
C_S = p * 100e-12   # F, soma capacitance (p × 100 pF)
C_D = q * 100e-12   # F, dendrite capacitance ((1-p) × 100 pF)

# ----------------------------
# Injected currents
# ----------------------------
Iinj_S = 0.0        # A
Iinj_D = 0.0        # A

# ----------------------------
# Calcium dynamics
# ----------------------------
tau_Ca = 50e-3                    # s, decay time constant (50 ms)
k      = 2.5e6 / (1.0 - p)        # M/C, charge-to-concentration factor

# ----------------------------
# Dictionary version 
# ----------------------------
PARAMS = {
    "p": p, "q": q,
    "gL_S": gL_S, "gL_D": gL_D,
    "gNa": gNa, "gK": gK,
    "gCa": gCa, "gKCa": gKCa, "gKahp": gKahp,
    "gc": gc,
    "ENa": ENa, "EK": EK, "ECa": ECa, "EL": EL,
    "C_S": C_S, "C_D": C_D,
    "Iinj_S": Iinj_S, "Iinj_D": Iinj_D,
    "tau_Ca": tau_Ca, "k": k,
}
