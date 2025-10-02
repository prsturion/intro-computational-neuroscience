# initial_values.py
# Initial state vector for the two-compartment neuron model
# Format:
# y = [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp]

import numpy as np
from parameters import PARAMS

# ----------------------------
# Initial voltages = leak reversal potential
# ----------------------------
Vs0 = PARAMS["EL"]   # soma initial potential [V]
Vd0 = PARAMS["EL"]   # dendrite initial potential [V]

# ----------------------------
# Initial calcium concentration
# ----------------------------
Ca0 = 0.0            # [M], starts at 0

# ----------------------------
# Gating variables
# ----------------------------
m0     = 0.0
h0     = 0.5
n0     = 0.4
mca0   = 0.0
mkca0  = 0.2
mkahp0 = 0.2

# ----------------------------
# State vector
# ----------------------------
Y0 = np.array([Vs0, Vd0, Ca0, m0, h0, n0, mca0, mkca0, mkahp0], dtype=float)

__all__ = ["Y0", "Vs0", "Vd0", "Ca0", "m0", "h0", "n0", "mca0", "mkca0", "mkahp0"]
