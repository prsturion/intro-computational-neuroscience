# initial_values_ex6.py
# Initial state vector for the two-compartment neuron model with Ih current.
# Format:
# y = [Vs, Vd, Ca, m, h, n, mca, mkca, mkahp, mh]

import numpy as np
from parameters_ex_6 import PARAMS
from utils_ex_6 import mh_inf  # Import the specific function needed

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
# Gating variables (manual setup)
# ----------------------------
m0     = 0.0
h0     = 0.5
n0     = 0.4
mca0   = 0.0
mkca0  = 0.2
mkahp0 = 0.2

# ----------------------------
# NEW: Gating variable for Ih current
# Initialized at its steady-state value for the initial dendritic voltage
# ----------------------------
mh0 = mh_inf(Vd0)

# ----------------------------
# State vector (now with 10 elements)
# ----------------------------
Y0 = np.array([
    Vs0, Vd0, Ca0,
    m0, h0, n0,
    mca0, mkca0, mkahp0,
    mh0
], dtype=float)

__all__ = ["Y0"]