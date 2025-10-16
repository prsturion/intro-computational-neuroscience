import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 0.7
b = 0.8
phi = 0.08
I = 0

# Nulclines definitions
def w_v(v):
    return v - (v**3)/3

def w_w(v):
    return (v + a)/b

# Vector field (phase plane)
v_vals = np.linspace(-3, 3, 25)
w_vals = np.linspace(-2, 4, 25)
V, W = np.meshgrid(v_vals, w_vals)

dvdt = V - (V**3)/3 - W + I
dwdt = phi * (V + a - b*W)

# Vector field normalization
mag = np.sqrt(dvdt**2 + dwdt**2)
dvdt /= mag
dwdt /= mag

# Plot
plt.figure(figsize=(8,6))
plt.quiver(V, W, dvdt, dwdt, color="gray", alpha=0.5)
plt.plot(v_vals, w_v(v_vals), label="Nulclina de v", color="tab:red", linewidth=2)
plt.plot(v_vals, w_w(v_vals), label="Nulclina de w", color="tab:blue", linewidth=2)
plt.xlabel("v (variável rápida)")
plt.ylabel("w (variável lenta)")
plt.title("Plano de fase do modelo de FitzHugh–Nagumo (I=0)")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_1a.png", dpi=300)
