import numpy as np
import matplotlib.pyplot as plt

# ---------- Model parameters ----------
a = 0.7
b = 0.8
phi = 0.08
I_const = 0.0   # after the impulse we keep I(t)=0

# ---------- Right-hand side (FitzHugh–Nagumo) ----------
def f(v, w, I=0.0):
    # Returns dv/dt and dw/dt for given (v, w)
    dv = v - (v**3)/3.0 - w + I
    dw = phi * (v + a - b*w)
    return dv, dw

# ---------- Time integrators ----------
def step_euler(v, w, dt):
    # One Euler step
    dv, dw = f(v, w, I_const)
    return v + dt*dv, w + dt*dw

def step_rk4(v, w, dt):
    # One classical RK4 step
    k1v, k1w = f(v, w, I_const)
    k2v, k2w = f(v + 0.5*dt*k1v, w + 0.5*dt*k1w, I_const)
    k3v, k3w = f(v + 0.5*dt*k2v, w + 0.5*dt*k2w, I_const)
    k4v, k4w = f(v + dt*k3v,     w + dt*k3w,     I_const)
    v_next = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    w_next = w + (dt/6.0)*(k1w + 2*k2w + 2*k3w + k4w)
    return v_next, w_next

# ---------- Find equilibrium (intersection of nullclines) ----------
# Solve cubic: -(b/3)v^3 + (b-1)v - a = 0 (for I=0)
coeffs = [-b/3.0, 0.0, (b - 1.0), -a]
roots = np.roots(coeffs)
v_star = np.real(roots[np.isclose(np.imag(roots), 0.0)])[0]
w_star = v_star - (v_star**3)/3.0  # v-nullcline

# ---------- Simulation settings ----------
T = 40.0
# Choose method: "rk4" (dt=0.01 as requested "maior") or "euler" (dt=0.001)
method = "rk4"     # change to "euler" if you prefer
dt = 0.01 if method == "rk4" else 0.001
n_steps = int(T/dt) + 1
t = np.linspace(0.0, T, n_steps)

# Initial conditions: impulse modeled as v(0) ≠ v*
v0_list = [-0.8, -0.65, -0.64, -0.6]  # given in the problem
w0 = w_star                            # keep w(0) = w* as instructed

# ---------- Run simulations ----------
sols = []  # list of dicts with 'v', 'w', 'label'
for v0 in v0_list:
    v = np.empty(n_steps)
    w = np.empty(n_steps)
    v[0], w[0] = v0, w0
    # Integrate
    if method == "rk4":
        step = step_rk4
    else:
        step = step_euler
    for k in range(n_steps-1):
        v[k+1], w[k+1] = step(v[k], w[k], dt)
    sols.append({"v": v, "w": w, "label": f"v(0) = {v0:.2f}"})

# ---------- Helper: nullclines ----------
def nullcline_v(v):
    # v-nullcline: dv/dt = 0 -> w = v - v^3/3
    return v - (v**3)/3.0

def nullcline_w(v):
    # w-nullcline: dw/dt = 0 -> w = (v + a)/b
    return (v + a) / b

# ---------- (i) v x t ----------
plt.figure(figsize=(9, 5))
for s in sols:
    plt.plot(t, s["v"], label=s["label"])
plt.axhline(v_star, color="k", ls="--", lw=1, label="v* (equilíbrio)")
plt.xlabel("Tempo (t)")
plt.ylabel("v")
plt.title("FitzHugh–Nagumo (I=0): v × t para diferentes v(0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_1h_1.png", dpi=300, bbox_inches="tight")  # save instead of show

# ---------- (ii) w x t ----------
plt.figure(figsize=(9, 5))
for s in sols:
    plt.plot(t, s["w"], label=s["label"])
plt.axhline(w_star, color="k", ls="--", lw=1, label="w* (equilíbrio)")
plt.xlabel("Tempo (t)")
plt.ylabel("w")
plt.title("FitzHugh–Nagumo (I=0): w × t para diferentes v(0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_1h_2.png", dpi=300, bbox_inches="tight")  # save instead of show

# ---------- (iii) Phase plane w × v with nullclines ----------
markers = ["o", "s", "^", "d"]  # different markers to visually distinguish
v_grid = np.linspace(-2.0, 2.0, 600)  # for nullclines
plt.figure(figsize=(7.5, 7))
# Nullclines
plt.plot(v_grid, nullcline_v(v_grid), "k-", lw=2, label="Nulclina de v")
plt.plot(v_grid[:-160], nullcline_w(v_grid)[:-160], "k--", lw=2, label="Nulclina de w")
# Trajectories
for s, m in zip(sols, markers):
    plt.plot(s["v"], s["w"], lw=1.8, label=f"trajetória ({s['label']})")
    plt.scatter(s["v"][0], s["w"][0], marker=m, s=50)  # initial point # type: ignore
# Equilibrium
plt.scatter([v_star], [w_star], c="k", s=60, zorder=5, label="Equilíbrio")
plt.xlabel("v (rápida)")
plt.ylabel("w (lenta)")
plt.title("Plano de fase w × v com nulclinas e trajetórias")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("intro-computational-neuroscience/list_5/figures/ex_1h_3.png", dpi=300, bbox_inches="tight")  # save instead of show

# plt.show()  # commented as requested
