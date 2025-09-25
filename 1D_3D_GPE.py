import numpy as np
import matplotlib.pyplot as plt

# --- small helper to avoid deprecation noise between numpy versions ---
def trapint(y, dx):
    try:
        return np.trapezoid(y, dx=dx)   # new name
    except Exception:
        return np.trapz(y, dx=dx)       # fallback

# ---------- CN step for diffusion with D = 1/2 (OPTION 2) ----------
def cn_diffusion_step(phi, dx, dt):
    """
    Solve: (I + alpha*L) phi^{n+1} = (I - alpha*L) phi^n
    with L being the standard second-difference operator and
    alpha = dt * D / 2, with D = 1/2 here.
    Dirichlet BC at both ends (set by caller).
    """
    N = phi.size
    alpha = dt/(4.0*dx*dx)  # dt * (1/2) / 2
    a = -alpha*np.ones(N-1)
    b = (1.0 + 2.0*alpha)*np.ones(N)
    c = -alpha*np.ones(N-1)

    # RHS vector
    r = (1.0 - 2.0*alpha)*phi.copy()
    r[1:-1] += alpha*(phi[2:] + phi[:-2])

    # Thomas algorithm
    cp = np.zeros(N-1); dp = np.zeros(N)
    cp[0] = c[0]/b[0]; dp[0] = r[0]/b[0]
    for i in range(1, N-1):
        denom = b[i] - a[i-1]*cp[i-1]
        cp[i] = c[i]/denom
        dp[i] = (r[i] - a[i-1]*dp[i-1])/denom
    dp[N-1] = (r[N-1] - a[N-2]*dp[N-2])/(b[N-1] - a[N-2]*cp[N-2])

    out = np.zeros_like(phi)
    out[-1] = dp[-1]
    for i in range(N-2, -1, -1):
        out[i] = dp[i] - (cp[i-1]*out[i+1] if i>0 else 0.0)

    # Dirichlet at boundaries
    out[0] = 0.0
    out[-1] = 0.0
    return out

def norm1d(phi, dx):
    return phi / np.sqrt(trapint(phi**2, dx))

def norm3d_phi(phi_r, dr):
    # normalization for ϕ(r) (the evolved variable): 4π∫|ϕ|² dr = 1
    return phi_r / np.sqrt(4*np.pi*trapint(phi_r**2, dr))

# ---------- ground states (OPTION 2) ----------
def gs_1d(G, L=13.0, dx=0.01, dt=2e-4, steps=15000):
    # grid & initial (harmonic osc. ground state, G=0)
    x = np.arange(-L, L+dx, dx)
    phi = (np.pi**-0.25)*np.exp(-0.5*x**2)
    phi = norm1d(phi, dx)

    for _ in range(steps):
        # H1: trap + nonlinearity
        phi *= np.exp(-dt*(0.5*x**2 + G*(phi**2)))
        phi = norm1d(phi, dx)
        # H2: kinetic with D=1/2
        phi = cn_diffusion_step(phi, dx, dt)
        phi = norm1d(phi, dx)

    return x, phi

def gs_radial3d(G, R=6.0, dr=0.004, dt=2e-4, steps=15000):
    # evolve ϕ(r)=r*ψ; later plot φ=ϕ/r
    r = np.arange(0.0, R+dr, dr)
    phi = r*(np.pi**-0.75)*np.exp(-0.5*r**2)  # ϕ init from HO ground state
    phi = norm3d_phi(phi, dr)
    eps = (dr/2)**2  # to avoid divide-by-zero in the nonlinear term

    for _ in range(steps):
        nl = G*(phi**2)/(r**2 + eps)          # |ϕ|² / r²
        phi *= np.exp(-dt*(0.5*r**2 + nl))    # H1 step
        phi = norm3d_phi(phi, dr)
        phi = cn_diffusion_step(phi, dr, dt)  # H2 step
        phi[0] = 0.0                           # ϕ(0)=0
        phi = norm3d_phi(phi, dr)

    # Convert to physical φ(r)=ϕ(r)/r for plotting
    r_safe = r.copy()
    r_safe[0] = dr
    phi_phys = phi / r_safe
    phi_phys[0] = phi[1]/dr                   # φ(0) = ϕ'(0) ≈ ϕ(dr)/dr

    # (optional) enforce physical norm for φ just to remove tiny r=0 error
    norm = np.sqrt(4*np.pi*trapint((phi_phys**2)*(r**2), dr))
    phi_phys /= norm

    return r, phi_phys

# ---------- lists of G from Fig. 2 ----------
G3D = [-3.1371, 0.0, 3.1371, 12.5484, 31.371, 125.484, 627.4, 3137.1]
G1D = [-2.5097, 0.0, 3.1371, 12.5484, 31.371, 62.742, 156.855, 313.71, 627.42, 1254.8]

# nice palette to mimic the paper
COLORS = ["#d62728","#2ca02c","#1f77b4","#e377c2","#17becf","#7f7f7f","#ff7f0e","#9467bd","#8c564b","#000000"]
STYLES = ["-","--","-.",":","-","--","-.",":","--","-"]

# ---------- compute & plot ----------
# Radial 3D
plt.figure(figsize=(9,4))
ax = plt.gca()
for i,G in enumerate(G3D):
    r,phi = gs_radial3d(G, R=6.0, dr=0.004, dt=2e-4, steps=15000)
    plt.plot(r, phi, lw=2.5, color=COLORS[i%len(COLORS)], ls=STYLES[i%len(STYLES)], label=f"{G:g}")
plt.xlim(0,6); plt.ylim(0,0.5)
ax.tick_params(colors="blue")
plt.xlabel("r", color="blue", fontsize=14)
plt.ylabel(r"$\phi(r)$", color="blue", fontsize=16)
plt.legend(loc="upper right", fontsize=11, frameon=False)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("figure2_left_radial3d.png", dpi=220)
plt.show()

# 1D
plt.figure(figsize=(9,4))
ax = plt.gca()
for i,G in enumerate(G1D):
    x,phi = gs_1d(G, L=13.0, dx=0.01, dt=2e-4, steps=15000)
    plt.plot(x, phi, lw=2.5, color=COLORS[i%len(COLORS)], ls=STYLES[i%len(STYLES)], label=f"{G:g}")
plt.xlim(-13,13); plt.ylim(0,1.0)
ax.tick_params(colors="blue")
plt.xlabel("x", color="blue", fontsize=14)
plt.ylabel(r"$\phi(x)$", color="blue", fontsize=16)
plt.legend(loc="upper right", fontsize=11, frameon=False)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("figure2_right_1d.png", dpi=220)
plt.show()

print("Saved: figure2_left_radial3d.png, figure2_right_1d.png")
