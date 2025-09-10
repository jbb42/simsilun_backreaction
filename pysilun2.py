import numpy as np
from matplotlib import pyplot as plt

# ------------------------
# Silent Universe derivatives
# ------------------------
def dx_dt(X, kap, lb):
    """
    X: (4, Ni)
    Returns dX/dt with same shape
    """
    rho, theta, sigma, weyl = X
    dx = np.zeros_like(X)
    dx[0] = -rho * theta
    dx[1] = -(theta**2)/3 - 0.5*kap*rho - 6*sigma**2 + lb
    dx[2] = -(2/3)*theta*sigma - weyl + sigma**2
    dx[3] = -3*weyl*sigma - theta*weyl - 0.5*kap*rho*sigma
    return dx

# ------------------------
# Vectorised RK4 with collapse mask
# ------------------------
def rk4_vectorised(X0, t0, t1, nsteps, kap, lb):

    X = X0.copy()
    active = np.ones(X.shape[1], dtype=bool)  # True = still evolving

    theta_active = X[1, active]
    sigma_active = X[2, active]
    max_dt = (t1 - t0) / nsteps

    dt = np.minimum(max_dt, 1e-3 / (theta_active + 0.33 * sigma_active))
    for _ in range(nsteps):
        if not np.any(active):
            break  # all cells collapsed

        # RK4 only for active cells
        k1 = dx_dt(X[:, active], kap, lb)
        k2 = dx_dt(X[:, active] + dt/2*k1, kap, lb)
        k3 = dx_dt(X[:, active] + dt/2*k2, kap, lb)
        k4 = dx_dt(X[:, active] + dt*k3, kap, lb)
        X[:, active] += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # update active mask
        active[active] = X[1, active] > 0
        X[1, ~active] = 0  # stop theta at 0

    return X

# ------------------------
# Main program
# ------------------------
delta0 = np.loadtxt("simsilun/grid") - 1
Ni = delta0.size

# constants
z_i, z_f, H_0 = 80, 78, 70
mu, lu, tu = 1.989e45, 3.085678e19, 31557600*1e6
gcons = 6.6742e-11*(mu*tu**2/lu**3)
cs = 299792458*(tu/lu)
kap = 8*np.pi*gcons*(1/cs**4)
kapc2 = 8*np.pi*gcons*(1/(cs**2))
Ho = tu/lu * H_0
lb = 0.0

t_0 = 2 / (3*Ho)
t_i = t_0*(1+z_i)**(-1.5)
t_f = t_0*(1+z_f)**(-1.5)

rho_bg_i = 3*(2/(3*t_i))**2/kapc2*cs**2
theta_bg = 3*Ho*(1+z_i)**1.5
rho_bg_f = 3*(2/(3*t_f))**2/kapc2*cs**2

rho0   = rho_bg_i * (1 + delta0)
theta0 = theta_bg * (1 - 1/3*delta0)
sigma0 = (1/9)*theta_bg*delta0
weyl0  = -(1/6)*rho_bg_i*delta0

x0 = np.stack([rho0, theta0, sigma0, weyl0], axis=0)

# ------------------------
# evolve all cells
# ------------------------
nsteps = 2000
X_final = rk4_vectorised(x0, t_i, t_f, nsteps, kap, lb)

rho_i = rho0.reshape(64,64)
rho_f = X_final[0].reshape(64,64)

# ------------------------
# Plot
# ------------------------
fig, axes = plt.subplots(1,2,figsize=(8,4))
coords = np.arange(64) - 31.5

def plot_universe(ax, grid, coords, title):
    im = ax.imshow(grid, origin='lower', extent=[coords.min(),coords.max(),coords.min(),coords.max()])
    fig.colorbar(im, ax=ax, label=r'$\rho / \rho_{\mathrm{EdS}}$')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

plot_universe(axes[0], rho_i/rho_bg_i, coords, "Initial density")
plot_universe(axes[1], rho_f/rho_bg_f, coords, "Final density")
plt.tight_layout()
plt.show()
