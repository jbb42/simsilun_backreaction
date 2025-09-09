import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
#from plot_all import plot_universe

def simsilun_odes(_, x_flat, kap, lb):
  """
  The simsilun equations with x being a vector of initial conditions
  of the form (5, Ni).
  """
  x = x_flat.reshape(4, Ni)
  dxdt = np.zeros_like(x)
  dxdt[0] = -x[0] * x[1]  # ρ̇
  dxdt[1] = -(x[1] ** 2) / 3 - 0.5 * kap * x[0] - 6 * x[2] ** 2 + lb  # θ̇
  dxdt[2] = -(2 / 3) * x[1] * x[2] - x[3] + x[2] ** 2  # σ̇
  dxdt[3] = -3 * x[3] * x[2] - x[1] * x[3] - 0.5 * kap * x[0] * x[2]  # Weyl̇

  return dxdt.flatten()

def plot_universe(axes, grid, coords, posx, title):
    ax = axes[posx]  # upper-left subplot

    im = ax.imshow(grid[:, :], origin='lower',
                   extent=[coords.min(), coords.max(), coords.min(), coords.max()])
    fig.colorbar(im, ax=ax, label=r'$\rho / \rho_{\mathrm{EdS}}$')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()

pi = np.pi
z_i = 80    # Initial redshift
z_f = 79     # Final redshift
H_0 = 70    # Hubble constant [km/s/Mpc]
omega_lambda = 0
omega_matter = 1

mu=1.989e45
lu=3.085678e19
tu=31557600*1e6

gcons   = 6.6742e-11*(mu*tu**2/lu**3)
cs      = 299792458*(tu/lu)
kap     = 8*pi*gcons*(1/cs**4)
kapc2   = 8*pi*gcons*(1/(cs**2))
Ho      = tu/lu * H_0
gkr     = 3*(Ho ** 2/ (8 * pi * gcons))
lb      = 3*omega_lambda*(Ho**2/cs**2)
gkr     = kapc2*gkr*omega_matter


#np.random.seed(42)
#delta0 = np.random.randn(64*64)*0.01
delta0 = np.loadtxt("simsilun/grid") - 1
Ni = 64*64

x0 = np.zeros((5,Ni))
x = np.zeros_like(x0)

t_0 = 2 / (3 * Ho) # Present time
t_i = t_0 * (1 + z_i) ** (-3 / 2)  # Initial time
t_f = t_0 * (1 + z_f) ** (-3 / 2)  # Final time

#rho_bg = gkr * (z_i+1)**3
rho_bg_i = 3 * (2 / (3 * t_i)) ** 2 / kapc2 * cs**2 # mass density
theta_bg = 3.0 * Ho * (z_i+1)** 1.5 #3H
#rho_bg_f = gkr * (z_f+1)**3
rho_bg_f = 3 * (2 / (3 * t_f)) ** 2 / kapc2 * cs**2 # mass density
print("rho_bg =", rho_bg_i)
print("theta_bg =", theta_bg)
print("rho_bg_f = ", rho_bg_f)

rho0 = rho_bg_i * (1 + delta0)
theta0 = theta_bg * (1 - 1/3 * delta0)
sigma0 = 1/9 * theta_bg * delta0
weyl0 = -1/6 * rho_bg_i * delta0

x0 = np.array([rho0, theta0, sigma0, weyl0])

t_span = (t_i, t_f)
sol = solve_ivp(simsilun_odes, t_span, x0.flatten(), args=(kap, lb),
                method='RK45', rtol=1e-6, atol=1e-9)

n_t = sol.y.shape[1]
X_sol = sol.y.reshape(4, Ni, n_t)
rho_i = X_sol[0, :, 0].reshape(64, 64)
rho_f = X_sol[0, :, -1].reshape(64, 64)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
g_size = np.sqrt(Ni)
coords = (np.arange(g_size) - (g_size-1)/2)


plot_universe(axes, rho_f/rho_bg_f, coords, 1, "Final density")
plot_universe(axes, rho_i/rho_bg_i, coords, 0, "Initial density")
plt.show()

#print(rho_f[32,32]/rho_bg_f)
