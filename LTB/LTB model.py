import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

plt.rc('axes', titlesize=16)  # fontsize of the axes title
plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=13)  # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

"""Constants"""

# Defining constants (from https://arxiv.org/pdf/1308.6731.pdf)
k_max = 2e-7
r_b = 32.0  # Boundary of void [Mpc]

m = 4
n = 4

small = 1e-3  # do not make smaller

H_0 = 71.58781594e-3  # Hubble constant [1/Gyr]
c = 306.5926758  # Speed of light [Mpc/Gyr]
G = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)

# Initial time values
a_i = 1 / 1000  # Initial scale factor (CMB)
t_0 = (2 / 3) * (1 / H_0)  # Present time [Gyr]
t_i = t_0 * a_i ** (3 / 2)  # Initial time [Gyr]

# Time
Nt = 5
t = np.array([0.002, 0.01, 0.05, 0.2, 1])*t_0

"""Functions"""
def M(r):
    return 1 / 2 * H_0 ** 2 * (r ** 3) / (c ** 2)

def M_dr(r):
    return (3 / 2) * ((H_0 ** 2) / (c ** 2)) * r ** 2

def E(r):
    return np.where(r < r_b, -1 * k_max * r ** 2 * ((r / r_b) ** n - 1) ** m, 0)

def E_dr(r):
    return np.where(r < r_b,
                    -2 * r * k_max * ((r / r_b) ** n - 1) ** m - r * k_max * n * m * ((r / r_b) ** n - 1) ** (m - 1) * (
                                r / r_b) ** n, 0)

# Defining a for einstein de sitter
def a_eds(i):
    return (t[i] / t_0) ** (2 / 3)

def f_eta(r, i):
    eta = np.zeros(np.size(r))
    x = np.ones(np.size(r)) * (1 + small)
    fder = lambda x, r: np.cosh(x) - 1
    f = lambda x, r: np.sinh(x) - x - ((-E(r)) ** (3 / 2) / M(r) * c * t[i])
    eta = optimize.newton(f, x, fprime=fder, args=(r,), maxiter=200, tol=1e-12)
    return eta

def R(r, i):
    return np.where(r < r_b*0.99, M(r) / (E(r)) * (1 - np.cosh(f_eta(r, i))), a_eds(i) * r)

def eta_dr(r, i):
    return np.where(r < r_b, 1 / (np.cosh(f_eta(r, i)) - 1) * c * t[i] * (-E(r)) ** (1 / 2) / (M(r) ** 2) * (
                E(r) * M_dr(r) - 3 / 2 * M(r) * E_dr(r)), 0)

def R_dr(r, i):
    return np.where(r < r_b, M_dr(r) / E(r) * (1 - np.cosh(f_eta(r, i))) - M(r) / (E(r) ** 2) * E_dr(r) * (
                1 - np.cosh(f_eta(r, i))) - M(r) / E(r) * np.sinh(f_eta(r, i)) * eta_dr(r, i), a_eds(i))

def rho(r, i):
    return (c ** 4) / (4 * np.pi * G) * M_dr(r) / ((R(r, i)) ** 2 * R_dr(r, i))

def rho_eds(i):
    return 3 * c ** 2 * ((2 / (3 * t[i])) ** 2 / (8 * np.pi * G))

import warnings
warnings.filterwarnings('ignore')

g_size = 64
scale = 1
coords = (np.arange(g_size) - (g_size-1)/2) * scale
X, Y, Z = np.meshgrid(coords, coords, coords)
rad = np.sqrt(X**2 + Y**2 + Z**2)


# Preallocate grid
grid = np.empty_like(rad)

def safe_rho(r, i):
    if r >= r_b*0.99:
        return rho_eds(i)
    else:
        return rho(r, i)


# Element-wise evaluation
timestep = 4
for ix in range(g_size):
    for iy in range(g_size):
        for iz in range(g_size):
            grid[ix, iy, iz] = safe_rho(rad[ix,iy,iz], timestep) / rho_eds(timestep)

#grid_v = grid.reshape(64*64*64)
#print(grid_v)
#grid = grid_v.reshape(64,64,64)
# plotting
plt.figure(figsize=(6,6))
im = plt.imshow(grid[:,:,32], origin='lower',
                extent=[coords.min(), coords.max(), coords.min(), coords.max()])
plt.colorbar(im, label=r'$\rho / \rho_{\mathrm{EdS}}$')
plt.xlabel("x")
plt.ylabel("y")
plt.title("LTB density slice")
plt.show()

#np.savetxt("grid", grid_v)