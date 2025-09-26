import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
from numba import njit, prange

"""Constants"""
# Defining constants
k_max = 1e-13#2e-13#2e-7 # why 1e-6 times original??
g_size = 32#64
r_b = g_size // 2 # 24
m = 4
n = 4
small = 1e-3  # do not make smaller

mu=1.989e45
lu=3.085678e19
tu=31557600e6

G = 6.6742e-11*((mu*(tu**2))/(lu**3))
c = 299792458*(tu/lu)

"""Functions"""
def M(r):
    return 1 / 2 * H0 ** 2 * (r ** 3) / (c ** 2)

def M_dr(r):
    return (3 / 2) * ((H0 ** 2) / (c ** 2)) * r ** 2

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

def R_dt(r, i):
  return np.sqrt(2*M(r)/R(r, i)-E(r))*c

def R_dr_dt(r,i):
  return c**2/(2*R_dt(r, i))*(2*M_dr(r)/R(r, i)-2*M(r)*R_dr(r, i)/((R(r, i))**2)-E_dr(r))

def a_eds_dt(i):
  return 2/3*(t[i])**(-1/3)*(t_0)**(-2/3)

def theta_eds(i):
  return 3*a_eds_dt(i)/a_eds(i)

def theta(r, i):
  return 2*R_dt(r, i)/R(r, i)+R_dr_dt(r, i)/R_dr(r, i)

def sigma(r, i):
  return 1/np.sqrt(3)*(-R_dt(r, i)/R(r, i)+R_dr_dt(r, i)/R_dr(r, i))#**2 #this is sigma**2 with units 1/s**2

def weyl(r, i):
    return M(r)/(3*R(r, i)**3)*(3*R_dr(r,i)-R(r,i)*M_dr(r)/M(r))/R_dr(r,i)

def safe_rho(r, i):
    if r >= r_b*0.995:
        return rho_eds(i)
    else:
        return rho(r, i)

def safe_theta(r, i):
    if r >= r_b*0.995:
        return theta_eds(i)
    else:
        return theta(r, i)

def safe_sigma(r, i):
    if r >= r_b*0.995:
        return 0
    else:
        return sigma(r, i)

def safe_weyl(r, i):
    if r >= r_b*0.995:
        return 0
    else:
        return weyl(r, i)


# Using simsilun values/units
def sim_rho(r,i):
    return 2*M_dr(r)/(R(r, i) ** 2 * R_dr(r, i))


def sim_theta(r,i):
    return 2*R_dt(r, i)/R(r, i)+R_dr_dt(r, i)/R_dr(r, i)

def sim_sigma(r,i):
    return -1/3*(R_dr_dt(r, i)/R_dr(r, i)-R_dt(r, i)/R(r, i))

def sim_weyl(r,i):
    return M(r)/(3*R(r, i)**3)*(3*R_dr(r,i)-R(r,i)*M_dr(r)/M(r))/R_dr(r,i)

def rho_eds(i):
    return 1/c**2 * 3 * (2 / (3 * t[i])) ** 2

def safe_rho(r, i):
    if r >= r_b*0.98:
        return rho_eds(i)
    else:
        return sim_rho(r, i)
def safe_theta(r, i):
    if r >= r_b*0.98:
        return theta_eds(i)
    else:
        return sim_theta(r, i)
def safe_sigma(r, i):
    if r >= r_b*0.98:
        return 0
    else:
        return sim_sigma(r, i)
def safe_weyl(r, i):
    if r >= r_b*0.98:
        return 0
    else:
        return sim_weyl(r, i)

def safe_V(r, i):
    if r >= r_b*0.98:
        return a_eds(i)**3#deter_eds(r, i)/r**2
    else:
        return R_dr(r,i)*R(r,i)**2/np.sqrt(1-E(r))/r**2

# Element-wise evaluation
def evolve_LTB(timestep, z_i, z_f, H_0):
    global t, t_0, H0
    H0 = (tu / lu) * H_0
    t_0 = (2 / 3) * (1 / H0)  # Present time
    a_i = 1 / (1 + z_i)  # Initial scale factor
    t_i = t_0 * a_i ** (3 / 2)  # Initial time
    a_f = 1 / (1 + z_f)  # Final scale factor
    t_f = t_0 * a_f ** (3 / 2)  # Final time
    t = np.array([t_i, t_f])

    coords = (np.arange(g_size) - (g_size-1)/2)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    rad = np.sqrt(X**2 + Y**2 + Z**2)
    # Preallocate grid
    grid_rho = np.empty_like(rad)
    grid_theta = np.empty_like(rad)
    grid_sigma = np.empty_like(rad)
    grid_weyl = np.empty_like(rad)
    grid_V = np.empty_like(rad)
    for ix in tqdm(range(g_size)):
        for iy in range(g_size):
            for iz in range(g_size):
                grid_rho[ix, iy, iz] = safe_rho(rad[ix, iy, iz], timestep) / rho_eds(timestep)
                grid_theta[ix, iy, iz] = safe_theta(rad[ix, iy, iz], timestep) / theta_eds(timestep)
                grid_sigma[ix, iy, iz] = 3*safe_sigma(rad[ix, iy, iz], timestep) / theta_eds(timestep)
                grid_weyl[ix, iy, iz] = safe_weyl(rad[ix, iy, iz], timestep)# / rho_eds(timestep)
                grid_V[ix, iy, iz] = safe_V(rad[ix, iy, iz], timestep)
    np.save(f"../data/np_arrays_LTB/grid_rho{timestep}.npy", grid_rho, allow_pickle=True)
    np.save(f"../data/np_arrays_LTB/grid_theta{timestep}.npy", grid_theta, allow_pickle=True)
    np.save(f"../data/np_arrays_LTB/grid_sigma{timestep}.npy", grid_sigma, allow_pickle=True)
    np.save(f"../data/np_arrays_LTB/grid_weyl{timestep}.npy", grid_weyl, allow_pickle=True)
    np.save(f"../data/np_arrays_LTB/grid_V{timestep}.npy", grid_V, allow_pickle=True)

    return grid_rho, grid_theta, grid_sigma, grid_weyl, grid_V