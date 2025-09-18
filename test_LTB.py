import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
from numba import njit, prange

"""Constants"""
# Defining constants
k_max = 1e-13#2e-13#2e-7 # why 1e-6 times original??
g_size = 64
r_b = g_size/2
m = 4
n = 4
small = 1e-3  # do not make smaller

mu=1.989e45
lu=3.085678e19
tu=31557600e6

G = 6.6742e-11*((mu*(tu**2))/(lu**3))
c = 299792458*(tu/lu)
z_i, z_f, H_0 = 1100, 0, 70


H0 = (tu / lu) * H_0
t_0 = (2 / 3) * (1 / H0)  # Present time
a_i = 1 / (1 + z_i)  # Initial scale factor
t_i = t_0 * a_i ** (3 / 2)  # Initial time
a_f = 1 / (1 + z_f)  # Final scale factor
t_f = t_0 * a_f ** (3 / 2)  # Final time
t = np.array([t_i, t_f])

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

def R_dt(r, i):
  return np.sqrt(2*M(r)/R(r, i)-E(r))*c

def R_dr_dt(r,i):
  return c**2/(2*R_dt(r, i))*(2*M_dr(r)/R(r, i)-2*M(r)*R_dr(r, i)/((R(r, i))**2)-E_dr(r))

def a_eds_dt(i):
  return 2/3*(t[i])**(-1/3)*(t_0)**(-2/3)

def theta_eds(i):
  return 3*a_eds_dt(i)/a_eds(i)

def rho(r,i):
    return 2*M_dr(r)/(R(r, i) ** 2 * R_dr(r, i))

def theta(r,i):
    return 2*R_dt(r, i)/R(r, i)+R_dr_dt(r, i)/R_dr(r, i)

def sigma(r,i):
    return -1/3*(R_dr_dt(r, i)/R_dr(r, i)-R_dt(r, i)/R(r, i))

def weyl(r,i):
    return M(r)/(3*R(r, i)**3)*(3*R_dr(r,i)-R(r,i)*M_dr(r)/M(r))/R_dr(r,i)

def rho_eds(i):
    return 1/c**2 * 3 * (2 / (3 * t[i])) ** 2

def deter(r, i):
    return (R_dr(r, i)) * R(r, i) ** 2 / np.sqrt(1 - E(r))


def deter_eds(r, i):
    return a_eds(i) ** 3 * r ** 2


def V_of_r_eds(r, i):
    return a_eds(i) ** 3 * r ** 3 * 1 / 3


# Volume LTB
def V_LTB(r, i):
    return 4 * np.pi * np.cumsum(deter(r, i) * dr)


# Volume Eds
def V_Eds(r, i):
    return 4 * np.pi / 3 * (a_eds(i) ** 3) * (r ** 3)


def integrating_Q(i):
    # Vectorized implementation
    theta_vals = theta(r, i)
    deter_vals = deter(r, i)
    sigma_vals = sigma(r, i)
    V_vals = V_LTB(r, i)

    # Compute integrals using cumulative sum
    theta_int = 4 * np.pi * np.cumsum(dr * theta_vals * deter_vals) / V_vals

    theta2_int = 4 * np.pi * np.cumsum(dr * theta_vals ** 2 * deter_vals) / V_vals

    sigma2_int = 4 * np.pi * np.cumsum(dr * sigma_vals ** 2 * deter_vals) / V_vals
    return theta_int, theta2_int, sigma2_int, V_vals


def Q(r, i):
    theta_int, theta2_int, sigma2_int, V_vals = integrating_Q(i)
    return 2 / 3 * (theta2_int - theta_int ** 2) - 2 * sigma2_int


def H_avg(i):
    theta_averaged = np.zeros(Nt)
    theta_int, theta2_int, sigma2_int, V_vals = integrating_Q(i)
    theta_averaged = theta_int[-1]
    return theta_averaged / 3


import warnings

warnings.filterwarnings('ignore')
r = np.linspace(small, r_b, g_size)
i = 1
dr = r[1]-r[0]
plt.plot(Q(r,i))
plt.show()