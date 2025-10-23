import numpy as np

mu = 1.989e45  # 10^15 solar masses
lu = 3.085678e19  # 10 kpc
tu = 31557600.0 * 1e6  # 1 mega years

G = 6.6742e-11 * mu * tu ** 2 / lu ** 3
c = 299792458.0 * tu / lu
kappa = 8 * np.pi * G / c**4

def H0(H):
    return tu / lu * H

def Lambda(OmegaL, Ho):
    return 3 * OmegaL * H0(Ho)**2 / c**2

# Volume-weighted means
def gm(X, V):
    return np.sum(X*V)/np.sum(V)

def get_omegas(data):
    theta = data["theta"]
    rho = data["rho"]
    sigma = data["sigma"]
    V = data["V"]
    H = gm(theta, V)/3
    Omega_m = gm(rho, V)/(3*H**2)
    Omega_Q = (gm(sigma**2, V)+1/9*gm(theta**2, V)-H**2)/H**2
    R = 2*gm(rho, V) + 6*gm(sigma**2, V) - 2/3*gm(theta**2, V) + 2*Lambda(data["Omega_Lambda"], data["H_0"])
    Omega_K = -R/(6*H**2)
    Omega_L = Lambda(data["Omega_Lambda"], data["H_0"])/(3*H**2)
    return Omega_m, Omega_Q, Omega_K, Omega_L, H

for i in range(44):
    print("\nFile number:", i)
    initial = np.load(f"./data/jusilun_output/initial_vals_{str(i).zfill(3)}.npz")
    print("Omega_m =", initial["Omega_m"],
          "\tOmega_Lambda =", initial["Omega_Lambda"],
          "\tOmega_k =", 1-initial["Omega_m"]-initial["Omega_Lambda"],
          "\tH_0 =", initial["H_0"])

    Omega_m, Omega_Q, Omega_K, Omega_L, H = get_omegas(initial)
    print(f"Density parameters at H = {H * lu / tu * c:.5f} \t z = {initial['z_i']}")
    print(f"Omega_m_i = {Omega_m:.5f}")
    print(f"Omega_L_i = {Omega_L:.5f}")
    print(f"Omega_Q_i = {Omega_Q:.5f}")
    print(f"Omega_K_i = {Omega_K:.5f}")
    print(f"Omega_T_i = {Omega_m + Omega_L + Omega_Q + Omega_K:.5f}\n")

    final = np.load(f"./data/jusilun_output/final_vals_{str(i).zfill(3)}.npz")
    Omega_m, Omega_Q, Omega_K, Omega_L, H = get_omegas(final)
    print(f"Density parameters at H = {H * lu / tu * c:.5f} \t z = 0")
    print(f"Omega_m_f = {Omega_m:.5f}")
    print(f"Omega_L_f = {Omega_L:.5f}")
    print(f"Omega_Q_f = {Omega_Q:.5f}")
    print(f"Omega_K_f = {Omega_K:.5f}")
    print(f"Omega_T_f = {Omega_m + Omega_L + Omega_Q + Omega_K:.5f}\n")





