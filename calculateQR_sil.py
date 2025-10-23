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


initial = np.load(f"./data/jusilun_output/initial_vals_043.npz")
final   = np.load(f"./data/jusilun_output/final_vals_043.npz")
print("JUSILUN","\tOmega_m =", initial["Omega_m"],
      "\tOmega_Lambda =", initial["Omega_Lambda"],
      "\tOmega_k =", 1-initial["Omega_m"]-initial["Omega_Lambda"],
      "\tH_0 =", initial["H_0"])

print("Q_i = ", initial["Q"])
print("Q_f = ", final["Q"])
print("R_i = ", initial["R"])
print("R_f = ", final["R"])

print("Omega_Q_i =", -initial["Q"]/(6*initial["H"]**2))
print("Omega_Q_f =", -final["Q"]/(6*final["H"]**2))
print("Omega_k_i =", -initial["R"]/(6*initial["H"]**2))
print("Omega_k_f =", -final["R"]/(6*final["H"]**2))
print("Omega_m_i =", np.sum(initial["rho"]*initial["V"])/np.sum(initial["V"])/(3*initial["H"]**2))
print("Omega_m_f =", np.sum(final["rho"]*final["V"])/np.sum(final["V"])/(3*final["H"]**2))
print("Omega_t_i =", -initial["Q"]/(6*initial["H"]**2) -
      initial["R"]/(6*initial["H"]**2) +
      Lambda(initial["Omega_Lambda"], initial["H_0"])/(3*initial["H"]**2) +
      np.sum(initial["rho"]*initial["V"])/np.sum(initial["V"])/(3*initial["H"]**2))
print("Omega_t_f =", -final["Q"]/(6*final["H"]**2) -
      final["R"]/(6*final["H"]**2) +
      Lambda(initial["Omega_Lambda"], final["H_0"])/(3*final["H"]**2) +
      np.sum(final["rho"]*final["V"])/np.sum(final["V"])/(3*final["H"]**2))
print("H", final["H"])



initial = np.load(f"./data/jusilun_output/initial_vals_043.npz")
final   = np.load(f"./data/simsilun_output/params_new.npz")
print("SIMSILUN","\tOmega_m =", initial["Omega_m"],
      "\tOmega_Lambda =", initial["Omega_Lambda"],
      "\tOmega_k =", 1-initial["Omega_m"]-initial["Omega_Lambda"],
      "\tH_0 =", initial["H_0"])

print("Q_i = ", initial["Q"])
print("Q_f = ", final["Q"])
print("R_i = ", initial["R"])
print("R_f = ", final["R"])

print("Omega_Q_i =", -initial["Q"]/(6*initial["H"]**2))
print("Omega_Q_f =", -final["Q"]/(6*final["H"]**2))
print("Omega_k_i =", -initial["R"]/(6*initial["H"]**2))
print("Omega_k_f =", -final["R"]/(6*final["H"]**2))
print("Omega_m_i =", np.sum(initial["rho"]*initial["V"])/np.sum(initial["V"])/(3*initial["H"]**2))
print("Omega_m_f =", np.sum(final["rho"]*final["V"])/np.sum(final["V"])/(3*final["H"]**2))
print("Omega_t_i =", -initial["Q"]/(6*initial["H"]**2) -
      initial["R"]/(6*initial["H"]**2) +
      Lambda(initial["Omega_Lambda"], initial["H_0"])/(3*initial["H"]**2) +
      np.sum(initial["rho"]*initial["V"])/np.sum(initial["V"])/(3*initial["H"]**2))
print("Omega_t_f =", -final["Q"]/(6*final["H"]**2) -
      final["R"]/(6*final["H"]**2) +
      Lambda(initial["Omega_Lambda"], 0.7)/(3*final["H"]**2) +
      np.sum(final["rho"]*final["V"])/np.sum(final["V"])/(3*final["H"]**2))
print("H", final["H"])



print("/n/n/n For Chatty:")
print("H final =", final["H"])
print("R final =", final["R"])
print("R/H^2 =", final["R"]/final["H"]**2)
