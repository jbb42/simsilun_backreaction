import numpy as np
initial = np.load("./data/jusilun_output/initial_vals_004.npz")
final = np.load("./data/jusilun_output/final_vals_004.npz")

def gr_mean(arr, vol):
    return np.sum(arr*vol)/np.sum(vol)

theta_i_avg = gr_mean(initial["theta"], initial["V"])
theta_f_avg = gr_mean(final["theta"], final["V"])
theta_i_avg2= gr_mean(initial["theta"]**2, initial["V"])
theta_f_avg2= gr_mean(final["theta"]**2, final["V"])

sigma_i_avg = gr_mean(initial["sigma"], initial["V"])
sigma_f_avg = gr_mean(final["sigma"], final["V"])
sigma_i_avg2= gr_mean(initial["sigma"]**2, initial["V"])
sigma_f_avg2= gr_mean(final["sigma"]**2, final["V"])

Q_i = 2 / 3 * (theta_i_avg2-theta_i_avg**2) - 2 * sigma_i_avg2
Q_f = 2 / 3 * (theta_f_avg2-theta_f_avg**2) - 2 * sigma_f_avg2

print("Q_i = ", Q_i)
print("Q_f = ", Q_f)

kappa = final["kappa"]
Lambda = final["Lambda"]

rho_i_avg = gr_mean(initial["rho"], initial["V"])
rho_f_avg = gr_mean(final["rho"], final["V"])
R_i = 2*rho_i_avg + 6*sigma_i_avg2 - 2/3*theta_i_avg2 + 2*Lambda
R_f = 2*rho_f_avg + 6*sigma_f_avg2 - 2/3*theta_f_avg2 + 2*Lambda

print("R_i = ", R_i)
print("R_f = ", R_f)

print("Omega_Q_i =", -Q_i/(2/3*theta_i_avg**2))
print("Omega_Q_f =", -Q_f/(2/3*theta_f_avg**2))

print("Omega_R_i =", -R_i/(2/3*theta_i_avg**2))
print("Omega_R_f =", -R_f/(2/3*theta_f_avg**2))
