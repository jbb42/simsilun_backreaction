import numpy as np
from classy import Class
from matplotlib import pyplot as plt

# --- CLASS parameters ---
params = {
    'output': 'mPk,dTk',  # matter power spectrum + density transfer functions
    'P_k_max_h/Mpc': 15.0,
    'z_pk': 1086,          # redshift for ICs
    'A_s': 2.3e-9,
    'n_s': 0.9624,
    'h': 0.6711,
    'omega_b': 0.022068,
    'omega_cdm': 0.12029,
}

# --- Create CLASS instance ---
cosmo = Class()
cosmo.set(params)
cosmo.compute()

# --- k grid (h/Mpc) ---
ks = np.logspace(-4, 1, 1000)  # 1000 points from large to small scales

# --- Power spectrum ---
z = 1086
pk = np.array([cosmo.pk(k, z) for k in ks])  # (Mpc)^3
ks_h = ks * params['h']        # convert to h/Mpc
pk_h = pk / params['h']**3     # convert to (Mpc/h)^3

# Save P(k)
np.savetxt("Pk_z1086.dat", np.column_stack([ks_h, pk_h]))
print("Power spectrum saved to Pk_z1086.dat")

# --- Transfer functions (keep your working structure) ---
T_cdm = np.array([cosmo.get_transfer(k)['d_cdm'] for k in ks])
T_b   = np.array([cosmo.get_transfer(k)['d_b']   for k in ks])

# Save transfer functions
np.savetxt("transfer_z1086.dat", np.column_stack([ks_h, T_cdm, T_b]))
print("Transfer functions saved to transfer_z1086.dat")

# --- Optional: plot ---
plt.figure()
plt.loglog(ks_h, pk_h, label='P(k)')
plt.xlabel('k [h/Mpc]')
plt.ylabel('P(k) [(Mpc/h)^3]')
plt.legend()
plt.show()

# --- Cleanup ---
cosmo.struct_cleanup()
cosmo.empty()
