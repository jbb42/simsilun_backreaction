from classy import Class
import subprocess
import numpy as np

z = 90.0

# create instance of the class "Class"
cosmo = Class()
# pass input parameters
cosmo.set({'h': 0.70,
               'Omega_m': 0.3,
               'Omega_Lambda': 0.7,
               'Omega_k': 0.0,
               'A_s': 2.1e-9,
               'n_s': 0.965,
               'tau_reio' :0.054,

               'output': 'mPk,dTk',
               'z_pk': z,
               'format':'camb',
               'headers': 'yes',
               'P_k_max_h/Mpc': 100,  # extend upper range if needed
           })
# run class
cosmo.compute()
# --- extract transfer functions ---
transfers = cosmo.get_transfer(z)

# k in h/Mpc
k_hMpc = np.array(transfers["k (h/Mpc)"])
h = cosmo.h()  # reduced Hubble
k_Mpc = k_hMpc * h  # convert to 1/Mpc

# species you care about
species = ["d_cdm", "d_b", "d_g", "d_ur", "d_tot"]

# build rescaled dictionary
Tk_rescaled = {}
for sp in species:
    Ti = np.array(transfers[sp])
    Tk_rescaled[sp] = -Ti / (k_Mpc**2)

# --- save in CAMB-like format ---
header = (
    f"k [h/Mpc]  " + "  ".join([f"-T_{sp}/k^2" for sp in species])
)
data = np.column_stack([k_hMpc] + [Tk_rescaled[sp] for sp in species])
np.savetxt(f"data/classy_Tk_z{int(z)}.dat", data, header=header)

print(f"Saved Tk table with {len(k_hMpc)} k-modes at z={z}")

# after your existing get_transfer() call
k_hMpc = np.array(transfers["k (h/Mpc)"])
h = cosmo.h()
z = 90.0

# P(k) in (Mpc/h)^3, same k-grid
Pk = np.array([cosmo.pk(k * h, z) * h**3 for
               k in k_hMpc])

data = np.column_stack([k_hMpc] + [Pk])
np.savetxt(f"data/classy_Pk_z{int(z)}.dat", data)

print(f"Saved Tk table with {len(k_hMpc)} k-modes at z={z}")

subprocess.run(
    ["mpiexec", "-np", "1", "./N-GenIC", "ngenic.param"],
    cwd="/home/jbb/Documents/simsilun_backreaction/initial_conditions/S-GenIC")
