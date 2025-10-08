import numpy as np
import subprocess
from classy import Class
from pygadgetreader import readsnap, readheader

def run_class(dict, z):

    # create instance of the class "Class"
    cosmo = Class()
    # pass input parameters
    cosmo.set(dict)
    # run class
    cosmo.compute()
    # --- extract transfer functions ---
    transfers = cosmo.get_transfer(z)

    # k in h/Mpc
    k_hMpc = np.array(transfers["k (h/Mpc)"])
    h = cosmo.h()  # reduced Hubble
    k_Mpc = k_hMpc * h  # convert to 1/Mpc

    # species you want to save
    species = ["d_cdm", "d_b", "d_g", "d_ur", "d_ncdm", "d_tot"]

    Tk_rescaled = {}
    for sp in species:
        if sp in transfers:          # CLASS actually returned this species
            Ti = np.array(transfers[sp])
            Tk_rescaled[sp] = -Ti / (k_Mpc**2)
        else:                        # species missing -> fill zeros
            print(f"WARNING: species {sp} not found in CLASS output")
            Tk_rescaled[sp] = np.zeros_like(k_Mpc)

    # --- save in CAMB-like format ---
    header = (
        f"k [h/Mpc]  " + "  ".join([f"-T_{sp}/k^2" for sp in species])
    )
    data = np.column_stack([k_hMpc] + [Tk_rescaled[sp] for sp in species])
    np.savetxt(f"./data/ics/tk.dat", data, header=header)

    print(f"Saved Tk table with {len(k_hMpc)} k-modes at z={z}")

    # after your existing get_transfer() call
    k_hMpc = np.array(transfers["k (h/Mpc)"])
    h = cosmo.h()

    # P(k) in (Mpc/h)^3, same k-grid
    Pk = np.array([cosmo.pk(k * h, z) * h**3 for
                   k in k_hMpc])

    data = np.column_stack([k_hMpc] + [Pk])
    np.savetxt(f"./data/ics/pk.dat", data)

    print(f"Saved Tk table with {len(k_hMpc)} k-modes at z={z}")

    param_path = "../ngenic.param"  # relative to S-GenIC folder
    subprocess.run(
        ["mpiexec", "-np", "1", "./N-GenIC", param_path],
        cwd="initial_conditions/S-GenIC",
    )
    return

def dens_contrast(grid_size):
    snapshot = ('./data/ics/ICs')

    # Read positions for each particle type
    pos_dm = readsnap(snapshot, 'pos', 'dm')
    pos_gas = readsnap(snapshot, 'pos', 'gas')

    # Read masses from header
    massarr = readheader(snapshot, 'mass')
    npart = readheader(snapshot, 'npart')
    mass_dm = np.full(npart[1], massarr[1])
    mass_gas = np.full(npart[0], massarr[0])

    # Combine all positions and masses
    positions = np.concatenate([pos_dm, pos_gas], axis=0)
    masses = np.concatenate([mass_dm, mass_gas], axis=0)

    # Grid setup
    box_size = readheader(snapshot, 'boxsize')
    cell_size = box_size / grid_size

    # Normalize positions to grid coordinates
    u = (positions / cell_size) % grid_size
    i = np.floor(u).astype(np.int64)

    # TSC kernel
    def W(d):
        w = np.zeros_like(d)
        m1 = d < 0.5
        m2 = (d >= 0.5) & (d < 1.5)
        w[m1] = 0.75 - d[m1]**2
        w[m2] = 0.5 * (1.5 - d[m2])**2
        return w

    # Initialize density grid
    rho = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    # Apply TSC mass assignment
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx = (i[:, 0] + dx + 0.5)
                cy = (i[:, 1] + dy + 0.5)
                cz = (i[:, 2] + dz + 0.5)
                wx = W(np.abs(u[:, 0] - cx))
                wy = W(np.abs(u[:, 1] - cy))
                wz = W(np.abs(u[:, 2] - cz))
                w = wx * wy * wz
                ix = (i[:, 0] + dx) % grid_size
                iy = (i[:, 1] + dy) % grid_size
                iz = (i[:, 2] + dz) % grid_size
                np.add.at(rho, (ix, iy, iz), masses * w)

    # Normalize by cell volume
    cell_volume = cell_size ** 3
    rho /= cell_volume

    # Compute mean density and density contrast
    rho_bar = masses.sum() / (box_size ** 3)
    delta = rho / rho_bar - 1.0
    np.save("./data/ics/delta.npy", delta)