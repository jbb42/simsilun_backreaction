import numpy as np
from matplotlib import pyplot as plt
import subprocess
from LTB_model import evolve_LTB as evolve_LTB

z_i = 80    # Initial redshift
z_f = 0.1     # Final redshift
H_0 = 70    # Hubble constant [km/s/Mpc]

def plot_universe(grid, coords, posx, posy, title):
    ax = axes[posy, posx]  # upper-left subplot

    im = ax.imshow(grid[:, :], origin='lower',
                   extent=[coords.min(), coords.max(), coords.min(), coords.max()])
    fig.colorbar(im, ax=ax, label=r'$\rho / \rho_{\mathrm{EdS}}$')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
grid_i, coords_i = evolve_LTB(0, z_i, z_f, H_0)
plot_universe(grid_i, coords_i, 1, 0, "LTB density slice at z="+str(z_i))

grid_v = grid_i.reshape(64*64)
np.savetxt("grid", grid_v)
np.savetxt("simsilun/grid", grid_v)

grid_f, coords_f = evolve_LTB(1, z_i, z_f, H_0)
plot_universe(grid_f, coords_f, 0, 0, "LTB density slice at z="+str(z_f))

subprocess.run(["simsilun/simsilun", str(z_i), str(z_f), str(H_0)], check=True)
data = np.loadtxt("density")

grid_s = np.reshape(data[:,1], [64,64])
plot_universe(grid_s, coords_i, 0, 1, "simsilun density slice at z="+str(z_f))

grid_d = grid_f-grid_s
plot_universe(grid_d, coords_i, 1, 1, "difference at z="+str(z_f))
plt.savefig("density.pdf")
plt.show()

