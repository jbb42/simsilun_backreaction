import numpy as np
from matplotlib import pyplot as plt
import subprocess
from LTB_model import evolve_LTB as evolve_LTB

z_i = 1100    # Initial redshift
z_f = 0    # Final redshift
H_0 = 70    # Hubble constant [km/s/Mpc]
all_ic = True # Use all initial conditions vs calculate from density contrast

def plot_universe(axes, grid, coords, posx, posy, title, lbl=r'$\rho / \rho_{\mathrm{EdS}}$'):
    ax = axes[posy, posx]  # upper-left subplot

    im = ax.imshow(grid[:, :], origin='lower',
                   extent=[coords.min(), coords.max(), coords.min(), coords.max()])
    fig.colorbar(im, ax=ax, label=lbl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()


rho_i, theta_i, sigma_i, weyl_i, coords = evolve_LTB(0, z_i, z_f, H_0)
np.savetxt("rho_i", rho_i.reshape(64*64))
np.savetxt("theta_i", theta_i.reshape(64*64))
np.savetxt("sigma_i", sigma_i.reshape(64*64))
np.savetxt("weyl_i", weyl_i.reshape(64*64))

rho_f, theta_f, sigma_f, weyl_f, _ = evolve_LTB(1, z_i, z_f, H_0)

subprocess.run(["simsilun/simsilun", str(z_i), str(z_f), str(H_0), ".true." if all_ic else ".false."], check=True)
#data_dens = np.loadtxt("density")

params = np.loadtxt("params") #dens(I),expa(I),shea(I),weyl(I)
rho_s = np.reshape(params[:,0], [64,64])
rho_d = rho_f-rho_s

params = np.loadtxt("params") #dens(I),expa(I),shea(I),weyl(I)
theta_s = np.reshape(params[:,1], [64,64])
theta_d = theta_f-theta_s

sigma_s = np.reshape(params[:,2], [64,64])
sigma_d = sigma_f-sigma_s

weyl_s = np.reshape(params[:,3], [64,64])
weyl_d = weyl_f-weyl_s

# Plot density
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plot_universe(axes, rho_i, coords, 1, 0, "LTB density at z="+str(z_i))
plot_universe(axes, rho_f, coords, 0, 0, "LTB density at z="+str(z_f))
plot_universe(axes, rho_s, coords, 0, 1, "Simsilun density at z="+str(z_f))
plot_universe(axes, rho_d, coords, 1, 1, "Difference at z="+str(z_f))
plt.savefig("density.pdf")
plt.show()

# Plot expansion
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plot_universe(axes, theta_i, coords, 1, 0, "LTB expansion at z="+str(z_i),lbl=r'$\theta / \theta_{\mathrm{EdS}}$')
plot_universe(axes, theta_f, coords, 0, 0, "LTB expansion at z="+str(z_f),lbl=r'$\theta / \theta_{\mathrm{EdS}}$')
plot_universe(axes, theta_s, coords, 0, 1, "Simsilun expansion at z="+str(z_f),lbl=r'$\theta / \theta_{\mathrm{EdS}}$')
plot_universe(axes, theta_d, coords, 1, 1, "Difference at z="+str(z_f),lbl=r'$\theta / \theta_{\mathrm{EdS}}$')
plt.savefig("expansion.pdf")
plt.show()

# Plot shear
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plot_universe(axes, sigma_i, coords, 1, 0, "LTB shear at z="+str(z_i),lbl=r'$3\sigma / \theta_{\mathrm{EdS}}$')
plot_universe(axes, sigma_f, coords, 0, 0, "LTB shear at z="+str(z_f),lbl=r'$3\sigma / \theta_{\mathrm{EdS}}$')
plot_universe(axes, sigma_s, coords, 0, 1, "Simsilun shear at z="+str(z_f),lbl=r'$3\sigma / \theta_{\mathrm{EdS}}$')
plot_universe(axes, sigma_d, coords, 1, 1, "Difference at z="+str(z_f),lbl=r'$3\sigma / \theta_{\mathrm{EdS}}$')
plt.savefig("shear.pdf")
plt.show()

# Plot weyl
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plot_universe(axes, weyl_i, coords, 1, 0, "LTB weyl at z="+str(z_i),lbl=r'$\mathcal{W}$')# / \rho_{\mathrm{EdS}}$')
plot_universe(axes, weyl_f, coords, 0, 0, "LTB weyl at z="+str(z_f),lbl=r'$\mathcal{W}$')# / \rho_{\mathrm{EdS}}$')
plot_universe(axes, weyl_s, coords, 0, 1, "Simsilun weyl at z="+str(z_f),lbl=r'$\mathcal{W}$')# / \rho_{\mathrm{EdS}}$')
plot_universe(axes, weyl_d, coords, 1, 1, "Difference at z="+str(z_f),lbl=r'$\mathcal{W}$')# / \rho_{\mathrm{EdS}}$')
plt.savefig("weyl.pdf")
plt.show()